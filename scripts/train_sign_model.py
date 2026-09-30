"""Train the isolated-sign recogniser on INCLUDE landmarks and export it to ONNX.

    python scripts/train_sign_model.py                  # uses whatever landmarks are extracted
    python scripts/train_sign_model.py --epochs 150 --min-samples 5

Input : data/include_landmarks/**.npz  (from scripts/include_extract.py)
        data/include_meta/{train,val,test}.parquet  (official INCLUDE split)
Output: models/sign/sign_model.onnx, models/sign/labels.json, models/sign/report.json

Needs the training extras:  pip install -r requirements-train.txt
"""
import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BASE_DIR, DATA_DIR  # noqa: E402
from app.sign_features import FEATURE_DIM, SEQ_LEN, featurize, flip_raw  # noqa: E402

LANDMARKS = DATA_DIR / "include_landmarks"
META = DATA_DIR / "include_meta"
OUT = BASE_DIR / "models" / "sign"


def gloss_from_label(label: str) -> str:
    """'40. I' -> 'I', '12. Good Morning' -> 'GOOD-MORNING', 'T-Shirt' -> 'T-SHIRT'."""
    name = re.sub(r"^\s*\d+\.\s*", "", label)
    return re.sub(r"[^A-Z0-9]+", "-", name.upper()).strip("-")


def load_split(split: str) -> list[tuple[str, np.ndarray]]:
    rows = pq.read_table(META / f"{split}.parquet").to_pylist()
    out = []
    for r in rows:
        f = LANDMARKS / Path(r["video_path"]).with_suffix(".npz")
        if f.exists():
            out.append((gloss_from_label(r["label"]), np.load(f)["raw"].astype(np.float32)))
    return out


# ------------------------------------------------------------------ augmentation (on raw landmarks)
def augment(raw: np.ndarray) -> np.ndarray:
    r = raw.copy()
    if random.random() < 0.5:
        r = flip_raw(r)
    # small affine jitter around the image centre
    ang = math.radians(random.uniform(-10, 10))
    sc = random.uniform(0.88, 1.12)
    rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]], np.float32) * sc
    r = (r - 0.5) @ rot.T + 0.5 + np.random.uniform(-0.05, 0.05, 2).astype(np.float32)
    # temporal crop / speed change
    T = len(r)
    if T > 8:
        keep = random.uniform(0.8, 1.0)
        n = max(6, int(T * keep))
        s = random.randint(0, T - n)
        r = r[s: s + n]
    # simulate detector dropouts
    drop = np.random.rand(len(r)) < 0.08
    r[drop, 13:] = np.nan
    # per-point jitter
    r = r + np.random.normal(0, 0.002, r.shape).astype(np.float32)
    return r


class AugmentedSigns(torch.utils.data.Dataset):
    """Fresh random augmentation every time a sample is drawn (runs in DataLoader workers)."""

    def __init__(self, items, lid):
        self.items, self.lid = items, lid

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        g, raw = self.items[i]
        return torch.from_numpy(featurize(augment(raw))), self.lid[g]


# ------------------------------------------------------------------ model
class SignTransformer(nn.Module):
    def __init__(self, n_classes: int, d: int = 256, layers: int = 3, heads: int = 4, drop: float = 0.25):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(FEATURE_DIM * 2, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop))
        self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, d))
        layer = nn.TransformerEncoderLayer(d, heads, d * 2, drop, batch_first=True, activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(d * 2), nn.Dropout(drop), nn.Linear(d * 2, n_classes))

    def forward(self, x):  # x [B, SEQ_LEN, FEATURE_DIM]
        vel = torch.cat([torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]], 1)
        h = self.inp(torch.cat([x, vel], -1)) + self.pos
        h = self.enc(h)
        return self.head(torch.cat([h.mean(1), h.amax(1)], -1))


def batches(xs, ys, bs, shuffle):
    idx = np.arange(len(xs))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i: i + bs]
        yield torch.from_numpy(np.stack([xs[k] for k in j])), torch.tensor([ys[k] for k in j])


@torch.no_grad()
def evaluate(model, xs, ys, bs=256):
    model.eval()
    top1 = top5 = 0
    for xb, yb in batches(xs, ys, bs, False):
        logits = model(xb)
        top = logits.topk(min(5, logits.shape[1]), 1).indices
        top1 += (top[:, 0] == yb).sum().item()
        top5 += (top == yb[:, None]).any(1).sum().item()
    n = max(len(xs), 1)
    return top1 / n, top5 / n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-samples", type=int, default=4, help="drop classes with fewer training videos")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4, help="parallel augmentation workers")
    ap.add_argument("--out", type=Path, default=OUT, help="output folder (default models/sign)")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    train, val, test = load_split("train"), load_split("val"), load_split("test")
    counts: dict[str, int] = {}
    for g, _ in train:
        counts[g] = counts.get(g, 0) + 1
    labels = sorted(g for g, c in counts.items() if c >= args.min_samples)
    lid = {g: i for i, g in enumerate(labels)}
    train = [(g, r) for g, r in train if g in lid]
    val = [(g, r) for g, r in val if g in lid]
    test = [(g, r) for g, r in test if g in lid]
    print(f"{len(labels)} signs | train {len(train)} | val {len(val)} | test {len(test)}", flush=True)
    if len(labels) < 2:
        sys.exit("Not enough extracted data yet - run scripts/include_extract.py first.")

    val_x = [featurize(r) for _, r in val]; val_y = [lid[g] for g, _ in val]
    test_x = [featurize(r) for _, r in test]; test_y = [lid[g] for g, _ in test]

    model = SignTransformer(len(labels))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * (len(train) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    loader = torch.utils.data.DataLoader(AugmentedSigns(train, lid), batch_size=args.batch, shuffle=True,
                                         num_workers=args.workers, persistent_workers=args.workers > 0, drop_last=True)
    best, best_state, t0 = -1.0, None, time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * len(yb)
        if ep % 5 == 0 or ep == args.epochs:
            v1, v5 = evaluate(model, val_x, val_y) if val_x else (0, 0)
            if v1 >= best:
                best, best_state = v1, {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  epoch {ep:3d} | loss {tot / len(train):.3f} | val top1 {v1:.3f} top5 {v5:.3f} | {time.time() - t0:.0f}s", flush=True)

    model.load_state_dict(best_state)
    t1, t5 = evaluate(model, test_x, test_y)
    print(f"TEST: top-1 {t1:.3f}, top-5 {t5:.3f} on {len(test_x)} videos ({len(labels)} signs)")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "labels.json").write_text(json.dumps(labels, indent=0))
    torch.save(model.state_dict(), args.out / "sign_model.pt")  # kept so a failed export never loses a run
    model.eval()
    dummy = torch.zeros(1, SEQ_LEN, FEATURE_DIM)
    torch.onnx.export(model, dummy, str(args.out / "sign_model.onnx"), input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    (args.out / "labels.json").write_text(json.dumps(labels, indent=0))
    report = {"signs": len(labels), "train": len(train), "val": len(val), "test": len(test_x),
              "val_top1": best, "test_top1": t1, "test_top5": t5, "epochs": args.epochs,
              "seq_len": SEQ_LEN, "feature_dim": FEATURE_DIM}
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()
