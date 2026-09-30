"""Add sign clips from the INCLUDE dataset (AI4Bharat, CC BY 4.0) for words our library lacks.

INCLUDE has 263 everyday ISL words, each filmed ~15-20 times by Deaf signers. For every word that
has no clip yet, this picks the best take using the landmarks already extracted for Phase 2
(data/include_landmarks, from scripts/include_extract.py), streams only that one video out of the
Zenodo zip, trims it to the sign, crops it to the signer's upper body, and imports it as
clips/<GLOSS>.mp4 (registered in the DB, so it joins the LLM vocabulary).

    python scripts/import_include.py --plan              # show what would be imported, download nothing
    python scripts/import_include.py                     # import every INCLUDE word we don't have
    python scripts/import_include.py HOSPITAL TOMORROW   # only these words
    python scripts/import_include.py --replace WATER     # also replace a word that already has a clip

Credit: Sridhar et al., "INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition",
ACM MM 2020 (AI4Bharat) - CC BY 4.0.
"""
import argparse
import re
import sqlite3
import sys
import tempfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DATA_DIR, DB_PATH  # noqa: E402
from app.sign_features import L_SH, LEFT, N_POSE, RIGHT, R_SH, mask_resting_hands  # noqa: E402
from scripts.import_clip import import_clip, probe  # noqa: E402
from scripts.include_extract import OUT_DIR as LANDMARKS, HttpFile, _client, list_zips  # noqa: E402

META = DATA_DIR / "include_meta"  # official INCLUDE split files (same as scripts/train_sign_model.py)


def gloss_from_label(label: str) -> str:
    """'40. I' -> 'I', '12. Good Morning' -> 'GOOD-MORNING' (same as scripts/train_sign_model.py)."""
    name = re.sub(r"^\s*\d+\.\s*", "", label)
    return re.sub(r"[^A-Z0-9]+", "-", name.upper()).strip("-")

# Extra short names registered for an imported compound gloss, when that name is still free.
ALIASES = {
    "STORE-OR-SHOP": ["SHOP", "STORE"],
    "STREET-OR-ROAD": ["ROAD", "STREET"],
    "CELL-PHONE": ["PHONE", "MOBILE"],
    "TELEVISION": ["TV"],
    "BIG-LARGE": ["BIG", "LARGE"],
    "SMALL-LITTLE": ["SMALL", "LITTLE"],
    "TRAIN-TICKET": [],
    "CLOTHING": ["CLOTHES"],
}
PAD_S = 0.25  # seconds kept before/after the signing span


def db_glosses() -> set[str]:
    with sqlite3.connect(DB_PATH) as conn:
        return {g for (g,) in conn.execute("SELECT gloss FROM gloss_clips")}


def include_videos() -> dict[str, list[str]]:
    """gloss -> INCLUDE video paths (all splits), e.g. 'Places/24. School/MVI_3290.MOV'.

    A few metadata rows are mislabelled (label 'Second' pointing into the 'quiet' folder, 'Store or
    Shop' into 'Father'), so a video only counts when its folder name agrees with its label."""
    out: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        for r in pq.read_table(META / f"{split}.parquet").to_pylist():
            gloss = gloss_from_label(r["label"])
            if gloss_from_label(Path(r["video_path"]).parent.name) == gloss:
                out.setdefault(gloss, []).append(r["video_path"])
    return {g: sorted(set(v)) for g, v in out.items()}


# ------------------------------------------------------------------ choosing the best take
def analyse(video_path: str) -> dict | None:
    """Score one take from its landmarks: how reliably the hands were tracked, and where the signer is."""
    f = LANDMARKS / Path(video_path).with_suffix(".npz")
    if not f.exists():
        return None
    d = np.load(f)
    raw, fps = d["raw"].astype(np.float32), float(d["fps"])
    masked = mask_resting_hands(raw)
    hand = ~np.isnan(masked[:, N_POSE:, 0]).all(1)            # a hand is raised in this frame
    idx = np.flatnonzero(hand)
    if idx.size < 4:
        return None
    a, b = idx[0], idx[-1]
    span = masked[a:b + 1]
    both = ~np.isnan(span[:, LEFT, 0]).all(1) | ~np.isnan(span[:, RIGHT, 0]).all(1)
    sh = raw[:, [L_SH, R_SH]]
    if np.isnan(sh).all():
        return None
    return {
        "path": video_path,
        "start": a / fps, "end": (b + 1) / fps, "dur": (b + 1 - a) / fps,
        "hand_rate": float(both.mean()),                      # tracked hands while signing
        "pose_rate": float((~np.isnan(sh[:, 0, 0])).mean()),
        "raw": raw, "masked": masked, "active": slice(a, b + 1),
    }


def pick_best(takes: list[dict]) -> dict:
    """Most reliably tracked take with a typical length (avoids cut-off or dawdling takes)."""
    med = float(np.median([t["dur"] for t in takes]))
    return max(takes, key=lambda t: 2 * t["hand_rate"] + t["pose_rate"] - abs(np.log(max(t["dur"], .1) / med)))


def upper_body_crop(take: dict, width: int, height: int) -> tuple[int, int, int, int]:
    """16:9 pixel box around head, shoulders and everywhere the hands went while signing."""
    raw = take["raw"]
    px = raw * np.array([width, height], np.float32)          # normalised -> pixels
    sh = px[:, [L_SH, R_SH]]
    mid = np.nanmedian(sh.mean(1), 0)
    sw = float(np.nanmedian(np.linalg.norm(sh[:, 0] - sh[:, 1], axis=1)))
    nose_y = float(np.nanmedian(px[:, 0, 1]))
    # only raised hands: a hand hanging at rest would stretch the box down to the knees
    hands = (take["masked"][take["active"], N_POSE:] * np.array([width, height], np.float32)).reshape(-1, 2)
    hands = hands[~np.isnan(hands).any(1)]
    x0, x1 = mid[0] - 1.3 * sw, mid[0] + 1.3 * sw
    y0, y1 = nose_y - 0.9 * sw, mid[1] + 1.9 * sw               # top of head .. waist
    if len(hands):
        x0, x1 = min(x0, hands[:, 0].min()), max(x1, hands[:, 0].max())
        y0, y1 = min(y0, hands[:, 1].min()), max(y1, hands[:, 1].max())
    m = 0.08 * sw
    x0, x1, y0, y1 = x0 - m, x1 + m, y0 - m, y1 + m
    w, h = x1 - x0, y1 - y0
    if w / h < 16 / 9:                                         # widen (or heighten) to 16:9 around the centre
        cx = (x0 + x1) / 2; w = h * 16 / 9; x0 = cx - w / 2
    else:
        cy = (y0 + y1) / 2; h = w * 9 / 16; y0 = cy - h / 2
    w, h = min(w, width), min(h, height)
    x0 = min(max(0.0, x0), width - w)
    y0 = min(max(0.0, y0), height - h)
    even = lambda v: int(v) // 2 * 2                           # libx264 needs even sizes
    return even(x0), even(y0), even(w), even(h)


# ------------------------------------------------------------------ streaming one video out of a zip
class ZipIndex:
    """Which Zenodo zip holds each video; each worker thread keeps its own open zips."""

    def __init__(self, categories: set[str]):
        zips = list_zips()
        self.where: dict[str, tuple[str, int]] = {}
        client = _client()
        for name, z in sorted(zips.items()):
            if name.rsplit("_", 1)[0] not in categories and name.removesuffix(".zip") not in categories:
                continue
            for info in zipfile.ZipFile(HttpFile(z["url"], z["size"], client, block=1 << 20)).infolist():
                self.where[info.filename] = (z["url"], z["size"])
        self._local = threading.local()

    def extract(self, member: str, dest: Path) -> None:
        url, size = self.where[member]
        loc = self._local
        if not hasattr(loc, "client"):
            loc.client, loc.zips = _client(), {}
        zf = loc.zips.get(url)
        if zf is None:
            zf = loc.zips[url] = zipfile.ZipFile(HttpFile(url, size, loc.client, block=256 << 10))
            zf.fp.block = 4 << 20                              # small reads for the directory, big for video
        with zf.open(member) as src, open(dest, "wb") as out:
            while chunk := src.read(4 << 20):
                out.write(chunk)


def register_aliases(gloss: str, have: set[str]) -> list[str]:
    added = [a for a in ALIASES.get(gloss, []) if a not in have]
    if added:
        with sqlite3.connect(DB_PATH, timeout=30) as conn:
            conn.executemany("INSERT OR IGNORE INTO gloss_clips (gloss, video_path) VALUES (?, ?)",
                             [(a, f"clips/{gloss}.mp4") for a in added])
    return added


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words", nargs="*", help="INCLUDE glosses to import (default: every one we lack)")
    ap.add_argument("--plan", action="store_true", help="only show what would be imported")
    ap.add_argument("--replace", action="store_true", help="also replace glosses that already have a clip")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    videos = include_videos()
    have = db_glosses()
    wanted = [w.upper() for w in args.words] if args.words else sorted(videos)
    unknown = [w for w in wanted if w not in videos]
    todo = [w for w in wanted if w in videos and (args.replace or w not in have)]
    if unknown:
        print(f"Not in INCLUDE: {' '.join(unknown)}")
    print(f"{len(todo)} word(s) to import from INCLUDE ({len(videos)} words in the dataset, "
          f"{sum(w in have for w in videos)} already in our library).")

    plan = []
    for g in todo:
        takes = [t for t in map(analyse, videos[g]) if t]
        if not takes:
            print(f"  SKIP {g}: no usable landmarks (run scripts/include_extract.py)")
            continue
        best = pick_best(takes)
        plan.append((g, best))
        if args.plan:
            print(f"  {g:<18} {best['path']:<48} {best['dur']:.1f}s  hands tracked {best['hand_rate']:.0%}  ({len(takes)} takes)")
    if args.plan or not plan:
        return

    print("Reading the Zenodo zip directories...", flush=True)
    zips = ZipIndex({p["path"].split("/")[0] for _, p in plan})
    done, failed, lock = [], [], threading.Lock()

    def work(item):
        g, take = item
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / Path(take["path"]).name
            try:
                zips.extract(take["path"], src)
                w, h, _ = probe(src)
                import_clip(src, g, trim=False, remove_overlays=False, quiet=True,
                            span=(take["start"] - PAD_S, take["end"] + PAD_S), crop=upper_body_crop(take, w, h))
                with lock:
                    aliases = register_aliases(g, have)
                    have.update([g, *aliases])
                    done.append(g)
                    extra = f"  (+ {' '.join(aliases)})" if aliases else ""
                    print(f"  [{len(done) + len(failed)}/{len(plan)}] {g}{extra}", flush=True)
            except Exception as exc:  # noqa: BLE001 - one bad take must not stop the batch
                with lock:
                    failed.append(g)
                    print(f"  FAILED {g}: {exc}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, plan))
    print(f"Done: {len(done)} imported, {len(failed)} failed.")
    if failed:
        print(f"  Re-run to retry: python scripts/import_include.py {' '.join(failed)}")


if __name__ == "__main__":
    main()
