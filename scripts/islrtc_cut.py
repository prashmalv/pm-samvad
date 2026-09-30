"""Cut the word sign out of long ISLRTC videos, for a person to review before the app uses it.

import_islrtc.py skips videos that are too long for one sign: most show the word sign (sometimes after
fingerspelling it) and then explain its meaning in ISL, in one flow. Where the signer clearly pauses,
this cuts up to two candidate clips from the start of the video - the first signing segments between
pauses - into clips/_review/. Nothing is registered: the review page (/review.html) shows the candidates
and only the clip a reviewer approves is added to the sign library. Videos without a clear pause are
marked "no-cut" and left out.

    python scripts/islrtc_cut.py --from Downloads/islrtc          # cut candidates for all long videos
    python scripts/islrtc_cut.py --from Downloads/islrtc AMPLIFIER  # only these words
"""
import argparse
import json
import sqlite3
import sys
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app.config import CLIPS_DIR, DATA_DIR, DB_PATH  # noqa: E402
from app.sign_features import L_SH, R_SH  # noqa: E402
from scripts.import_clip import import_clip, probe  # noqa: E402
from scripts.import_islrtc import SKIPPED_PATH, clean_signer, local_candidates  # noqa: E402
from scripts.include_extract import landmarks_from_video  # noqa: E402

REVIEW_DIR = CLIPS_DIR / "_review"
REVIEW_PATH = DATA_DIR / "islrtc_review.json"
L_WR, R_WR = 11, 12      # wrists as indices into sign_features.POSE_IDX
HANDS_UP = 0.35          # wrist height above the waist, in shoulder widths, that counts as signing
MOVING, STILL = 1.5, 0.8  # wrist speed (shoulder widths / s) that starts a segment / counts as still
PAUSE_S = 0.4            # hands down or still this long ends a segment
JOIN_S = 0.6             # segments closer than this may be one sign split by a short hold
SIGN_S = (0.6, 3.5)      # a candidate word sign lasts this long
PAD_S = 0.25
MAX_START_S = 12.0       # the word sign comes at the start; later segments are the explanation


def segments(raw: np.ndarray, fps: float) -> list[tuple[float, float]]:
    """Signing segments (start_s, end_s) that end in a pause: hands dropped or held still >= PAUSE_S."""
    with np.errstate(all="ignore"):
        sw = np.nanmedian(np.abs(raw[:, L_SH, 0] - raw[:, R_SH, 0]))
        hip_y = np.nanmedian(raw[:, [L_SH, R_SH], 1]) + 1.6 * sw
        wr = raw[:, [L_WR, R_WR]]
        h = np.nan_to_num(np.nanmax((hip_y - wr[..., 1]) / sw, 1), nan=-1)
        sp = np.nan_to_num(np.nanmax(np.linalg.norm(np.diff(wr, axis=0, prepend=wr[:1]), axis=-1), 1)) / sw * fps
    k = max(1, int(fps / 4))
    sp = np.convolve(sp, np.ones(k) / k, "same")
    still = (h < HANDS_UP) | (sp < STILL)
    n, p = len(sp), int(round(PAUSE_S * fps))
    out, i = [], 0
    while i < n:
        while i < n and not (h[i] >= HANDS_UP and sp[i] > MOVING):
            i += 1
        if i >= n:
            break
        j = i
        while j < n and not still[j:j + p].all():
            j += 1
        if j < n:  # a segment running into the end of the video has no pause after it
            out.append((i / fps, j / fps))
        i = j + p
    return out


def candidate_spans(raw: np.ndarray, fps: float) -> list[tuple[float, float]]:
    """Up to two (start, end) spans, in time order, that may be the word sign."""
    segs = [s for s in segments(raw, fps) if s[0] <= MAX_START_S]
    joined: list[list[float]] = []
    for s, e in segs:
        if joined and s - joined[-1][1] < JOIN_S:
            joined[-1][1] = e
        else:
            joined.append([s, e])
    spans = set(segs[:2]) | {tuple(joined[0])} if joined else set()
    ok = sorted(s for s in spans if SIGN_S[0] <= s[1] - s[0] <= SIGN_S[1])
    return [(s - PAD_S, e + PAD_S) for s, e in ok[:2]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words", nargs="*", help="glosses to cut (default: every video import_islrtc.py found too long)")
    ap.add_argument("--from", dest="source", type=Path, required=True, help="folder with the ISLRTC zips")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cands = local_candidates(args.source)
    skipped = json.loads(SKIPPED_PATH.read_text()) if SKIPPED_PATH.exists() else {}
    review = json.loads(REVIEW_PATH.read_text()) if REVIEW_PATH.exists() else {}
    with sqlite3.connect(DB_PATH) as conn:
        have = {g for (g,) in conn.execute("SELECT gloss FROM gloss_clips")}
    wanted = [w.upper() for w in args.words] or sorted(g for g, why in skipped.items() if "long" in why)
    todo = [g for g in wanted if g in cands and g not in have and (args.words or g not in review)]
    print(f"{len(todo)} long video(s) to cut.")
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    new: dict[str, dict] = {}

    def save() -> None:  # merge into the file: the review page may have recorded decisions meanwhile
        cur = json.loads(REVIEW_PATH.read_text()) if REVIEW_PATH.exists() else {}
        cur.update(new)
        REVIEW_PATH.write_text(json.dumps(cur, indent=0))
        new.clear()

    local, lock = threading.local(), threading.Lock()
    stats = {"cut": 0, "no-cut": 0, "failed": 0}
    t0 = time.time()

    def work(gloss: str) -> None:
        if not hasattr(local, "zips"):
            local.zips = {}
        entry = cands[gloss]
        item = {"source": entry["path"]}
        with tempfile.TemporaryDirectory() as tmp:
            try:
                if "zip" in entry:
                    zf = local.zips.get(entry["zip"]) or local.zips.setdefault(entry["zip"], zipfile.ZipFile(entry["zip"]))
                    src = Path(tmp) / Path(entry["member"]).name
                    with zf.open(entry["member"]) as fin, open(src, "wb") as fout:
                        while chunk := fin.read(1 << 20):
                            fout.write(chunk)
                else:
                    src = entry["file"]
                w, h, dur = probe(src)
                lm = landmarks_from_video(str(src))
                spans = candidate_spans(*lm)
                item["duration"] = round(dur, 1)
                if spans:
                    paint = Path(tmp) / "paint.png"
                    clean_signer(src, w, h, paint, lm=lm)
                    item["clips"] = []
                    for n, span in enumerate(spans, 1):
                        out = import_clip(src, gloss, trim=False, remove_overlays=False, span=span, paint=paint,
                                          out_path=REVIEW_DIR / f"{gloss}_{n}.mp4")
                        item["clips"].append({"file": f"clips/_review/{out.name}", "span": [round(x, 2) for x in span]})
                    item["status"], key = "pending", "cut"
                else:
                    item["status"], key = "no-cut", "no-cut"
            except Exception as exc:  # noqa: BLE001 - one bad video must not stop the batch
                key = "failed"
                print(f"  FAILED {gloss}: {str(exc)[:200]}", flush=True)
        with lock:
            stats[key] += 1
            if key != "failed":
                new[gloss] = item
            n = sum(stats.values())
            if n % 25 == 0 or n == len(todo):
                rate = n / max(time.time() - t0, 1)
                print(f"  [{n}/{len(todo)}] {stats['cut']} with candidates, {stats['no-cut']} no clear pause, "
                      f"{stats['failed']} failed | ETA {(len(todo) - n) / rate / 60:.0f} min", flush=True)
                save()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, todo))
    save()
    print(f"Done: {stats['cut']} words have candidate clips to review at http://127.0.0.1:8000/review.html, "
          f"{stats['no-cut']} had no clear pause, {stats['failed']} failed.")


if __name__ == "__main__":
    main()
