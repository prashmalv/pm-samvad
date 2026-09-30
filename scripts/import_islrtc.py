"""Add sign clips from the ISLRTC Indian Sign Language Dictionary (Govt. of India) for words we lack.

ISLRTC shares its dictionary on Google Drive. The A-Z folders hold ~10,800 short word videos on a flat
grey background, with a word label, sometimes a topic picture, and a logo placed differently from video
to video. This imports the single-word signs our library doesn't have yet, plus a few useful phrases
(WHAT-TIME, HOW-MANY ...). The other Drive folders (New 2500, MHSL, NCERT) are 12-60 s explainer
videos, not word signs, and are skipped. For each video, MediaPipe finds the signer: the clip is trimmed
to the signing, everything outside the signer's column is painted background grey (removing label,
picture and logo), and the result is normalised and registered.

    python scripts/import_islrtc.py index            # one-time: list the Drive folders -> data/islrtc_index.json
    python scripts/import_islrtc.py --plan           # show what would be imported
    python scripts/import_islrtc.py                  # import (resumable: re-run to continue / retry)
    python scripts/import_islrtc.py PLEASE TEA NO    # only these words
    python scripts/import_islrtc.py --from Downloads/islrtc   # from Drive folders you downloaded yourself
                                                     # (zips or unzipped A..Z folders; no Drive quota)

Terms (islrtc.nic.in/faq): free for research, teaching and ISL technology; not to be resold or used for
profit; credit "Indian Sign Language Research and Training Centre, DEPwD, MoSJE, Govt. of India".
"""
import argparse
import html
import json
import re
import sqlite3
import sys
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app import video  # noqa: E402
from app.config import CLIPS_DIR, DATA_DIR, DB_PATH  # noqa: E402
from app.sign_features import L_SH, N_POSE, R_SH, mask_resting_hands  # noqa: E402
from scripts.import_clip import import_clip, probe  # noqa: E402
from scripts.include_extract import landmarks_from_video  # noqa: E402

ROOT_FOLDER = "1U-Pr4r1-cupgNOOq9NH_uTsQnPSVEKco"  # "ISL Dictionary", linked from islrtc.nic.in/faq
INDEX_PATH = DATA_DIR / "islrtc_index.json"
SKIPPED_PATH = DATA_DIR / "islrtc_skipped.json"   # too long / unusable, so re-runs don't re-download them
MAX_SOURCE_S = 15.0   # longer source videos are explanations, not word signs
MAX_CLIP_S = 8.0      # after trimming, a word sign should be well under this

# Multi-word entries worth having as one sign (the rest are mostly idioms and technical terms).
PHRASES = (
    "CHECK-IN CHECK-OUT KEY-CARD HOW-MANY WHAT-TIME SEE-YOU-TOMORROW EARLY-MORNING FAST-FOOD "
    "CLOSE-FRIEND I-UNDERSTAND I-KNOW BYE-GOODBYE ROOM-SERVICE FIRST-AID BLOOD-PRESSURE HEART-ATTACK "
    "BLOOD-TEST X-RAY MY-NAME-IS THANK-YOU-VERY-MUCH GOOD-LUCK HAPPY-BIRTHDAY NEXT-WEEK LAST-WEEK "
    "HOT-WATER COLD-WATER DRINKING-WATER BED-SHEET"
).split()

ENTRY = re.compile(r'<div class="flip-entry" id="entry-([^"]+)".*?flip-entry-title">([^<]*)<', re.S)


# ------------------------------------------------------------------ index (folder listing only, no videos)
def build_index() -> None:
    client = httpx.Client(follow_redirects=True, timeout=60)

    def listing(fid: str) -> list[tuple[str, str, bool]]:
        page = client.get(f"https://drive.google.com/embeddedfolderview?id={fid}").text
        return [(m[1], html.unescape(m[2]).strip(), f"/drive/folders/{m[1]}" in page) for m in ENTRY.finditer(page)]

    files, seen, frontier = [], set(), [(ROOT_FOLDER, "")]
    while frontier:
        with ThreadPoolExecutor(8) as pool:
            results = list(pool.map(lambda f: (f, listing(f[0])), frontier))
        frontier = []
        for (_, path), entries in results:
            for eid, title, is_folder in entries:
                if eid in seen:
                    continue
                seen.add(eid)
                p = f"{path}/{title}" if path else title
                if is_folder:
                    frontier.append((eid, p))
                else:
                    files.append({"id": eid, "path": p})
        print(f"  {len(files)} files, {len(frontier)} folders to read", flush=True)
    INDEX_PATH.write_text(json.dumps(files, indent=0), encoding="utf-8")
    print(f"Indexed {len(files)} files -> {INDEX_PATH.name}")


def gloss_of(path: str) -> str:
    """'T/Thank_You.mp4' -> 'THANK-YOU'."""
    name = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return re.sub(r"[^A-Z0-9]+", "-", name.upper()).strip("-")


def candidates() -> dict[str, dict]:
    """gloss -> index entry, from the A-Z word folders only; explanation videos and 2nd/3rd variants skipped."""
    if not INDEX_PATH.exists():
        sys.exit("No index yet. Run: python scripts/import_islrtc.py index")
    out: dict[str, dict] = {}
    for f in json.loads(INDEX_PATH.read_text(encoding="utf-8")):
        folder, _, name = f["path"].partition("/")
        if not re.fullmatch(r"[A-Z]", folder) or "/" in name:
            continue
        if re.search(r"explanation|\((sign|sing)[ _]?\d\)", name, re.I):
            continue
        out.setdefault(gloss_of(f["path"]), f)
    return out


VIDEO_EXTS = (".mp4", ".mpg", ".mpeg", ".mov", ".avi", ".mkv", ".webm", ".m4v")


def _usable(folder: str, name: str) -> bool:
    """Same rules as candidates(): an A-Z word folder, not an explanation or a 2nd/3rd variant."""
    return bool(re.fullmatch(r"[A-Z]", folder)) and not re.search(r"explanation|\((sign|sing)[ _]?\d\)", name, re.I)


def local_candidates(root: Path) -> dict[str, dict]:
    """gloss -> {"zip", "member"} or {"file"} for ISLRTC videos in `root`: zips downloaded from Google
    Drive (Drive names them e.g. ISL Dictionary-2026...-001.zip) and/or already unzipped A..Z folders."""
    out: dict[str, dict] = {}
    for z in sorted(root.rglob("*.zip")):
        try:
            names = zipfile.ZipFile(z).namelist()
        except zipfile.BadZipFile:
            print(f"  not a readable zip, skipped: {z.name}")
            continue
        for m in names:
            parts = m.replace("\\", "/").split("/")
            if len(parts) >= 2 and m.lower().endswith(VIDEO_EXTS) and _usable(parts[-2], parts[-1]):
                out.setdefault(gloss_of(parts[-1]), {"zip": z, "member": m, "path": f"{parts[-2]}/{parts[-1]}"})
    for f in sorted(root.rglob("*")):
        if f.suffix.lower() in VIDEO_EXTS and _usable(f.parent.name, f.name):
            out.setdefault(gloss_of(f.name), {"file": f, "path": f"{f.parent.name}/{f.name}"})
    return out


# ------------------------------------------------------------------ cleaning one video
L_EL, R_EL, L_WR, R_WR = 9, 10, 11, 12  # elbows, wrists as indices into sign_features.POSE_IDX
PAD_S = 0.25                            # seconds kept before/after the signing
SMALL_W = 320                           # analysis resolution for background / overlay detection
ARMS = ((L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR))


def sample_frames(src: Path, n: int = 24) -> np.ndarray:
    """~n frames spread over the video, downscaled to SMALL_W wide, BGR uint8 [n, h, w, 3]."""
    import cv2

    cap = cv2.VideoCapture(str(src))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frames = []
    for i in np.linspace(0, total - 1, min(n, total)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, img = cap.read()
        if ok:
            h = round(img.shape[0] * SMALL_W / img.shape[1])
            frames.append(cv2.resize(img, (SMALL_W, h), interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        raise RuntimeError("could not read frames")
    return np.stack(frames)


def signer_mask(raw: np.ndarray, sw: float, w: int, h: int) -> np.ndarray:
    """[h, w] bool: every pixel the signer may cover in any frame - disks around all landmarks, thick
    lines along both arms, the head and the torso down to the bottom of the frame. Generous on purpose:
    anything in here is never painted."""
    import cv2

    mask = np.zeros((h, w), np.uint8)
    r_pose, r_hand, r_arm = (max(2, int(f * sw * w)) for f in (0.25, 0.16, 0.2))
    P = lambda p: (int(p[0] * w), int(p[1] * h))
    for fr in raw[:: max(1, len(raw) // 60)]:
        for n_, p in enumerate(fr):
            if not np.isnan(p).any():
                cv2.circle(mask, P(p), r_pose if n_ < N_POSE else r_hand, 1, -1)
        for i, j in ARMS:
            if not np.isnan(fr[[i, j]]).any():
                cv2.line(mask, P(fr[i]), P(fr[j]), 1, 2 * r_arm)
    sh = np.nanmedian(raw[:, [L_SH, R_SH]], axis=0)                 # [2, 2] normalised
    nose = np.nanmedian(raw[:, 0], axis=0)
    if not np.isnan(nose).any():
        cv2.circle(mask, P(nose), int(0.7 * sw * w), 1, -1)
    if not np.isnan(sh).any():
        x0, x1 = sorted((sh[0, 0] * w, sh[1, 0] * w))
        top = int(min(sh[:, 1]) * h - 0.2 * sw * w)
        cv2.rectangle(mask, (int(x0 - 0.3 * sw * w), top), (int(x1 + 0.3 * sw * w), h), 1, -1)
    return mask.astype(bool)


def clean_signer(src: Path, width: int, height: int, paint_path: Path,
                 lm: tuple[np.ndarray, float] | None = None) -> tuple[float, float] | None:
    """Find the signer with MediaPipe, write an RGBA paint layer to `paint_path` and return the signing
    span in seconds (None if no raised hands were found).

    The paint layer removes the word label, topic picture and logo wherever a video has them, without
    ever touching the signer: it is opaque background grey
      1. everywhere outside the signer's column (body + wherever the hands and elbows go), and
      2. inside the column, on graphics (pixels that differ from the background in most frames and lie
         outside the signer mask).
    lm: landmarks already extracted from `src` (raw, fps), to avoid a second pass."""
    import cv2

    raw, fps = lm or landmarks_from_video(str(src))
    sh = raw[:, [L_SH, R_SH], 0] * width
    if np.isnan(sh).all():
        raise RuntimeError("no signer found")
    cx = float(np.nanmedian(sh.mean(1)))
    sw = float(np.nanmedian(np.abs(sh[:, 0] - sh[:, 1])))
    masked = mask_resting_hands(raw)
    active = np.flatnonzero(~np.isnan(masked[:, N_POSE:, 0]).all(1))
    span, window = None, slice(None)
    if active.size >= 3:
        span = (active[0] / fps - PAD_S, (active[-1] + 1) / fps + PAD_S)
        window = slice(active[0], active[-1] + 1)
    xs = [cx - 1.0 * sw, cx + 1.0 * sw]
    for pts, margin in ((masked[window, N_POSE:, 0], 0.35 * sw), (raw[window][:, [L_EL, R_EL], 0], 0.3 * sw)):
        pts = pts[~np.isnan(pts)] * width
        if pts.size:
            xs += [pts.min() - margin, pts.max() + margin]

    small = sample_frames(src)
    _, sh_, sw_ = small.shape[:3]
    k = sw_ / width                                                  # full-res px -> small px
    a, b = int(max(0, min(xs)) * k), int(min(width, max(xs)) * k)
    body = signer_mask(raw, sw / width, sw_, sh_)

    # background grey: sampled just inside each side of the column, where the signer never is
    def grey(lo: int, hi: int) -> np.ndarray | None:
        strip = np.zeros((sh_, sw_), bool)
        strip[:, max(0, lo): max(0, hi)] = True
        strip &= ~body
        return np.median(small[:, strip].reshape(-1, 3), axis=0) if strip.any() else None

    fallback = np.median(small[:, : max(2, sh_ // 20)].reshape(-1, 3), axis=0)
    left, right = grey(a, a + 5), grey(b - 4, b + 1)
    left = fallback if left is None else left
    right = fallback if right is None else right
    bg = (left + right) / 2

    paint = np.zeros((sh_, sw_), bool)
    paint[:, :a] = paint[:, b:] = True
    diff = np.abs(small.astype(np.int16) - bg.astype(np.int16)).sum(-1) > 45        # [n, h, w]
    paint |= (diff.mean(0) >= 0.6) & ~body                          # graphics shown most of the time
    paint = cv2.dilate(paint.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~body

    colour = np.empty((sh_, sw_, 3), np.float32)                     # left grey -> right grey across the column
    t = np.clip((np.arange(sw_) - a) / max(1, b - a), 0, 1)[None, :, None]
    colour[:] = left * (1 - t) + right * t
    rgba = np.dstack([colour[..., ::-1], paint[..., None] * 255.0]).astype(np.uint8)  # BGR -> RGB, alpha
    full = cv2.resize(rgba, (width, height), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(paint_path), cv2.cvtColor(full, cv2.COLOR_RGBA2BGRA))
    return span


# ------------------------------------------------------------------ download + import
class DriveBlocked(RuntimeError):
    """Google answered with a web page instead of the video (quota / rate limit)."""


def download(client: httpx.Client, file_id: str, dest: Path) -> None:
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
    for attempt in range(4):
        try:
            with client.stream("GET", url) as r:
                r.raise_for_status()
                if r.headers.get("content-type", "").startswith("text/html"):
                    raise DriveBlocked("Google Drive returned a web page (download limit reached?)")
                with open(dest, "wb") as out:
                    for chunk in r.iter_bytes(1 << 20):
                        out.write(chunk)
            return
        except httpx.HTTPError:
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words", nargs="*", help="'index', or glosses to import (default: all new single words + PHRASES)")
    ap.add_argument("--plan", action="store_true", help="only show what would be imported")
    ap.add_argument("--replace", action="store_true", help="re-import named words that already have a clip")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--from", dest="source", type=Path,
                    help="folder with ISLRTC zips / A..Z folders downloaded from Google Drive (instead of downloading)")
    args = ap.parse_args()
    if args.words == ["index"]:
        build_index()
        return

    if args.source:
        if not args.source.is_dir():
            sys.exit(f"{args.source} is not a folder")
        cands = local_candidates(args.source)
        print(f"Found {len(cands)} usable word videos in {args.source}.")
    else:
        cands = candidates()
    with sqlite3.connect(DB_PATH) as conn:
        have = {g for (g,) in conn.execute("SELECT gloss FROM gloss_clips")}
    skipped: dict[str, str] = json.loads(SKIPPED_PATH.read_text()) if SKIPPED_PATH.exists() else {}
    if args.words:
        wanted = [w.upper() for w in args.words]
        missing = [w for w in wanted if w not in cands]
        if missing:
            print(f"Not in the ISLRTC A-Z folders: {' '.join(missing)}")
        todo = [w for w in wanted if w in cands and (args.replace or w not in have)]
    else:
        todo = sorted(g for g in cands if "-" not in g and g not in have) + [p for p in PHRASES if p in cands and p not in have]
        print(f"Phrases not in ISLRTC: {' '.join(p for p in PHRASES if p not in cands)}")
        todo = [g for g in todo if g not in skipped]
    print(f"{len(todo)} sign(s) to import ({len(cands)} usable ISLRTC entries, {len(skipped)} skipped earlier).")
    if args.plan:
        print("  " + " ".join(todo[:400]) + (" ..." if len(todo) > 400 else ""))
        return

    local = threading.local()
    lock = threading.Lock()
    stop = threading.Event()
    stats = {"ok": 0, "long": 0, "failed": 0}
    t0 = time.time()

    def work(gloss: str) -> None:
        if stop.is_set():
            return
        if not hasattr(local, "client"):
            local.client = httpx.Client(follow_redirects=True, timeout=120)
            local.zips = {}
        entry = cands[gloss]
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / Path(entry["path"]).name
            try:
                if "zip" in entry:        # one ZipFile per thread: reads aren't thread-safe on a shared one
                    zf = local.zips.get(entry["zip"]) or local.zips.setdefault(entry["zip"], zipfile.ZipFile(entry["zip"]))
                    with zf.open(entry["member"]) as fin, open(src, "wb") as fout:
                        while chunk := fin.read(1 << 20):
                            fout.write(chunk)
                elif "file" in entry:
                    src = entry["file"]
                else:
                    download(local.client, entry["id"], src)
                w, h, dur = probe(src)
                if dur > MAX_SOURCE_S:
                    raise ValueError(f"source is {dur:.0f}s long")
                span = clean_signer(src, w, h, Path(tmp) / "paint.png")
                out = import_clip(src, gloss, trim=span is None, remove_overlays=False, quiet=True,
                                  span=span, paint=Path(tmp) / "paint.png")
                if (d := video.duration(out)) > MAX_CLIP_S:
                    out.unlink(missing_ok=True)
                    with sqlite3.connect(DB_PATH, timeout=30) as conn:
                        conn.execute("DELETE FROM gloss_clips WHERE gloss = ?", (gloss,))
                    raise ValueError(f"trimmed clip is {d:.0f}s long")
                key = "ok"
            except DriveBlocked as exc:
                stop.set()
                print(f"  STOPPED: {exc}. Wait an hour or so, then re-run to continue.", flush=True)
                return
            except ValueError as exc:  # too long: an explanation, not a word sign
                key = "long"
                with lock:
                    skipped[gloss] = str(exc)
            except Exception as exc:  # noqa: BLE001 - one bad video must not stop the batch
                key = "failed"
                print(f"  FAILED {gloss}: {str(exc)[:200]}", flush=True)
        with lock:
            stats[key] += 1
            n = sum(stats.values())
            if n % 50 == 0 or n == len(todo):
                rate = n / max(time.time() - t0, 1)
                print(f"  [{n}/{len(todo)}] {stats['ok']} ok, {stats['long']} too long, {stats['failed']} failed"
                      f" | ETA {(len(todo) - n) / rate / 60:.0f} min", flush=True)
                SKIPPED_PATH.write_text(json.dumps(skipped, indent=0))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, todo))
    SKIPPED_PATH.write_text(json.dumps(skipped, indent=0))
    print(f"Done: {stats['ok']} imported, {stats['long']} skipped as too long, {stats['failed']} failed.")
    if stats["failed"] or stop.is_set():
        print("  Re-run the same command to continue / retry.")
    print(f"Library: {sum(1 for _ in CLIPS_DIR.glob('*.mp4'))} clip files.")


if __name__ == "__main__":
    main()
