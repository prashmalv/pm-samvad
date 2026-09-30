"""Import real ISL sign videos (e.g. from the ISLRTC dictionary) as stitchable clips.

For each source video this:
  1. trims the idle "hands down" time at the start/end (motion detection),
  2. removes the ISLRTC overlays (topic picture + label top-left, logo bottom-right),
  3. normalises to 1280x720 @ 25fps H.264, no audio,
  4. saves it as clips/<GLOSS>.mp4 and registers it in the SQLite DB.

    python scripts/import_clip.py path\\to\\Hello.mp4 --gloss HELLO
    python scripts/import_clip.py path\\to\\folder          # gloss taken from each filename
    python scripts/import_clip.py video.mp4 --gloss X --keep-overlays --no-trim
    python scripts/import_clip.py path\\to\\letters --alphabet   # A.mp4..Z.mp4 -> fingerspelling clips

Filename -> gloss: "Thank_You.mp4" -> THANK-YOU, "adipose cells.mp4" -> ADIPOSE-CELLS.
"""
import argparse
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ALPHABET_DIR, CLIPS_DIR, DB_PATH, OUTPUT_FPS, OUTPUT_HEIGHT, OUTPUT_WIDTH  # noqa: E402
from app.video import _ffmpeg_exe  # noqa: E402

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

# ISLRTC overlay boxes (x, y, w, h) measured on a 1920x1080 frame; scaled for other sizes.
ISLRTC_OVERLAYS = [(40, 100, 570, 530), (1555, 665, 275, 345)]
# The A-Z dictionary folders on ISLRTC's Google Drive: word label top-left, logo lower in the corner.
ISLRTC_DICT_OVERLAYS = [(40, 100, 570, 530), (1650, 800, 265, 275)]


def gloss_from_filename(path: Path) -> str:
    return re.sub(r"[^A-Z0-9]+", "-", path.stem.upper()).strip("-")


def probe(path: Path) -> tuple[int, int, float]:
    """Return (width, height, duration) by parsing ffmpeg's banner (no ffprobe needed)."""
    err = subprocess.run([_ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    size = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", err)
    dur = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    if not size or not dur:
        raise RuntimeError(f"Could not read video info from {path}")
    h, m, s = dur.groups()
    return int(size[1]), int(size[2]), int(h) * 3600 + int(m) * 60 + float(s)


def detect_signing(path: Path, pad: float = 0.3) -> tuple[float, float]:
    """Find the span where the signer is moving, using frame differences on a tiny greyscale copy."""
    w, h, fps = 160, 90, 25
    raw = subprocess.run(
        [_ffmpeg_exe(), "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale={w}:{h}",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    ).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.int16)
    duration = len(frames) / fps
    if len(frames) < fps:  # under a second: nothing to trim
        return 0.0, duration
    motion = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    motion = np.convolve(motion, np.ones(5) / 5, mode="same")  # smooth over 0.2 s
    idle = np.percentile(motion, 10)  # stillest moments = resting pose
    active = np.flatnonzero(motion > max(0.5, idle * 2.5))
    if active.size == 0:
        return 0.0, duration
    start = max(0.0, active[0] / fps - pad)
    end = min(duration, (active[-1] + 1) / fps + pad)
    return start, end


def import_clip(src: Path, gloss: str, trim: bool, remove_overlays: bool, quiet: bool = False,
                letter: bool = False, span: tuple[float, float] | None = None,
                crop: tuple[int, int, int, int] | None = None,
                overlays: list[tuple[int, int, int, int]] = ISLRTC_OVERLAYS,
                pre_filters: list[str] | None = None, paint: Path | None = None) -> Path:
    """letter=True saves to the fingerspelling alphabet folder instead of registering a gloss.

    span: (start, end) seconds to keep, overriding motion-based trimming.
    crop: (x, y, w, h) pixel box of the source frame to keep (e.g. the signer's upper body).
    overlays: boxes (on a 1920x1080 frame) blanked out when remove_overlays is set.
    pre_filters: extra ffmpeg filters applied to the source frame first (e.g. painting out overlays).
    paint: an RGBA image the size of the source frame, laid over every frame before anything else
        (opaque where overlays should be painted out, transparent elsewhere).
    """
    width, height, duration = probe(src)
    if span:
        start, end = max(0.0, span[0]), min(duration, span[1])
    else:
        start, end = detect_signing(src) if trim else (0.0, duration)

    filters = list(pre_filters or [])
    if crop:
        x, y, w, h = crop
        filters.append(f"crop={w}:{h}:{x}:{y}")
    if remove_overlays:
        sx, sy = width / 1920, height / 1080
        for x, y, w, h in overlays:
            filters.append(f"delogo=x={int(x * sx)}:y={int(y * sy)}:w={int(w * sx)}:h={int(h * sy)}")
    filters.append(
        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps={OUTPUT_FPS},format=yuv420p"
    )

    out = (ALPHABET_DIR if letter else CLIPS_DIR) / f"{gloss}.mp4"
    inputs = ["-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", str(src)]
    if paint:
        inputs += ["-i", str(paint)]
        graph = ["-filter_complex", "[0:v][1:v]overlay=0:0:eof_action=repeat," + ",".join(filters) + "[v]", "-map", "[v]"]
    else:
        graph = ["-vf", ",".join(filters)]
    result = subprocess.run(
        [_ffmpeg_exe(), "-y", "-v", "error", *inputs, *graph,
         "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-movflags", "+faststart", str(out)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip()[-600:])

    if letter:
        print(f"  {src.name} -> {out.relative_to(CLIPS_DIR.parent).as_posix()}  [letter {gloss}]  ({end - start:.1f}s)")
        return out
    with sqlite3.connect(DB_PATH, timeout=30) as conn:  # timeout: parallel fetch workers share the DB
        conn.execute("CREATE TABLE IF NOT EXISTS gloss_clips (gloss TEXT PRIMARY KEY, video_path TEXT NOT NULL)")
        conn.execute("INSERT OR REPLACE INTO gloss_clips (gloss, video_path) VALUES (?, ?)", (gloss, f"clips/{out.name}"))

    if quiet:
        return out
    trimmed = f"trimmed {start:.1f}-{end:.1f}s of {duration:.1f}s" if trim else f"{duration:.1f}s"
    print(f"  {src.name} -> clips/{out.name}  [{gloss}]  ({trimmed})")
    if end - start > 6:
        print(f"    note: {end - start:.0f}s is long for one sign - this may be an explanation video, not a word sign.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="a video file or a folder of videos")
    parser.add_argument("--gloss", help="gloss for a single file (default: derived from filename)")
    parser.add_argument("--no-trim", action="store_true", help="keep the full video length")
    parser.add_argument("--keep-overlays", action="store_true", help="don't blank out ISLRTC picture/label/logo")
    parser.add_argument("--alphabet", action="store_true",
                        help="import fingerspelling letters (files named A.mp4..Z.mp4, 0.mp4..9.mp4)")
    args = parser.parse_args()

    if args.source.is_dir():
        if args.gloss:
            parser.error("--gloss only works with a single file")
        files = sorted(p for p in args.source.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    elif args.source.is_file():
        files = [args.source]
    else:
        parser.error(f"{args.source} not found")

    print(f"Importing {len(files)} clip(s):")
    failed = 0
    for f in files:
        gloss = (args.gloss or gloss_from_filename(f)).upper()
        if args.alphabet and not re.fullmatch(r"[A-Z0-9]", gloss):
            print(f"  SKIPPED {f.name}: alphabet files must be named after one letter or digit, e.g. A.mp4")
            continue
        try:
            import_clip(f, gloss, trim=not args.no_trim, remove_overlays=not args.keep_overlays, letter=args.alphabet)
        except Exception as exc:  # noqa: BLE001 - keep going with the rest of the folder
            failed += 1
            print(f"  FAILED {f.name}: {exc}")
    print(f"Done: {len(files) - failed} imported, {failed} failed.")


if __name__ == "__main__":
    main()
