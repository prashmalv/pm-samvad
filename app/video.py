"""Stitch ISL clips into one MP4 (H.264, yuv420p, 1280x720 @ 25fps, letterboxed, no audio).

Uses moviepy when available; falls back to ffmpeg via subprocess (system ffmpeg, or the
binary bundled with imageio-ffmpeg).
"""
import logging
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from app.config import OUTPUT_FPS, OUTPUT_HEIGHT, OUTPUT_WIDTH, VIDEO_BACKEND

log = logging.getLogger(__name__)

W, H, FPS = OUTPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_FPS
_BROWSER_FLAGS = ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]


def _ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("ffmpeg not found: install it or `pip install imageio-ffmpeg`.") from exc


Item = Path | tuple[Path, float]  # a clip, or (clip, speed factor) e.g. faster fingerspelled letters


def _items(items: list[Item]) -> list[tuple[Path, float]]:
    return [(i, 1.0) if isinstance(i, Path) else (Path(i[0]), float(i[1])) for i in items]


@lru_cache(maxsize=4096)
def _duration_cached(path: str, mtime: float) -> float:
    err = subprocess.run([_ffmpeg_exe(), "-hide_banner", "-i", path], capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else 0.0


def duration(path: Path) -> float:
    """Clip length in seconds, parsed from ffmpeg's banner (no ffprobe needed); cached per file version."""
    return _duration_cached(str(path), path.stat().st_mtime)


def _stitch_moviepy(items: list[tuple[Path, float]], out: Path) -> None:
    from moviepy import CompositeVideoClip, VideoFileClip, concatenate_videoclips

    sources, framed = [], []
    try:
        for p, speed in items:
            clip = VideoFileClip(str(p), audio=False)
            sources.append(clip)
            scale = min(W / clip.w, H / clip.h)
            clip = clip.resized(scale) if abs(scale - 1) > 1e-3 else clip
            if abs(speed - 1) > 1e-3:
                clip = clip.with_speed_scaled(speed)
            framed.append(CompositeVideoClip([clip.with_position("center")], size=(W, H), bg_color=(0, 0, 0)))
        final = concatenate_videoclips(framed, method="chain")
        final.write_videofile(
            str(out), fps=FPS, codec="libx264", audio=False, preset="veryfast",
            ffmpeg_params=_BROWSER_FLAGS, logger=None,
        )
        final.close()
    finally:
        for c in framed + sources:
            c.close()


def _stitch_ffmpeg(items: list[tuple[Path, float]], out: Path) -> None:
    cmd = [_ffmpeg_exe(), "-y", "-loglevel", "error"]
    for p, _ in items:
        cmd += ["-i", str(p)]
    norm = (
        f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps={FPS},format=yuv420p"
    )
    parts = [
        f"[{i}:v:0]" + (f"setpts=PTS/{speed:g}," if abs(speed - 1) > 1e-3 else "") + f"{norm}[v{i}]"
        for i, (_, speed) in enumerate(items)
    ]
    concat = "".join(f"[v{i}]" for i in range(len(items))) + f"concat=n={len(items)}:v=1:a=0[out]"
    cmd += [
        "-filter_complex", ";".join(parts + [concat]),
        "-map", "[out]", "-an", "-c:v", "libx264", "-preset", "veryfast", *_BROWSER_FLAGS, str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[-800:]}")


def stitch(clips: list[Item], out: Path) -> Path:
    """Concatenate clips in order into `out`. Items may be (path, speed). Raises ValueError if empty."""
    items = _items(clips)
    if not items:
        raise ValueError("No clips to stitch.")
    if VIDEO_BACKEND in ("auto", "moviepy"):
        try:
            _stitch_moviepy(items, out)
            log.info("Stitched %d clips with moviepy -> %s", len(items), out.name)
            return out
        except ImportError:
            if VIDEO_BACKEND == "moviepy":
                raise
            log.warning("moviepy not installed - falling back to ffmpeg.")
        except Exception:
            if VIDEO_BACKEND == "moviepy":
                raise
            log.exception("moviepy stitching failed - falling back to ffmpeg.")
    _stitch_ffmpeg(items, out)
    log.info("Stitched %d clips with ffmpeg -> %s", len(items), out.name)
    return out
