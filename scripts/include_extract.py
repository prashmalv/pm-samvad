"""Extract MediaPipe hand + pose landmarks from the INCLUDE ISL dataset (Zenodo record 4010759).

The dataset is 56.8 GB of zipped videos. Instead of downloading zips, this reads each zip's
directory with HTTP range requests and streams individual videos in parallel. Every video is
turned into a small landmark file and the video itself is thrown away.

    python scripts/include_extract.py                 # all 44 zips, resumable
    python scripts/include_extract.py --only Greetings Pronouns
    python scripts/include_extract.py --workers 4

Output: data/include_landmarks/<Category>/<N. Label>/<video>.npz with
    raw [T, 55, 2] float16 (NaN = missing), fps (sampled)
Dataset licence: CC BY 4.0 - Sridhar et al., "INCLUDE: A Large Scale Dataset for Indian Sign
Language Recognition", ACM MM 2020 (AI4Bharat).
"""
import argparse
import io
import json
import os
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DATA_DIR, MEDIAPIPE_DIR  # noqa: E402

RECORD_API = "https://zenodo.org/api/records/4010759"
OUT_DIR = DATA_DIR / "include_landmarks"
MODELS = MEDIAPIPE_DIR
VIDEO_EXTS = (".mov", ".mp4", ".avi", ".mkv")
TARGET_FPS = 15
MAX_WIDTH = 640


class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (enough for zipfile)."""

    def __init__(self, url: str, size: int, client: httpx.Client, block: int = 4 << 20):
        self.url, self.size, self.client, self.block = url, size, client, block
        self.pos, self._buf_start, self._buf = 0, -1, b""

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        for attempt in range(6):
            try:
                r = self.client.get(self.url, headers={"Range": f"bytes={start}-{end}"})
                if r.status_code == 206:
                    return r.content
                raise httpx.HTTPError(f"status {r.status_code}")
            except httpx.HTTPError:
                if attempt == 5:
                    raise
                time.sleep(2 * (attempt + 1))
        raise AssertionError

    def read(self, n=-1):
        if self.pos >= self.size:
            return b""
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        in_buf = self._buf_start <= self.pos and self.pos + n <= self._buf_start + len(self._buf)
        if not in_buf:
            length = max(n, self.block)
            self._buf_start = self.pos
            self._buf = self._fetch(self.pos, min(self.size, self.pos + length) - 1)
        off = self.pos - self._buf_start
        data = self._buf[off: off + n]
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def _client() -> httpx.Client:
    return httpx.Client(follow_redirects=True, timeout=120)


def list_zips() -> dict[str, dict]:
    files = _client().get(RECORD_API).json()["files"]
    return {f["key"]: {"url": f["links"]["self"], "size": f["size"]} for f in files if f["key"].endswith(".zip")}


# ------------------------------------------------------------------ worker (runs in a subprocess)
_worker: dict = {}


def _init_worker():
    _worker["client"] = _client()
    _worker["zips"] = {}  # url -> ZipFile, so each zip's directory is read once per worker


def _new_landmarkers():
    """Fresh trackers per video, so tracking state never leaks from one video into the next."""
    from mediapipe.tasks.python import BaseOptions, vision

    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS / "hand_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4, min_tracking_confidence=0.4))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS / "pose_landmarker_lite.task")),
        running_mode=vision.RunningMode.VIDEO, num_poses=1))
    return hand, pose


def landmarks_from_video(path: str):
    """Video file -> (raw [T,55,2] float32, sampled fps). Also used by tests / tools."""
    import cv2
    import mediapipe as mp
    import numpy as np

    from app.sign_features import POSE_IDX, build_frame

    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(src_fps / TARGET_FPS))
    frame_ms = 1000.0 * step / src_fps
    frames, i = [], 0
    hand_lm, pose_lm = _new_landmarkers()
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i % step == 0:
                h, w = img.shape[:2]
                if w > MAX_WIDTH:
                    img = cv2.resize(img, (MAX_WIDTH, int(h * MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)
                rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts = int(len(frames) * frame_ms)
                pr = pose_lm.detect_for_video(image, ts)
                hr = hand_lm.detect_for_video(image, ts)
                pose_xy = None
                if pr.pose_landmarks:
                    lm = pr.pose_landmarks[0]
                    pose_xy = np.array([[lm[j].x, lm[j].y] for j in POSE_IDX], dtype=np.float32)
                hands = [np.array([[q.x, q.y] for q in hl], dtype=np.float32) for hl in hr.hand_landmarks]
                frames.append(build_frame(pose_xy, hands))
            i += 1
    finally:
        cap.release()
        hand_lm.close()
        pose_lm.close()
    if not frames:
        raise RuntimeError("no frames decoded")
    return np.stack(frames), src_fps / step


def process_member(url: str, size: int, member: str, out_path: str) -> tuple[str, str]:
    import numpy as np

    zf = _worker["zips"].get(url)
    if zf is None:
        zf = _worker["zips"][url] = zipfile.ZipFile(HttpFile(url, size, _worker["client"], block=256 << 10))
        zf.fp.block = 4 << 20  # small reads for the directory, big reads for video data
    fd, tmp = tempfile.mkstemp(suffix=Path(member).suffix)
    try:
        with os.fdopen(fd, "wb") as f, zf.open(member) as src:
            while chunk := src.read(4 << 20):
                f.write(chunk)
        raw, fps = landmarks_from_video(tmp)
    finally:
        os.unlink(tmp)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, raw=raw.astype(np.float16), fps=np.float32(fps))
    return member, f"{raw.shape[0]} frames"


# ------------------------------------------------------------------ driver
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="categories to process, e.g. Greetings Pronouns")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    zips = list_zips()
    if args.only:
        zips = {k: v for k, v in zips.items() if k.split("_")[0] in args.only or k.rsplit("_", 1)[0] in args.only}
    print(f"{len(zips)} zip(s), {sum(z['size'] for z in zips.values()) / 1e9:.1f} GB remote. Listing contents...", flush=True)

    jobs = []
    client = _client()
    for name, z in sorted(zips.items()):
        zf = zipfile.ZipFile(HttpFile(z["url"], z["size"], client, block=1 << 20))
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith(VIDEO_EXTS) or "__MACOSX" in info.filename:
                continue
            out = OUT_DIR / Path(info.filename).with_suffix(".npz")
            if not out.exists():
                jobs.append((z["url"], z["size"], info.filename, str(out), info.compress_size))
    total_gb = sum(j[4] for j in jobs) / 1e9
    print(f"{len(jobs)} video(s) to process ({total_gb:.1f} GB), {args.workers} workers.", flush=True)
    if not jobs:
        return

    t0, done_bytes, ok, failed = time.time(), 0, 0, []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        futures = {pool.submit(process_member, *j[:4]): j for j in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            j = futures[fut]
            done_bytes += j[4]
            try:
                fut.result()
                ok += 1
            except Exception as exc:  # noqa: BLE001 - keep going, report at the end
                failed.append((j[2], str(exc)[:200]))
            if n % 20 == 0 or n == len(jobs):
                el = time.time() - t0
                rate = done_bytes / el / 1e6
                eta = (total_gb * 1e9 - done_bytes) / max(rate * 1e6, 1) / 60
                print(f"  [{n}/{len(jobs)}] ok {ok}, failed {len(failed)} | {rate:.1f} MB/s | ETA {eta:.0f} min", flush=True)

    print(f"Done: {ok} extracted, {len(failed)} failed.")
    for m, e in failed[:20]:
        print(f"  FAILED {m}: {e}")
    if failed:
        print("Re-run the same command to retry failures.")
    (DATA_DIR / "include_extract_failures.json").write_text(json.dumps(failed, indent=1))


if __name__ == "__main__":
    main()
