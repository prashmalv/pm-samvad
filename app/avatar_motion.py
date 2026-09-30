"""3D avatar motion for Phase 1's "Avatar" mode.

For each sign clip, MediaPipe's 3D (world) pose + hand landmarks are retargeted onto a fixed
avatar skeleton: the torso is held steady, arm and hand directions come from the video, and every
bone keeps the avatar's own length. A sentence is built by trimming each sign's rest pose and
blending one sign into the next - the avatar's advantage over stitched video.

Joint layout sent to the browser (static/avatar.js), in metres, three.js axes (x = avatar's left,
y = up, z = towards the viewer):
  0 pelvis, 1 neck, 2 head, 3 face (point in front of the nose),
  4/5 L/R shoulder, 6/7 L/R elbow, 8/9 L/R wrist, 10-30 left hand (21), 31-51 right hand (21)
"""
import logging
import threading
from pathlib import Path

import numpy as np

from app import db
from app.config import DATA_DIR, MEDIAPIPE_DIR

log = logging.getLogger(__name__)

VERSION = 1  # bump when retargeting changes, to invalidate the cache
CACHE_DIR = DATA_DIR / "avatar_motion"
MODELS = MEDIAPIPE_DIR
FPS = 25
N_JOINTS = 52
L_HAND, R_HAND = slice(10, 31), slice(31, 52)

# avatar proportions (metres)
SHOULDER_Y, SHOULDER_HALF = 1.40, 0.18
UPPER_ARM, FOREARM, PALM = 0.27, 0.25, 0.092  # PALM = wrist -> middle-finger knuckle
PELVIS = np.array([0.0, 0.93, 0.0])
NECK = np.array([0.0, 1.47, 0.0])
HEAD = np.array([0.0, 1.62, 0.02])
L_SH = np.array([SHOULDER_HALF, SHOULDER_Y, 0.0])
R_SH = np.array([-SHOULDER_HALF, SHOULDER_Y, 0.0])

BLEND_FRAMES = 7   # frames used to move from one sign into the next
REST_FRAMES = 8    # frames from/to the resting pose at the start/end of a sentence
_lock = threading.Lock()


# ------------------------------------------------------------------ geometry helpers
def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < 1e-8, 1.0, n)


def _conv(p):
    """MediaPipe world (x right-of-image = signer's left, y down, z away) -> three.js axes."""
    return np.array([p[0], -p[1], -p[2]], dtype=np.float64)


def _torso_frame(ls, rs, lh, rh):
    """Rotation that expresses camera-space vectors in the signer's torso frame."""
    x = _unit(ls - rs)
    up = _unit((ls + rs) / 2 - (lh + rh) / 2)
    up = _unit(up - np.dot(up, x) * x)
    z = np.cross(x, up)
    return np.stack([x, up, z])  # rows


def _relaxed_hand(wrist, forearm_dir, side):
    """A plausible resting hand when the tracker never saw one: fingers continue the forearm."""
    f = _unit(forearm_dir)
    across = _unit(np.cross(f, np.array([0.0, 0.0, 1.0])) + 1e-6) * (1 if side == "L" else -1)
    pts = np.zeros((21, 3))
    pts[0] = wrist
    # thumb
    for k, t in enumerate((0.25, 0.5, 0.7, 0.85), start=1):
        pts[k] = wrist + f * PALM * t * 0.8 + across * PALM * (0.35 + 0.1 * k)
    # four fingers: MCP row, then three joints each, slightly curled
    for fi, off in enumerate((0.28, 0.1, -0.08, -0.25)):
        mcp = wrist + f * PALM + across * PALM * off
        pts[5 + fi * 4] = mcp
        d = f
        for j in range(1, 4):
            d = _unit(d - np.array([0.0, 0.0, 0.12]))
            pts[5 + fi * 4 + j] = pts[5 + fi * 4 + j - 1] + d * PALM * (0.45 - 0.08 * j)
    return pts


def _rest_frame():
    fr = np.zeros((N_JOINTS, 3))
    fr[0], fr[1], fr[2] = PELVIS, NECK, HEAD
    fr[3] = HEAD + np.array([0.0, -0.01, 0.11])
    for sh, el, wr, hand, side, sx in ((L_SH, 6, 8, L_HAND, "L", 1), (R_SH, 7, 9, R_HAND, "R", -1)):
        fr[4 if side == "L" else 5] = sh
        down = _unit(np.array([0.06 * sx, -1.0, 0.08]))
        fr[el] = sh + down * UPPER_ARM
        fore = _unit(np.array([0.02 * sx, -1.0, 0.18]))
        fr[wr] = fr[el] + fore * FOREARM
        fr[hand] = _relaxed_hand(fr[wr], fore, side)
    return fr


REST = _rest_frame()


# ------------------------------------------------------------------ extraction (MediaPipe, cached)
def _landmarkers():
    from mediapipe.tasks.python import BaseOptions, vision

    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS / "hand_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4, min_tracking_confidence=0.4))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS / "pose_landmarker_lite.task")),
        running_mode=vision.RunningMode.VIDEO, num_poses=1))
    return hand, pose


def _smooth(a: np.ndarray, k: int = 5) -> np.ndarray:
    if len(a) < k:
        return a
    pad = k // 2
    padded = np.concatenate([np.repeat(a[:1], pad, 0), a, np.repeat(a[-1:], pad, 0)])
    kernel = np.ones(k) / k
    return np.stack([np.apply_along_axis(lambda s: np.convolve(s, kernel, "valid"), 0, padded[:, j])
                     for j in range(a.shape[1])], 1)


def _fill(track: list):
    """Linear-interpolate missing (None) entries; returns None if nothing was ever seen."""
    idx = [i for i, v in enumerate(track) if v is not None]
    if not idx:
        return None
    arr = np.zeros((len(track),) + track[idx[0]].shape)
    for d in np.ndindex(track[idx[0]].shape):
        arr[(slice(None),) + d] = np.interp(np.arange(len(track)), idx, [track[i][d] for i in idx])
    return arr


def extract_clip(path: Path) -> dict:
    """Clip -> {'frames': [T,52,3] avatar joints, 'active': [T] bool (hands raised)}."""
    import cv2
    import mediapipe as mp

    hand_lm, pose_lm = _landmarkers()
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    rots, arms, heads, hands = [], [], [], {"L": [], "R": []}
    i = 0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(i * 1000 / fps)
            pr, hr = pose_lm.detect_for_video(image, ts), hand_lm.detect_for_video(image, ts)
            i += 1
            if not pr.pose_world_landmarks:
                rots.append(None); arms.append(None); heads.append(None)
                hands["L"].append(None); hands["R"].append(None)
                continue
            w = pr.pose_world_landmarks[0]
            P = {j: _conv((w[j].x, w[j].y, w[j].z)) for j in (0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24)}
            R = _torso_frame(P[11], P[12], P[23], P[24])
            rots.append(R)
            arms.append(np.stack([R @ _unit(P[13] - P[11]), R @ _unit(P[15] - P[13]),
                                  R @ _unit(P[14] - P[12]), R @ _unit(P[16] - P[14])]))
            heads.append(R @ _unit(P[0] - (P[7] + P[8]) / 2))
            # assign detected hands to the nearest pose wrist in the image
            norm = pr.pose_landmarks[0]
            lw, rw = np.array([norm[15].x, norm[15].y]), np.array([norm[16].x, norm[16].y])
            got = {"L": None, "R": None}
            for k, h2d in enumerate(hr.hand_landmarks):
                c = np.array([h2d[0].x, h2d[0].y])
                side = "L" if np.linalg.norm(c - lw) < np.linalg.norm(c - rw) else "R"
                hw = np.array([_conv((q.x, q.y, q.z)) for q in hr.hand_world_landmarks[k]])
                rel = (hw - hw[0]) @ R.T
                size = np.linalg.norm(rel[9])
                if size > 1e-4:
                    got[side] = rel * (PALM / size)
            hands["L"].append(got["L"]); hands["R"].append(got["R"])
    finally:
        cap.release(); hand_lm.close(); pose_lm.close()

    arms_f = _fill(arms)
    if arms_f is None:
        raise RuntimeError(f"no body detected in {path.name}")
    heads_f = _fill(heads)
    arms_f = _smooth(arms_f.reshape(len(arms_f), -1)).reshape(-1, 4, 3)
    heads_f = _smooth(heads_f)
    T = len(arms_f)
    frames = np.repeat(REST[None], T, axis=0)
    for side, sh, (iu, if_), el, wr, sl in (("L", L_SH, (0, 1), 6, 8, L_HAND), ("R", R_SH, (2, 3), 7, 9, R_HAND)):
        frames[:, el] = sh + _unit(arms_f[:, iu]) * UPPER_ARM
        frames[:, wr] = frames[:, el] + _unit(arms_f[:, if_]) * FOREARM
        h = _fill(hands[side])
        if h is None:
            for t in range(T):
                frames[t, sl] = _relaxed_hand(frames[t, wr], frames[t, wr] - frames[t, el], side)
        else:
            h = _smooth(h.reshape(T, -1)).reshape(T, 21, 3)
            frames[:, sl] = frames[:, wr][:, None] + h
    frames[:, 3] = HEAD + _unit(heads_f) * 0.11
    # a sign is "active" while a wrist is lifted well above the hanging position
    active = np.maximum(frames[:, 8, 1], frames[:, 9, 1]) > SHOULDER_Y - 0.36
    return {"frames": frames.astype(np.float32), "active": active}


def clip_motion(path: Path) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{path.stem}.v{VERSION}.npz"
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        d = np.load(cache)
        return {"frames": d["frames"], "active": d["active"]}
    with _lock:  # MediaPipe extraction is CPU heavy; one clip at a time
        m = extract_clip(path)
    np.savez_compressed(cache, **m)
    log.info("Avatar motion extracted for %s (%d frames).", path.name, len(m["frames"]))
    return m


# ------------------------------------------------------------------ sentence assembly
def _trim(m: dict, pad: int = 3) -> np.ndarray:
    idx = np.flatnonzero(m["active"])
    if idx.size == 0:
        return m["frames"]
    return m["frames"][max(0, idx[0] - pad): idx[-1] + 1 + pad]


def _blend(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    w = ((np.arange(1, n + 1) / (n + 1)) ** 2 * (3 - 2 * np.arange(1, n + 1) / (n + 1)))[:, None, None]  # smoothstep
    return a[None] * (1 - w) + b[None] * w


def build_sequence(glosses: list[str]) -> dict:
    """Gloss tokens -> {fps, frames [T][52*3], segments, matched, missing} for the browser avatar."""
    found = db.lookup(glosses)
    parts, missing = [], []
    for g in glosses:
        path = found.get(g)
        if path is None:
            missing.append(g)
            continue
        try:
            parts.append((g, _trim(clip_motion(path))))
        except Exception as exc:  # noqa: BLE001 - skip a bad clip, keep the sentence
            log.warning("No avatar motion for %s: %s", g, exc)
            missing.append(g)

    seq, segments = [], []
    if parts:
        seq.append(_blend(REST, parts[0][1][0], REST_FRAMES))
        for k, (g, fr) in enumerate(parts):
            start = sum(len(x) for x in seq)
            seq.append(fr)
            segments.append({"gloss": g, "start": round(start / FPS, 3), "end": round((start + len(fr)) / FPS, 3)})
            nxt = parts[k + 1][1][0] if k + 1 < len(parts) else REST
            seq.append(_blend(fr[-1], nxt, BLEND_FRAMES if k + 1 < len(parts) else REST_FRAMES))
        seq.append(REST[None].repeat(4, 0))
    frames = np.concatenate(seq) if seq else np.zeros((0, N_JOINTS, 3))
    return {
        "fps": FPS,
        "joints": N_JOINTS,
        "frames": np.round(frames.reshape(len(frames), -1), 4).tolist(),
        "segments": segments,
        "matched": [g for g, _ in parts],
        "missing": missing,
    }
