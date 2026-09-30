"""Landmark layout + feature engineering shared by training and live recognition.

A "raw" sign is an array [T, 55, 2] of (x, y) image coordinates in 0..1, NaN where missing:
  rows 0-12  : 13 upper-body pose points (POSE_IDX from MediaPipe PoseLandmarker)
  rows 13-33 : 21 points of the signer's LEFT hand   (MediaPipe HandLandmarker)
  rows 34-54 : 21 points of the signer's RIGHT hand
Hands are assigned to left/right by distance to the pose wrists, not by MediaPipe's handedness
label, so the result is the same for mirrored and non-mirrored video.

The browser (static/sign.js) builds exactly this layout and sends it to /recognize.
"""
import numpy as np

# nose, eyes (inner), ears, mouth corners, shoulders, elbows, wrists
POSE_IDX = [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
N_POSE, N_HAND = len(POSE_IDX), 21
N_POINTS = N_POSE + 2 * N_HAND  # 55
L_SH, R_SH = POSE_IDX.index(11), POSE_IDX.index(12)
L_WR, R_WR = POSE_IDX.index(15), POSE_IDX.index(16)
LEFT = slice(N_POSE, N_POSE + N_HAND)
RIGHT = slice(N_POSE + N_HAND, N_POINTS)

SEQ_LEN = 32  # every sign is resampled to this many frames
# A hand whose wrist is lower than this (in shoulder-widths below the shoulder line) is "at rest".
# INCLUDE films the full body, so resting hands hang visibly at ~2.5-3.0; signing happens at -0.5..1.3.
# Webcams usually don't see resting hands at all - masking them makes both sources look the same.
REST_Y = 1.75
FEATURE_DIM = N_POINTS * 2 + 2 * N_HAND * 2 + 2  # body-relative (110) + hand shape (84) + presence (2) = 196

# pose index pairs that swap under a horizontal flip (left <-> right body side)
_POSE_FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]


def assign_hands(pose_xy: np.ndarray | None, hands: list[np.ndarray]) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Split detected hands [21,2] into (left, right) using the nearest pose wrist."""
    left = right = None
    if not hands:
        return None, None
    if pose_xy is not None and not np.isnan(pose_xy[[L_WR, R_WR]]).any():
        lw, rw = pose_xy[L_WR], pose_xy[R_WR]
        scored = sorted(hands, key=lambda h: np.linalg.norm(h[0] - lw) - np.linalg.norm(h[0] - rw))
        if len(scored) == 1:
            h = scored[0]
            return (h, None) if np.linalg.norm(h[0] - lw) < np.linalg.norm(h[0] - rw) else (None, h)
        return scored[0], scored[-1]
    # no pose: the hand further right in the image is the signer's left (camera faces the signer)
    ordered = sorted(hands, key=lambda h: -h[0, 0])
    left = ordered[0]
    right = ordered[1] if len(ordered) > 1 else None
    return left, right


def build_frame(pose_xy: np.ndarray | None, hands: list[np.ndarray]) -> np.ndarray:
    """One raw frame [55, 2] from pose points [13,2] (already POSE_IDX-selected) and hand arrays."""
    frame = np.full((N_POINTS, 2), np.nan, dtype=np.float32)
    if pose_xy is not None:
        frame[:N_POSE] = pose_xy
    left, right = assign_hands(pose_xy, hands)
    if left is not None:
        frame[LEFT] = left
    if right is not None:
        frame[RIGHT] = right
    return frame


def trim_rest(raw: np.ndarray, margin: int = 2) -> np.ndarray:
    """Drop leading/trailing frames where no hand is visible (signer at rest)."""
    has_hand = ~np.isnan(raw[:, N_POSE:, 0]).all(axis=1)
    idx = np.flatnonzero(has_hand)
    if idx.size == 0:
        return raw
    return raw[max(0, idx[0] - margin): idx[-1] + 1 + margin]


def _fill_time(a: np.ndarray) -> np.ndarray:
    """Forward/backward-fill NaN frames along time for each point (keeps hand motion continuous)."""
    out = a.copy()
    T = out.shape[0]
    for p in range(out.shape[1]):
        col = out[:, p, 0]
        valid = np.flatnonzero(~np.isnan(col))
        if valid.size == 0 or valid.size == T:
            continue
        idx = np.interp(np.arange(T), valid, valid).round().astype(int)  # nearest valid frame
        out[:, p] = out[idx, p]
    return out


def _resample(a: np.ndarray, n: int) -> np.ndarray:
    T = a.shape[0]
    if T == n:
        return a
    src = np.linspace(0, T - 1, n)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (src - lo)[:, None, None]
    return a[lo] * (1 - w) + a[hi] * w


def body_frame(raw: np.ndarray) -> tuple[np.ndarray, float] | None:
    """(centre, scale) of the signer: mid-shoulders and shoulder width, median over the clip."""
    sh = raw[:, [L_SH, R_SH]]
    if np.isnan(sh).all():
        return None
    centre = np.nanmedian((sh[:, 0] + sh[:, 1]) / 2, axis=0)
    scale = float(np.nanmedian(np.linalg.norm(sh[:, 0] - sh[:, 1], axis=1)))
    return (centre, scale) if np.isfinite(scale) and scale > 1e-3 else None


def mask_resting_hands(raw: np.ndarray) -> np.ndarray:
    """Mark hands hanging below REST_Y as missing (see REST_Y)."""
    bf = body_frame(raw)
    if bf is None:
        return raw
    centre, scale = bf
    out = raw.copy()
    for sl in (LEFT, RIGHT):
        wrist_y = (out[:, sl][:, 0, 1] - centre[1]) / scale
        out[wrist_y > REST_Y, sl] = np.nan
    return out


def featurize(raw: np.ndarray) -> np.ndarray:
    """raw [T,55,2] (NaN = missing) -> features [SEQ_LEN, FEATURE_DIM] float32."""
    raw = trim_rest(mask_resting_hands(np.asarray(raw, dtype=np.float32)))
    if raw.shape[0] < 2:
        raw = np.repeat(raw[:1], 2, axis=0) if raw.shape[0] else np.full((2, N_POINTS, 2), np.nan, np.float32)

    presence = np.stack([~np.isnan(raw[:, LEFT, 0]).all(1), ~np.isnan(raw[:, RIGHT, 0]).all(1)], 1).astype(np.float32)

    # body frame of reference: centre = mid-shoulders, unit = shoulder width (median over the clip)
    sh = raw[:, [L_SH, R_SH]]
    if not np.isnan(sh).all():
        centre = np.nanmedian((sh[:, 0] + sh[:, 1]) / 2, axis=0)
        scale = np.nanmedian(np.linalg.norm(sh[:, 0] - sh[:, 1], axis=1))
    else:  # no pose at all: fall back to the hands' bounding box
        pts = raw[:, N_POSE:].reshape(-1, 2)
        centre = np.nanmean(pts, axis=0) if not np.isnan(pts).all() else np.array([0.5, 0.5])
        scale = 0.25
    if not np.isfinite(scale) or scale < 1e-3:
        scale = 0.25

    filled = _fill_time(raw)
    body = (filled - centre) / scale  # [T,55,2]

    shapes = []
    for sl in (LEFT, RIGHT):
        h = filled[:, sl]
        size = np.linalg.norm(h[:, 9] - h[:, 0], axis=1, keepdims=True)[:, :, None]  # wrist -> middle knuckle
        shapes.append((h - h[:, :1]) / np.where(np.isfinite(size) & (size > 1e-4), size, 1.0))
    feats = np.concatenate(
        [body.reshape(len(body), -1), shapes[0].reshape(len(body), -1), shapes[1].reshape(len(body), -1), presence], 1
    )
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    feats = _resample(feats[:, None, :], SEQ_LEN)[:, 0, :]
    return np.clip(feats, -6, 6).astype(np.float32)


def flip_raw(raw: np.ndarray) -> np.ndarray:
    """Horizontal mirror of a raw sign (left-handed signer augmentation)."""
    out = raw.copy()
    out[..., 0] = 1.0 - out[..., 0]
    out[:, :N_POSE] = out[:, _POSE_FLIP]
    left, right = out[:, LEFT].copy(), out[:, RIGHT].copy()
    out[:, LEFT], out[:, RIGHT] = right, left
    return out
