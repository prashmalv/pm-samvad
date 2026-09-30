"""Phase 2: recognise one isolated ISL sign from a landmark sequence (ONNX model, CPU)."""
import json
import logging
import threading

import numpy as np

from app.config import BASE_DIR
from app.sign_features import N_POINTS, featurize

log = logging.getLogger(__name__)

MODEL_DIR = BASE_DIR / "models" / "sign"
_session = None
_labels: list[str] = []
_lock = threading.Lock()


def available() -> bool:
    return (MODEL_DIR / "sign_model.onnx").is_file() and (MODEL_DIR / "labels.json").is_file()


def _load():
    global _session, _labels
    with _lock:
        if _session is None:
            import onnxruntime as ort

            if not available():
                raise RuntimeError("No sign model yet. Train it with: python scripts/train_sign_model.py")
            _session = ort.InferenceSession(str(MODEL_DIR / "sign_model.onnx"), providers=["CPUExecutionProvider"])
            _labels = json.loads((MODEL_DIR / "labels.json").read_text())
            log.info("Loaded sign model with %d signs.", len(_labels))
    return _session, _labels


def labels() -> list[str]:
    return _load()[1] if available() else []


def report() -> dict:
    f = MODEL_DIR / "report.json"
    return json.loads(f.read_text()) if f.is_file() else {}


def parse_frames(frames: list[list[float | None]]) -> np.ndarray:
    """Browser payload (T frames x 110 numbers, null = missing) -> raw [T, 55, 2]."""
    arr = np.array([[np.nan if v is None else v for v in f] for f in frames], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != N_POINTS * 2:
        raise ValueError(f"each frame must have {N_POINTS * 2} values, got shape {arr.shape}")
    return arr.reshape(len(arr), N_POINTS, 2)


def recognize(raw: np.ndarray, top_k: int = 5) -> list[dict]:
    """raw [T,55,2] -> [{gloss, confidence}] best first."""
    session, names = _load()
    x = featurize(raw)[None]
    logits = session.run(None, {"x": x})[0][0]
    p = np.exp(logits - logits.max())
    p /= p.sum()
    order = np.argsort(-p)[:top_k]
    return [{"gloss": names[i], "confidence": round(float(p[i]), 4)} for i in order]
