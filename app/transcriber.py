"""Local speech-to-text with faster-whisper (CPU, int8)."""
import logging
import threading

from faster_whisper import WhisperModel

from app.config import WHISPER_COMPUTE_TYPE, WHISPER_MODEL

log = logging.getLogger(__name__)

_model: WhisperModel | None = None
_lock = threading.Lock()


def get_model() -> WhisperModel:
    """Load the model once (first call downloads it from Hugging Face, ~150 MB for base.en)."""
    global _model
    with _lock:
        if _model is None:
            log.info("Loading faster-whisper model '%s' (%s, CPU)...", WHISPER_MODEL, WHISPER_COMPUTE_TYPE)
            _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type=WHISPER_COMPUTE_TYPE)
        return _model


def transcribe(audio_path: str, prompt: str | None = None) -> str:
    """Transcribe an audio file (wav/mp3/m4a/webm/ogg — anything PyAV can decode) to English text.

    `prompt` is optional context (e.g. the topic's words) that biases Whisper towards those words.
    """
    model = get_model()
    segments, info = model.transcribe(
        audio_path,
        language="en",
        beam_size=5,
        vad_filter=True,  # trims silence; noticeably faster on CPU
        initial_prompt=prompt or None,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    log.info("Transcribed %.1fs of audio: %r", info.duration, text)
    return text
