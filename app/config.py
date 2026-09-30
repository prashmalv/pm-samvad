"""Central configuration. All secrets come from environment variables (.env is loaded if present)."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

CLIPS_DIR = BASE_DIR / "clips"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"
DB_PATH = Path(os.getenv("ISL_DB_PATH", DATA_DIR / "isl_clips.db"))
# MediaPipe .task models (hand + pose landmarkers). The browser's copy in static/models is the same file,
# so a fresh clone (which doesn't ship models/mediapipe) uses that one.
MEDIAPIPE_DIR = BASE_DIR / "models" / "mediapipe"
if not (MEDIAPIPE_DIR / "hand_landmarker.task").is_file():
    MEDIAPIPE_DIR = STATIC_DIR / "models"

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
VIDEO_BACKEND = os.getenv("VIDEO_BACKEND", "auto").lower()

# Fingerspelling fallback: words without a sign clip are spelled with letter clips A.mp4..Z.mp4 (+ 0-9).
ALPHABET_DIR = Path(os.getenv("ALPHABET_DIR", CLIPS_DIR / "alphabet"))
FINGERSPELL = os.getenv("FINGERSPELL", "1").lower() not in ("0", "false", "no", "off")
FINGERSPELL_SPEED = float(os.getenv("FINGERSPELL_SPEED", "1.5"))  # letters are signed quicker than words

# Every stitched clip is normalised to this so mixed sources join cleanly.
OUTPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_FPS = 1280, 720, 25

AZURE_ENV_VARS = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
)

for d in (CLIPS_DIR, DATA_DIR, OUTPUT_DIR, ALPHABET_DIR):
    d.mkdir(parents=True, exist_ok=True)


def azure_settings() -> dict[str, str]:
    """Return the Azure OpenAI settings, raising a clear error if any are missing."""
    missing = [name for name in AZURE_ENV_VARS if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Missing Azure OpenAI environment variables: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill them in."
        )
    return {name: os.environ[name] for name in AZURE_ENV_VARS}
