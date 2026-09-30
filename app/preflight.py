"""Pre-flight check: verifies every dependency before the server starts (run by run.py).

    python run.py --check     # check only, don't start the server

Each line is ✓ ok (green), ! warning (yellow: works, with a limitation) or ✗ failure (red: the app
would break). Any failure stops the launch. Secrets are only checked for presence, never printed.
"""
import importlib.metadata as md
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
from pathlib import Path

from app import config

OK, WARN, FAIL = "ok", "warn", "fail"
_COLOURS = {OK: "\033[92m", WARN: "\033[93m", FAIL: "\033[91m"}
_MARKS = {OK: "✓", WARN: "!", FAIL: "✗"}
BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"

# Distribution names whose import name differs, so the message can say what to pip install.
PIP_NAMES = {"python-multipart", "python-dotenv", "opencv-contrib-python", "pillow"}


def _enable_ansi() -> None:
    """Colours + ✓ on Windows consoles."""
    if os.name == "nt":
        os.system("")  # switches the console into VT (ANSI colour) mode
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


class Report:
    def __init__(self):
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str) -> None:
        self.rows.append((status, name, detail))
        c = _COLOURS[status]
        print(f"  {c}{_MARKS[status]}{RESET} {name:<18} {c if status != OK else ''}{detail}{RESET}", flush=True)

    def count(self, status: str) -> int:
        return sum(1 for s, _, _ in self.rows if s == status)


# ------------------------------------------------------------------ individual checks
def check_python(r: Report) -> None:
    v = sys.version_info
    r.add(OK if v >= (3, 12) else FAIL, "Python", f"{v.major}.{v.minor}.{v.micro}" + ("" if v >= (3, 12) else " (needs 3.12+)"))


def check_packages(r: Report) -> None:
    """Every pin in requirements.txt is installed; version drift is a warning."""
    pins = []
    for line in (config.BASE_DIR / "requirements.txt").read_text().splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?\s*==\s*([^\s#;]+)", line)
        if m:
            pins.append((m[1], m[2]))
    missing, drift = [], []
    for name, want in pins:
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            missing.append(name)
            continue
        if have != want:
            drift.append(f"{name} {have} (wants {want})")
    if missing:
        r.add(FAIL, "Python packages", f"missing: {', '.join(missing)} -> pip install -r requirements.txt")
    elif drift:
        r.add(WARN, "Python packages", f"{len(pins)} installed, versions differ: {'; '.join(drift)}")
    else:
        r.add(OK, "Python packages", f"{len(pins)}/{len(pins)} installed at the pinned versions")


def check_ffmpeg(r: Report) -> None:
    try:
        from app.video import _ffmpeg_exe

        exe = _ffmpeg_exe()
        out = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=20).stdout
        ver = re.search(r"ffmpeg version (\S+)", out)
        r.add(OK, "ffmpeg", f"{ver[1] if ver else 'found'} ({'system' if shutil.which('ffmpeg') else 'bundled'})")
    except Exception as exc:  # noqa: BLE001
        r.add(FAIL, "ffmpeg", f"not usable: {exc}")


def check_azure(r: Report) -> None:
    missing = [n for n in config.AZURE_ENV_VARS if not os.getenv(n)]
    if missing:
        r.add(FAIL, "Azure OpenAI", f"not set in .env: {', '.join(missing)} (translation would fail)")
        return
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    if not endpoint.startswith("https://"):
        r.add(WARN, "Azure OpenAI", "AZURE_OPENAI_ENDPOINT should start with https://")
    else:
        r.add(OK, "Azure OpenAI", "endpoint, key, API version and deployment are set")


def check_database(r: Report) -> None:
    if not config.DB_PATH.exists():
        r.add(FAIL, "Sign database", f"{config.DB_PATH.name} missing -> python scripts/seed_db.py")
        return
    with sqlite3.connect(config.DB_PATH) as conn:
        rows = conn.execute("SELECT gloss, video_path FROM gloss_clips").fetchall()
    broken = [g for g, p in rows if not (config.BASE_DIR / p).is_file()]
    usable = len(rows) - len(broken)
    if usable == 0:  # Sign -> Speech still works; only Speech -> Sign needs the videos
        r.add(WARN, "Sign database", f"{len(rows)} signs listed but no clip files: Speech → Sign needs the "
              "clips zip unpacked into clips/ (README: Setup on a new machine)")
    elif broken:
        r.add(WARN, "Sign database", f"{usable} signs ready, {len(broken)} point to missing files (skipped)")
    else:
        r.add(OK, "Sign database", f"{usable} signs, all clip files present")


def check_topics(r: Report) -> None:
    try:
        from app import db, topics

        have = set(db.get_vocabulary())
        parts, gaps = [], 0
        for t in topics.TOPICS.values():
            n = sum(w in have for w in t.words)
            gaps += len(t.words) - n
            parts.append(f"{t.label} {n}/{len(t.words)}")
        r.add(OK if gaps == 0 else WARN, "Topics", " · ".join(parts) + ("" if gaps == 0 else " (missing words are skipped)"))
    except Exception as exc:  # noqa: BLE001
        r.add(WARN, "Topics", f"could not check: {exc}")


def check_whisper(r: Report) -> None:
    try:
        from huggingface_hub import constants

        repo = config.WHISPER_MODEL if "/" in config.WHISPER_MODEL else f"Systran/faster-whisper-{config.WHISPER_MODEL}"
        cached = Path(constants.HF_HUB_CACHE, "models--" + repo.replace("/", "--"), "snapshots")
        if cached.is_dir() and any(cached.iterdir()):
            r.add(OK, "Speech (Whisper)", f"{config.WHISPER_MODEL} downloaded, runs offline")
        else:
            r.add(WARN, "Speech (Whisper)", f"{config.WHISPER_MODEL} not downloaded yet: first recording fetches it (~150 MB)")
    except Exception as exc:  # noqa: BLE001
        r.add(WARN, "Speech (Whisper)", f"could not check the model cache: {exc}")


def check_fingerspelling(r: Report) -> None:
    from app import fingerspell

    n = len(fingerspell.available_letters())
    if fingerspell.enabled():
        r.add(OK, "Fingerspelling", f"on ({n} letter/digit clips)")
    elif not config.FINGERSPELL:
        r.add(OK, "Fingerspelling", "switched off in .env")
    else:
        r.add(WARN, "Fingerspelling", f"off: {n}/26 letter clips in clips/alphabet (words without a sign are skipped)")


def check_avatar(r: Report) -> None:
    need = [config.MEDIAPIPE_DIR / f for f in ("hand_landmarker.task", "pose_landmarker_lite.task")]
    need += [config.STATIC_DIR / "vendor/three/three.module.js", config.STATIC_DIR / "vendor/three/OrbitControls.js"]
    missing = [p.relative_to(config.BASE_DIR).as_posix() for p in need if not p.is_file()]
    cache = config.DATA_DIR / "avatar_motion"
    cached = len(list(cache.glob("*.npz"))) if cache.is_dir() else 0
    if missing:
        r.add(WARN, "3D avatar", f"unavailable, missing: {', '.join(missing)}")
    else:
        r.add(OK, "3D avatar", f"ready ({cached} signs pre-computed, others extracted on first use)")
    model = config.STATIC_DIR / "models" / "signer.glb"
    if model.is_file():
        r.add(OK, "Avatar character", f"signer.glb ({model.stat().st_size / 1e6:.0f} MB)")


def check_sign_to_speech(r: Report) -> None:
    from app import sign_recognizer

    web = [config.STATIC_DIR / p for p in ("models/hand_landmarker.task", "models/pose_landmarker_lite.task",
                                            "vendor/mediapipe/vision_bundle.mjs", "vendor/mediapipe/wasm/vision_wasm_internal.wasm")]
    missing = [p.relative_to(config.BASE_DIR).as_posix() for p in web if not p.is_file()]
    if missing:
        r.add(WARN, "Sign → Speech", f"browser tracking files missing: {', '.join(missing)}")
    elif not sign_recognizer.available():
        r.add(WARN, "Sign → Speech", "no trained model in models/sign (see README Phase 2)")
    else:
        rep = sign_recognizer.report()
        r.add(OK, "Sign → Speech", f"model ready: {rep.get('signs', '?')} signs, {rep.get('test_top1', 0):.0%} test accuracy")


def check_disk(r: Report) -> None:
    free = shutil.disk_usage(config.BASE_DIR).free / 1e9
    r.add(OK if free >= 2 else WARN, "Disk space", f"{free:.1f} GB free" + ("" if free >= 2 else " (low: rendered videos need space)"))


def check_port(r: Report, host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            r.add(OK, "Network port", f"{host}:{port} is free")
        except OSError:
            r.add(FAIL, "Network port", f"{port} is already in use - is the app already running? Close it first.")


# ------------------------------------------------------------------ runner
def run(host: str = "127.0.0.1", port: int = 8000) -> bool:
    """Run every check, print the report, return True when nothing failed."""
    _enable_ansi()
    print(f"\n  {BOLD}Speak to Sign · pre-flight check{RESET}\n  {DIM}{'─' * 60}{RESET}")
    r = Report()
    for check in (check_python, check_packages, check_ffmpeg, check_azure, check_database, check_topics,
                  check_whisper, check_fingerspelling, check_avatar, check_sign_to_speech, check_disk):
        try:
            check(r)
        except Exception as exc:  # noqa: BLE001 - a broken check must not hide the others
            r.add(FAIL, check.__name__.removeprefix("check_"), f"check crashed: {exc}")
    check_port(r, host, port)
    print(f"  {DIM}{'─' * 60}{RESET}")
    fails, warns = r.count(FAIL), r.count(WARN)
    if fails:
        print(f"  \033[91m{BOLD}✗ {fails} problem(s) must be fixed before take-off.{RESET}\n")
        return False
    note = f"  {DIM}({warns} warning(s) above: the app works, with those limits){RESET}" if warns else ""
    print(f"  \033[92m{BOLD}✓ All systems OK — ready for take-off 🚀{RESET}   http://{host}:{port}{note}\n")
    return True
