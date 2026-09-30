"""SQLite lookup: gloss -> video clip path."""
import logging
import sqlite3
from pathlib import Path

from app.config import BASE_DIR, DB_PATH

log = logging.getLogger(__name__)


def _connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise RuntimeError(f"Database not found at {DB_PATH}. Run: python scripts/seed_db.py")
    return sqlite3.connect(DB_PATH)


def available_clips() -> list[tuple[str, Path]]:
    """(gloss, clip path) for every gloss whose clip file actually exists."""
    with _connect() as conn:
        rows = conn.execute("SELECT gloss, video_path FROM gloss_clips ORDER BY gloss").fetchall()
    return [(gloss, BASE_DIR / path) for gloss, path in rows if (BASE_DIR / path).is_file()]


def get_vocabulary() -> list[str]:
    """Glosses that have a clip (fed to the LLM as preferred vocabulary)."""
    try:
        return [gloss for gloss, _ in available_clips()]
    except RuntimeError as exc:
        log.warning("%s - continuing with no vocabulary hint.", exc)
        return []


def lookup(glosses: list[str]) -> dict[str, Path | None]:
    """Map each gloss to its clip file, or None if it has no DB row or the file is missing (warned, not fatal)."""
    found: dict[str, Path | None] = {}
    with _connect() as conn:
        for gloss in dict.fromkeys(glosses):
            row = conn.execute("SELECT video_path FROM gloss_clips WHERE gloss = ?", (gloss,)).fetchone()
            if row is None:
                log.warning("No clip mapped for gloss '%s'.", gloss)
                found[gloss] = None
                continue
            path = Path(row[0])
            if not path.is_absolute():
                path = BASE_DIR / path
            if not path.is_file():
                log.warning("Clip for gloss '%s' is mapped to %s but the file does not exist.", gloss, path)
                found[gloss] = None
                continue
            found[gloss] = path
    return found


def resolve_clips(glosses: list[str]) -> tuple[list[tuple[str, Path]], list[str]]:
    """Map each gloss to an existing clip file, in order. Returns (matched, missing)."""
    found = lookup(glosses)
    matched = [(g, found[g]) for g in glosses if found[g] is not None]
    missing = [g for g in glosses if found[g] is None]
    return matched, missing  # type: ignore[return-value]
