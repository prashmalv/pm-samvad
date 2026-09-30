"""Fingerspelling fallback: spell a word with the ISL manual alphabet (one clip per letter/digit).

Letter clips live in ALPHABET_DIR as A.mp4 ... Z.mp4 and optionally 0.mp4 ... 9.mp4.
They are kept out of the gloss DB on purpose: the dataset already has words "A" and "I".
"""
import logging
import string
from pathlib import Path

from app.config import ALPHABET_DIR, FINGERSPELL

log = logging.getLogger(__name__)

CHARACTERS = string.ascii_uppercase + string.digits


def letter_clip(ch: str) -> Path | None:
    path = ALPHABET_DIR / f"{ch.upper()}.mp4"
    return path if path.is_file() else None


def available_letters() -> list[str]:
    return [c for c in CHARACTERS if letter_clip(c)]


def enabled() -> bool:
    """On when switched on in config and at least the 26 letters are present."""
    return FINGERSPELL and all(letter_clip(c) for c in string.ascii_uppercase)


def spell(word: str) -> list[tuple[str, Path]] | None:
    """[(letter, clip)] for every letter/digit in `word`, or None if fingerspelling can't cover it."""
    if not FINGERSPELL:
        return None
    chars = [c for c in word.upper() if c in CHARACTERS]
    if not chars:
        return None
    clips = [(c, letter_clip(c)) for c in chars]
    lacking = sorted({c for c, p in clips if p is None})
    if lacking:
        log.warning("Cannot fingerspell '%s': no clip for %s in %s", word, ", ".join(lacking), ALPHABET_DIR)
        return None
    return clips  # type: ignore[return-value]
