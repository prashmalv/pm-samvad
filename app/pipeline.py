"""End-to-end: audio -> English -> ISL gloss -> clips -> stitched video."""
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app import db, fingerspell, topics, video
from app.config import FINGERSPELL_SPEED, OUTPUT_DIR
from app.gloss_graph import english_to_gloss
from app.transcriber import transcribe

log = logging.getLogger(__name__)

KEEP_OUTPUTS = 30  # older stitched videos are pruned


@dataclass
class PipelineResult:
    transcript: str
    gloss: list[str]
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    spelled: list[str] = field(default_factory=list)  # glosses rendered by fingerspelling
    segments: list[dict] = field(default_factory=list)  # [{gloss, start, end, spelled?, letters?}] in seconds
    video_path: Path | None = None

    def info(self) -> dict:
        return {
            "transcript": self.transcript,
            "gloss": " ".join(self.gloss),
            "matched": self.matched,
            "missing": self.missing,
            "spelled": self.spelled,
            "segments": self.segments,
        }


def text_to_gloss(english: str, topic: str | None = None) -> list[str]:
    return english_to_gloss(english, db.get_vocabulary(), topic)


def render(gloss: list[str], transcript: str = "") -> PipelineResult:
    """Gloss tokens -> stitched video.

    Each gloss plays its sign clip; if it has none, it is fingerspelled letter by letter;
    if that isn't possible either (no alphabet clips), it is skipped with a warning.
    """
    result = PipelineResult(transcript=transcript, gloss=gloss)
    if not gloss:
        return result

    found = db.lookup(gloss)
    plan: list[tuple[str, list[tuple[str, Path, float]]]] = []  # (gloss, [(label, clip, speed)])
    for g in gloss:
        if found[g] is not None:
            plan.append((g, [(g, found[g], 1.0)]))
        elif letters := fingerspell.spell(g):
            log.info("Fingerspelling '%s' (%d letters).", g, len(letters))
            plan.append((g, [(ch, p, FINGERSPELL_SPEED) for ch, p in letters]))
            result.spelled.append(g)
        else:
            result.missing.append(g)
    result.matched = [g for g, _ in plan]
    if result.missing:
        log.warning("Skipped %d gloss(es) with no clip: %s", len(result.missing), " ".join(result.missing))
    if not plan:
        return result

    out = OUTPUT_DIR / f"isl_{uuid.uuid4().hex[:12]}.mp4"
    result.video_path = video.stitch([(p, speed) for _, parts in plan for _, p, speed in parts], out)

    t = 0.0
    for g, parts in plan:
        seg: dict = {"gloss": g, "start": round(t, 3)}
        letters = []
        for label, p, speed in parts:
            d = video.duration(p) / speed
            letters.append({"letter": label, "start": round(t, 3), "end": round(t + d, 3)})
            t += d
        seg["end"] = round(t, 3)
        if g in result.spelled:
            seg["spelled"] = True
            seg["letters"] = letters
        result.segments.append(seg)
    _prune_outputs()
    return result


def run(audio_path: str, topic: str | None = None) -> PipelineResult:
    transcript = transcribe(audio_path, topics.whisper_prompt(topics.get(topic)))
    return render(text_to_gloss(transcript, topic) if transcript else [], transcript)


def _prune_outputs() -> None:
    files = sorted(OUTPUT_DIR.glob("isl_*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[KEEP_OUTPUTS:]:
        old.unlink(missing_ok=True)
