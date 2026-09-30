"""Review queue for sign clips cut automatically from long ISLRTC videos (scripts/islrtc_cut.py).

Candidates wait in clips/_review/ and data/islrtc_review.json; nothing reaches the sign library until a
reviewer approves one clip for a word on /review.html. Approving moves that clip to clips/<GLOSS>.mp4
and registers it; rejecting deletes the candidates.
"""
import json
import sqlite3
import threading

from app.config import BASE_DIR, CLIPS_DIR, DATA_DIR, DB_PATH

REVIEW_PATH = DATA_DIR / "islrtc_review.json"
_lock = threading.Lock()


def _load() -> dict:
    return json.loads(REVIEW_PATH.read_text()) if REVIEW_PATH.exists() else {}


def items() -> dict:
    """Pending words (with clip URLs) plus how many were approved / rejected so far."""
    review = _load()
    counts = {s: sum(1 for i in review.values() if i["status"] == s) for s in ("pending", "approved", "rejected")}
    pending = [
        {"gloss": g, "source": i["source"], "duration": i.get("duration"),
         "clips": ["/" + c["file"] for c in i["clips"]]}
        for g, i in sorted(review.items()) if i["status"] == "pending"
    ]
    return {"pending": pending, "counts": counts}


def decide(gloss: str, choice: int | None) -> dict:
    """choice: 1-based candidate to approve, or None to reject every candidate."""
    with _lock:
        review = _load()
        item = review.get(gloss)
        if item is None or item["status"] != "pending":
            raise KeyError(gloss)
        clips = [BASE_DIR / c["file"] for c in item["clips"]]
        if choice is not None:
            if not 1 <= choice <= len(clips):
                raise ValueError(f"choice must be 1-{len(clips)}")
            dest = CLIPS_DIR / f"{gloss}.mp4"
            clips[choice - 1].replace(dest)
            with sqlite3.connect(DB_PATH, timeout=30) as conn:
                conn.execute("INSERT OR REPLACE INTO gloss_clips (gloss, video_path) VALUES (?, ?)",
                             (gloss, f"clips/{dest.name}"))
            item.update(status="approved", choice=choice)
        else:
            item["status"] = "rejected"
        for c in clips:
            c.unlink(missing_ok=True)
        REVIEW_PATH.write_text(json.dumps(review, indent=0))
    return {"gloss": gloss, "status": item["status"]}
