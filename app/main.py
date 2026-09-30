"""FastAPI app: /transcribe, /gloss, /generate-isl-video, plus the single-page frontend."""
import json
import logging
import mimetypes
import os
import tempfile
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import db, fingerspell, pipeline, review, sign_recognizer, topics
from app.english_graph import gloss_to_english
from app.config import ALPHABET_DIR, CLIPS_DIR, STATIC_DIR
from app.transcriber import transcribe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
log = logging.getLogger("isl")

MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAX_SIGN_FRAMES = 600  # 40 s at 15 fps - far longer than any single sign

# Windows' registry often lacks these; the browser needs them for MediaPipe's WebAssembly build.
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("text/javascript", ".mjs")

app = FastAPI(title="Audio → ISL Avatar", version="1.0.0")


@app.middleware("http")
async def revalidate_static(request, call_next):
    """Make browsers re-check the page, CSS and JS on every load (a cheap 304 when unchanged),
    so a code update shows up after a normal refresh instead of an old cached stylesheet."""
    response = await call_next(request)
    if request.method == "GET" and "cache-control" not in response.headers:
        response.headers["Cache-Control"] = "no-cache"
    return response


class GlossRequest(BaseModel):
    text: str
    topic: str = topics.DEFAULT_TOPIC  # casual | hotel | hospital


class GlossResponse(BaseModel):
    text: str
    gloss: str
    tokens: list[str]
    topic: str


class TranscribeResponse(BaseModel):
    text: str


def _save_upload(audio: UploadFile) -> str:
    """Write the upload to a temp file (keeping its extension so the decoder can sniff it)."""
    data = audio.file.read()
    if not data:
        raise HTTPException(400, "Uploaded audio file is empty.")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "Audio file too large (max 25 MB).")
    suffix = Path(audio.filename or "").suffix or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def _llm_error(exc: Exception) -> HTTPException:
    if isinstance(exc, RuntimeError) and "AZURE_OPENAI" in str(exc):
        return HTTPException(500, str(exc))
    log.exception("Gloss generation failed")
    return HTTPException(502, f"Azure OpenAI gloss step failed: {exc}")


@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe_endpoint(audio: UploadFile = File(...), topic: str = Form(topics.DEFAULT_TOPIC)):
    path = _save_upload(audio)
    try:
        return TranscribeResponse(text=transcribe(path, topics.whisper_prompt(topics.get(topic))))
    except Exception as exc:  # noqa: BLE001
        log.exception("Transcription failed")
        raise HTTPException(422, f"Could not transcribe audio: {exc}") from exc
    finally:
        os.unlink(path)


@app.post("/gloss", response_model=GlossResponse)
def gloss_endpoint(req: GlossRequest):
    if not req.text.strip():
        raise HTTPException(400, "Text is empty.")
    topic = topics.get(req.topic).id
    try:
        tokens = pipeline.text_to_gloss(req.text, topic)
    except Exception as exc:  # noqa: BLE001
        raise _llm_error(exc) from exc
    return GlossResponse(text=req.text, gloss=" ".join(tokens), tokens=tokens, topic=topic)


@app.get("/topics")
def topics_endpoint():
    """Conversation topics with their quick phrases and sign coverage.

    Each word is {gloss, clip} where clip is null when the sign still needs recording
    (it is then fingerspelled if the alphabet is installed, otherwise skipped)."""
    try:
        clips = {g: p for g, p in db.available_clips()}
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc
    out = []
    for t in topics.TOPICS.values():
        words = [{"gloss": w, "clip": f"/clips/{clips[w].relative_to(CLIPS_DIR).as_posix()}" if w in clips else None}
                 for w in t.words]
        out.append({
            "id": t.id, "label": t.label, "description": t.description, "examples": list(t.examples),
            "words": words, "covered": sum(1 for w in words if w["clip"]), "total": len(words),
        })
    return {"default": topics.DEFAULT_TOPIC, "topics": out}


@app.post("/generate-isl-video")
def generate_isl_video(audio: UploadFile = File(...), topic: str = Form(topics.DEFAULT_TOPIC)):
    """Full pipeline. Returns video/mp4; transcript and gloss details are in X-ISL-* headers
    (URL-encoded JSON in X-ISL-Info). If no clip matched, returns 422 JSON with the same info."""
    path = _save_upload(audio)
    try:
        result = pipeline.run(path, topic)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        if "AZURE_OPENAI" in str(exc) or "openai" in type(exc).__module__:
            raise _llm_error(exc) from exc
        log.exception("Pipeline failed")
        raise HTTPException(500, f"Pipeline failed: {exc}") from exc
    finally:
        os.unlink(path)

    return _video_response(result)


class RenderRequest(BaseModel):
    tokens: list[str]
    transcript: str = ""


@app.post("/render")
def render_endpoint(req: RenderRequest):
    """Gloss tokens in, stitched video out (same response format as /generate-isl-video)."""
    tokens = [t.strip().upper() for t in req.tokens if t.strip()]
    if not tokens:
        raise HTTPException(400, "No gloss tokens given.")
    try:
        result = pipeline.render(tokens, req.transcript)
    except Exception as exc:  # noqa: BLE001
        log.exception("Render failed")
        raise HTTPException(500, f"Video rendering failed: {exc}") from exc
    return _video_response(result)


def _video_response(result: pipeline.PipelineResult):
    info = result.info()
    if result.video_path is None:
        reason = "No speech detected." if not result.transcript and not result.gloss else "None of the gloss words have a video clip."
        return JSONResponse(status_code=422, content={"detail": reason, **info})
    return FileResponse(
        result.video_path,
        media_type="video/mp4",
        filename=result.video_path.name,
        headers={"X-ISL-Info": quote(json.dumps(info))},
    )


class AvatarRequest(BaseModel):
    tokens: list[str]


@app.post("/avatar-motion")
def avatar_motion_endpoint(req: AvatarRequest):
    """Gloss tokens in, 3D avatar animation out (Phase 1 "Avatar" mode, rendered in the browser).

    Returns {fps, joints, frames: [[x,y,z]*52 per frame], segments, matched, missing}. The first
    request for a sign extracts its motion from the clip with MediaPipe (a few seconds), then it's cached.
    """
    tokens = [t.strip().upper() for t in req.tokens if t.strip()]
    if not tokens:
        raise HTTPException(400, "No gloss tokens given.")
    try:
        from app.avatar_motion import build_sequence

        seq = build_sequence(tokens)
    except ImportError as exc:
        raise HTTPException(500, "Avatar mode needs mediapipe: pip install mediapipe==1.0.1") from exc
    if not seq["segments"]:
        return JSONResponse(status_code=422, content={"detail": "None of the gloss words have a sign clip.", **seq})
    return seq


@app.get("/vocabulary")
def vocabulary():
    """Glosses that currently have a clip (for the sign library in the UI)."""
    try:
        clips = db.available_clips()
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc
    return {
        "signs": [{"gloss": g, "clip": f"/clips/{p.relative_to(CLIPS_DIR).as_posix()}"} for g, p in clips],
        "fingerspelling": fingerspell.enabled(),
        "alphabet": [
            {"letter": c, "clip": f"/alphabet/{c}.mp4"} for c in fingerspell.available_letters()
        ],
    }


# ------------------------------------------------------------------ Phase 2: sign -> text
class RecognizeRequest(BaseModel):
    frames: list[list[float | None]]  # T x 110: 55 (x, y) points per frame, null = not detected
    top_k: int = 5


@app.post("/recognize")
def recognize_endpoint(req: RecognizeRequest):
    """Landmarks of ONE sign in, ranked guesses out: {"candidates": [{"gloss", "confidence"}]}."""
    if not sign_recognizer.available():
        raise HTTPException(503, "Sign model not trained yet. Run: python scripts/train_sign_model.py")
    if not 3 <= len(req.frames) <= MAX_SIGN_FRAMES:
        raise HTTPException(400, f"Expected 3-{MAX_SIGN_FRAMES} frames, got {len(req.frames)}.")
    try:
        raw = sign_recognizer.parse_frames(req.frames)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"candidates": sign_recognizer.recognize(raw, max(1, min(req.top_k, 10)))}


class EnglishRequest(BaseModel):
    glosses: list[str]


@app.post("/to-english")
def to_english_endpoint(req: EnglishRequest):
    """Recognised gloss sequence in, natural English sentence out."""
    if not any(g.strip() for g in req.glosses):
        raise HTTPException(400, "No signs given.")
    try:
        english = gloss_to_english(req.glosses)
    except Exception as exc:  # noqa: BLE001
        raise _llm_error(exc) from exc
    return {"gloss": " ".join(g.upper() for g in req.glosses), "english": english}


@app.get("/sign-model")
def sign_model_info():
    """Which signs the recogniser knows, and how well it scored on the INCLUDE test set."""
    return {"available": sign_recognizer.available(), "signs": sign_recognizer.labels(), "report": sign_recognizer.report()}


# ------------------------------------------------------------------ review of auto-cut ISLRTC clips
class ReviewDecision(BaseModel):
    choice: int | None = None  # 1-based candidate clip to approve; null rejects all


@app.get("/review/items")
def review_items():
    """Words cut from long ISLRTC videos that wait for a reviewer (scripts/islrtc_cut.py)."""
    return review.items()


@app.post("/review/{gloss}")
def review_decide(gloss: str, req: ReviewDecision):
    try:
        return review.decide(gloss.upper(), req.choice)
    except KeyError as exc:
        raise HTTPException(404, f"No pending review for {gloss}.") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/health")
def health():
    return {"status": "ok"}


app.mount("/alphabet", StaticFiles(directory=ALPHABET_DIR), name="alphabet")
app.mount("/clips", StaticFiles(directory=CLIPS_DIR), name="clips")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
