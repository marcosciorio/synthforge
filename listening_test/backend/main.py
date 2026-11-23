"""Backend of the Listening Test website (FastAPI).

Serves the audio samples, collects the participants' descriptions and stores
them for offline analysis. No model is loaded by this process.
"""

import json
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pool import Pool

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent.parent

DATASET_DIR = PROJECT_ROOT / "data" / "Dataset_TAL_44kHz" / "Handcrafted" / "audio_wav"
FRONTEND_DIST = BACKEND_DIR.parent / "frontend" / "dist"

DATA_DIR = Path(os.environ.get("DATA_DIR", str(BACKEND_DIR)))
RESULTS_DIR = DATA_DIR / "results"
DB_PATH = DATA_DIR / "data.db"

SAMPLE_SELECTION_PATH = BACKEND_DIR / "sample_selection.json"
with open(SAMPLE_SELECTION_PATH, encoding="utf-8") as f:
    ALLOWED_SAMPLE_IDS = set(json.load(f)["sample_ids"])

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

USERS_INDEX_PATH = RESULTS_DIR / "users_index.json"
_users_index_lock = threading.Lock()

app = FastAPI(title="Listening Test API")
pool = Pool(db_path=str(DB_PATH), audio_dir=str(DATASET_DIR), allowed_ids=ALLOWED_SAMPLE_IDS)


EXPERTISE_LABELS = {
    1: "Expert / sound designer",
    2: "Knowledgeable listener",
    3: "Outside listener",
}


def _safe_sample_id(sample_id: str) -> str:
    if not sample_id or "/" in sample_id or "\\" in sample_id or sample_id in (".", ".."):
        raise HTTPException(400, "invalid sample_id")
    wav_path = (DATASET_DIR / f"{sample_id}.wav").resolve()
    if DATASET_DIR.resolve() not in wav_path.parents:
        raise HTTPException(400, "invalid sample_id")
    if not wav_path.is_file():
        raise HTTPException(404, "sample not found")
    return sample_id


@app.get("/api/stats")
def get_stats():
    return pool.stats()


def _next_response(session_id: str, exclude: Optional[str] = None) -> dict:
    sample_id = pool.claim_next(session_id, exclude=exclude)
    if sample_id is None:
        return {"done": True}
    stats = pool.stats()
    return {
        "done": False,
        "sample_id": sample_id,
        "audio_url": f"/api/audio/{sample_id}",
        "total": stats["total"],
        "current_round": stats["current_round"],
        "completed_this_round": stats["completed_this_round"],
    }


@app.get("/api/next")
def get_next(session_id: str):
    if not session_id or len(session_id) > 128:
        raise HTTPException(400, "invalid session_id")
    return _next_response(session_id)


@app.get("/api/audio/{sample_id}")
def get_audio(sample_id: str):
    sample_id = _safe_sample_id(sample_id)
    return FileResponse(DATASET_DIR / f"{sample_id}.wav", media_type="audio/wav")


class SkipBody(BaseModel):
    session_id: str
    sample_id: str


@app.post("/api/skip")
def skip(body: SkipBody):
    sample_id = _safe_sample_id(body.sample_id)
    ok = pool.skip(body.session_id, sample_id)
    if not ok:
        raise HTTPException(
            409,
            "This sample is no longer assigned to this session "
            "(claim expired or already handled elsewhere). Please request the next one.",
        )
    return _next_response(body.session_id, exclude=sample_id)


def _record_user_submission(
    session_id: str,
    expertise: int,
    expertise_label: str,
    note: Optional[str],
    sample_id: str,
    free_description: str,
    guided_description: str,
    submitted_at: float,
) -> None:
    entry = {
        "sample_id": sample_id,
        "free_description": free_description,
        "guided_description": guided_description,
        "submitted_at": datetime.fromtimestamp(submitted_at).isoformat(sep=" ", timespec="seconds"),
    }
    with _users_index_lock:
        if USERS_INDEX_PATH.is_file():
            with open(USERS_INDEX_PATH, encoding="utf-8") as f:
                users = json.load(f)
        else:
            users = []

        user = next((u for u in users if u["session_id"] == session_id), None)
        if user is None:
            user = {
                "session_id": session_id,
                "describer_expertise": expertise,
                "describer_expertise_label": expertise_label,
                "describer_note": note,
                "descriptions": [],
            }
            users.insert(0, user)
        else:
            user["describer_expertise"] = expertise
            user["describer_expertise_label"] = expertise_label
            user["describer_note"] = note

        user["descriptions"].insert(0, entry)

        with open(USERS_INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2, ensure_ascii=False)


class SubmitBody(BaseModel):
    session_id: str
    sample_id: str
    free_description: str
    guided_description: str
    guided_choices: Optional[dict] = None
    describer_expertise: int
    describer_note: Optional[str] = None


def _validate_description(text: str, field: str) -> str:
    text = text.strip()
    if not text:
        raise HTTPException(400, f"{field} is empty")
    if not text.lower().startswith("the sound of"):
        raise HTTPException(400, f'{field} must start with "The sound of..."')
    return text


@app.post("/api/submit")
def submit(body: SubmitBody):
    sample_id = _safe_sample_id(body.sample_id)
    free_description = _validate_description(body.free_description, "free_description")
    guided_description = _validate_description(body.guided_description, "guided_description")
    if body.describer_expertise not in EXPERTISE_LABELS:
        raise HTTPException(400, "invalid describer_expertise")

    order_id = pool.record_description(body.session_id, sample_id)
    if order_id is None:
        raise HTTPException(
            409,
            "This sample is no longer assigned to this session "
            "(claim expired or already completed elsewhere). Please request the next one.",
        )

    out_dir = RESULTS_DIR / sample_id / str(order_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DATASET_DIR / f"{sample_id}.wav", out_dir / f"{sample_id}.wav")
    note = (body.describer_note or "").strip() or None
    submitted_at = time.time()
    with open(out_dir / "description.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "sample_id": sample_id,
                "free_description": free_description,
                "guided_description": guided_description,
                "guided_choices": body.guided_choices,
                "describer_expertise": body.describer_expertise,
                "describer_expertise_label": EXPERTISE_LABELS[body.describer_expertise],
                "describer_note": note,
                "session_id": body.session_id,
                "submitted_at": submitted_at,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    _record_user_submission(
        session_id=body.session_id,
        expertise=body.describer_expertise,
        expertise_label=EXPERTISE_LABELS[body.describer_expertise],
        note=note,
        sample_id=sample_id,
        free_description=free_description,
        guided_description=guided_description,
        submitted_at=submitted_at,
    )
    return {"ok": True}


if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
