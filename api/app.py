"""LongVideoAgent FastAPI backend.

Wraps LongVideoAgentSession — no business logic is duplicated here.
"""
from __future__ import annotations

import hashlib
import os
from typing import Dict

import config
import model_registry
from api import session_manager
from api.schemas import AskRequest, SessionCreateRequest
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title="LongVideoAgent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXT = {".mp4", ".mov", ".mkv", ".webm"}
MAX_SIZE = 300 * 1024 * 1024

_videos: Dict[str, str] = {}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():
    return {"models": model_registry.get_models()}


@app.get("/backend/status")
def backend_status():
    return model_registry.get_backend_status()


@app.post("/videos")
async def upload_video(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"unsupported file type: {ext}")
    data = await file.read()
    if len(data) > MAX_SIZE:
        raise HTTPException(413, "file too large")
    if not data:
        raise HTTPException(400, "empty file")
    # 内容哈希命名：相同视频只存一份（与 memory 缓存的 video_fingerprint 一致）
    video_id = hashlib.sha256(data).hexdigest()[:16]
    os.makedirs(config.VIDEO_DIR, exist_ok=True)
    path = str(config.VIDEO_DIR / f"{video_id}{ext}")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(data)
    _videos[video_id] = path
    return {"video_id": video_id, "filename": file.filename}


@app.post("/sessions")
def create_session(req: SessionCreateRequest):
    path = _videos.get(req.video_id)
    if not path:
        raise HTTPException(404, "video not found")
    try:
        sid, session = session_manager.create_session(path)
    except Exception as e:
        raise HTTPException(500, f"failed to build session: {e}")
    return {
        "session_id": sid,
        "status": "ready",
        "duration": session.duration,
        "n_segments": len(session.video_memory.get("segments", [])),
    }


@app.post("/sessions/{sid}/ask")
def ask(sid: str, req: AskRequest):
    try:
        session = session_manager.get_session(sid)
        result = session.ask(req.question)
    except KeyError:
        raise HTTPException(404, "session not found")
    except Exception as e:
        raise HTTPException(500, f"ask failed: {e}")
    cm = session.conversation
    return {
        "answer": result.get("current_answer"),
        "status": result.get("status"),
        "timestamp": result.get("final_timestamp"),
        "resolved_question": result.get("resolved_question"),
        "temporal_type": result.get("temporal_type"),
        "reference_timestamp": result.get("reference_timestamp"),
        "evidence": result.get("evidence", []),
        "trace": result.get("trace", []),
        "working_memory": cm.working_memory,
        "conversation_context": {
            "entities": list(cm.entities.keys()),
            "confirmed_events": cm.confirmed_events,
            "tentative_events": cm.tentative_events,
            "turns": cm.turns,
        },
    }


@app.get("/sessions/{sid}/memory")
def memory(sid: str):
    try:
        session = session_manager.get_session(sid)
    except KeyError:
        raise HTTPException(404, "session not found")
    cm = session.conversation
    return {
        "working_memory": cm.working_memory,
        "conversation": {
            "entities": cm.entities,
            "confirmed_events": cm.confirmed_events,
            "tentative_events": cm.tentative_events,
            "turns": cm.turns,
        },
        "video": {
            "global_summary": session.video_memory.get("global_summary"),
            "chapters": session.video_memory.get("chapters", []),
            "segments": session.video_memory.get("segments", []),
        },
    }


@app.get("/sessions/{sid}/trace")
def trace(sid: str):
    try:
        session = session_manager.get_session(sid)
    except KeyError:
        raise HTTPException(404, "session not found")
    turns = session.conversation.turns
    return {"turns": turns, "last_trace": turns[-1].get("trace", []) if turns else []}


@app.post("/sessions/{sid}/reset")
def reset(sid: str):
    try:
        session_manager.reset_session(sid)
    except KeyError:
        raise HTTPException(404, "session not found")
    return {"session_id": sid, "status": "reset"}


@app.get("/videos/{video_id}")
def get_video(video_id: str):
    path = _videos.get(video_id)
    if not path:
        raise HTTPException(404, "video not found")
    return FileResponse(path, media_type="video/mp4")
