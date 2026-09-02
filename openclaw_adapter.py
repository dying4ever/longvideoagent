"""OpenClaw adapter: thin bridge between OpenClaw and LongVideoAgentSession.

OpenClaw skills do NOT re-implement the agent. They call this adapter, which
wraps the single LongVideoAgentSession — the same session used by the CLI and
the Web API.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from session import LongVideoAgentSession

_SESSIONS: Dict[str, LongVideoAgentSession] = {}


def create_session(video_path: str, **kwargs: Any) -> Dict[str, Any]:
    session_id = uuid.uuid4().hex[:12]
    session = LongVideoAgentSession(video_path, **kwargs)
    _SESSIONS[session_id] = session
    return {
        "session_id": session_id,
        "status": "ready",
        "video_path": video_path,
        "duration": session.duration,
        "n_segments": len(session.video_memory.get("segments", [])),
    }


def ask(session_id: str, question: str) -> Dict[str, Any]:
    session = _get(session_id)
    return session.ask(question)


def get_trace(session_id: str) -> Dict[str, Any]:
    session = _get(session_id)
    turns = session.conversation.turns
    if not turns:
        return {"turns": [], "last_trace": []}
    return {"turns": turns, "last_trace": turns[-1].get("trace", [])}


def get_memory(session_id: str) -> Dict[str, Any]:
    session = _get(session_id)
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


def reset_session(session_id: str) -> Dict[str, Any]:
    session = _get(session_id)
    session.reset()
    return {"session_id": session_id, "status": "reset"}


def _get(session_id: str) -> LongVideoAgentSession:
    if session_id not in _SESSIONS:
        raise KeyError(f"session not found: {session_id}")
    return _SESSIONS[session_id]
