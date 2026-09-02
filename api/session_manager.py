"""Session manager: keeps LongVideoAgentSession instances alive across requests.

A session persists (model + video memory + conversation memory stay loaded),
so repeated asks reuse everything instead of reloading Qwen per request.
"""
from __future__ import annotations

import uuid
from typing import Dict, Tuple

from session import LongVideoAgentSession

_sessions: Dict[str, LongVideoAgentSession] = {}


def create_session(video_path: str, progress_cb=None, **kwargs) -> Tuple[str, LongVideoAgentSession]:
    sid = uuid.uuid4().hex[:12]
    session = LongVideoAgentSession(video_path, progress_cb=progress_cb, **kwargs)
    _sessions[sid] = session
    return sid, session


def get_session(sid: str) -> LongVideoAgentSession:
    if sid not in _sessions:
        raise KeyError(f"session not found: {sid}")
    return _sessions[sid]


def reset_session(sid: str) -> LongVideoAgentSession:
    session = get_session(sid)
    session.reset()
    return session


def session_ids() -> list:
    return list(_sessions.keys())
