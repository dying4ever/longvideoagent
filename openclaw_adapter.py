"""OpenClaw adapter: thin bridge between OpenClaw and LongVideoAgentSession.

OpenClaw skills do NOT re-implement the agent. They call this adapter, which
wraps the single LongVideoAgentSession — the same session used by the CLI and
the Web API.
"""
from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, Optional

import config
from session import LongVideoAgentSession

_SESSIONS: Dict[str, LongVideoAgentSession] = {}


def _session_file(video_path: str) -> str:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return str(config.DATA_DIR / "memory" / f"{stem}_session.json")


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


def main() -> None:
    """CLI entry: OpenClaw runs this to ask LongVideoAgent a question.

    The conversation memory is persisted next to the video memory, so repeated
    invocations (one per turn) reuse entities / reference events / occurrences.
    """
    import argparse

    p = argparse.ArgumentParser(description="LongVideoAgent CLI (OpenClaw bridge)")
    p.add_argument("--video", required=True)
    p.add_argument("--question", required=True)
    args = p.parse_args()

    session = LongVideoAgentSession(args.video)
    session.load(_session_file(args.video))
    result = session.ask(args.question)
    session.save(_session_file(args.video))

    print(json.dumps({
        "answer": result.get("current_answer"),
        "status": result.get("status"),
        "timestamp": result.get("final_timestamp"),
        "temporal_type": result.get("temporal_type"),
        "resolved_question": result.get("resolved_question"),
        "evidence": result.get("evidence", []),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
