"""Lightweight multi-turn session tests (no model weights).

Verifies that turn 2 resolves coreferences and reuses turn-1 facts, and that
reset clears the conversation memory.

Run:  python tests/test_multiturn.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import agent  # noqa: E402
from agents import grounding, reasoning  # noqa: E402
from memory import video_memory  # noqa: E402
from session import LongVideoAgentSession  # noqa: E402
from temporal import parser as temporal_parser  # noqa: E402


def _mem():
    return {
        "video_path": "/tmp/v.mp4",
        "duration": 285.0,
        "global_summary": "佩奇一家",
        "chapters": [],
        "segments": [
            {"segment_id": i, "start": i * 60.0, "end": (i + 1) * 60.0, "summary": f"s{i}", "events": []}
            for i in range(5)
        ],
    }


def _session():
    with mock.patch.object(video_memory, "load_video_memory", return_value=_mem()):
        return LongVideoAgentSession("/tmp/v.mp4")


def test_multiturn_reuse_reference() -> None:
    targets = [
        {"target": {"subject": "乔治", "gender": "male", "predicate": "出现", "object": None}, "reference_event": None},
        {"target": {"subject": "乔治", "gender": "male", "predicate": "做什么", "object": None},
         "reference_event": {"subject": "乔治", "predicate": "出现", "gender": "male", "object": None}},
    ]
    agent_result = {
        "current_answer": "首次出现时间约在 6.0s", "final_timestamp": 6.0,
        "status": "finished", "evidence": [], "trace": [], "searched_intervals": [],
    }
    reason_result = {
        "answer": "乔治出现后和佩奇一起玩", "occurrences": [], "evidence": [],
        "inspected_intervals": [{"start": 6.0, "end": 285.0}],
    }
    ground_result = {"query": "q", "candidates": [{"start": 6.0, "end": 285.0, "score": 0.9, "reason": "r"}]}
    with mock.patch.object(video_memory, "load_video_memory", return_value=_mem()), \
         mock.patch.object(temporal_parser, "extract_target_reference", side_effect=targets), \
         mock.patch.object(agent, "run_agent", return_value=agent_result), \
         mock.patch.object(reasoning, "reason_over_candidates", return_value=reason_result), \
         mock.patch.object(grounding, "ground_video", return_value=ground_result):
        session = LongVideoAgentSession("/tmp/v.mp4")
        r1 = session.ask("乔治第一次什么时候出现？")
        r2 = session.ask("他出现之后做了什么？")

    assert r1["temporal_type"] == "FIRST"
    assert session.conversation.entities.get("乔治") is not None
    assert len(session.conversation.confirmed_events) == 1

    assert r2["resolved_question"] == "乔治出现之后做了什么？"
    assert r2["temporal_type"] == "AFTER"
    assert r2["reference_timestamp"] == 6.0
    print("[ok] turn 2 resolves '他' and reuses turn-1 reference timestamp 6s")


def test_reset_clears_memory() -> None:
    session = _session()
    session.conversation.register_entity("乔治", "male")
    session.conversation.add_turn({"turn_id": 1, "question": "q", "answer": "a"})
    assert len(session.conversation.turns) == 1
    session.reset()
    assert len(session.conversation.turns) == 0
    assert session.conversation.entities == {}
    print("[ok] reset clears conversation memory")


if __name__ == "__main__":
    test_multiturn_reuse_reference()
    test_reset_clears_memory()
    print("ALL MULTITURN TESTS PASSED")
