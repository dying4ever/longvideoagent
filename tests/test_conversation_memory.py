"""Lightweight tests for Conversation Memory + coreference resolution (no model).

Run:  python tests/test_conversation_memory.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from memory import context_resolver  # noqa: E402
from memory.conversation_memory import ConversationMemory  # noqa: E402


def _mem() -> ConversationMemory:
    m = ConversationMemory(video_path="/tmp/v.mp4")
    m.register_entity("乔治", "male")
    m.register_entity("佩奇", "female")
    m.add_confirmed_event({
        "subject": "乔治", "predicate": "出现", "object": None,
        "timestamp": 6.0, "confidence": "high", "source_turn": 1,
        "fact": "乔治出现于6s", "evidence": [],
    })
    m.update_working_memory(current_subject="乔治")
    return m


def test_save_load() -> None:
    m = _mem()
    m.add_turn({"turn_id": 1, "question": "q", "answer": "a"})
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "cm.json")
        m.save(p)
        loaded = ConversationMemory.load(p)
        assert loaded is not None
        assert loaded.session_id == m.session_id
        assert len(loaded.turns) == 1
        assert loaded.entities["乔治"]["gender"] == "male"
    print("[ok] conversation memory save/load round-trip")


def test_confirmed_event_dedup() -> None:
    m = ConversationMemory()
    m.add_confirmed_event({"subject": "乔治", "predicate": "出现", "timestamp": 6.0})
    m.add_confirmed_event({"subject": "乔治", "predicate": "出现", "timestamp": 6.0})
    assert len(m.confirmed_events) == 1
    print("[ok] confirmed event dedup")


def test_tentative_vs_verified() -> None:
    m = ConversationMemory()
    m.add_confirmed_event({"subject": "乔治", "predicate": "出现", "timestamp": 6.0})
    m.add_tentative_event({"subject": "乔治", "predicate": "跳舞", "timestamp": 30.0})
    assert m.confirmed_events[0]["verification_status"] == "verified"
    assert m.tentative_events[0]["verification_status"] == "tentative"
    assert len(m.confirmed_events) == 1
    print("[ok] tentative vs verified separation")


def test_coref_he() -> None:
    r = context_resolver.resolve_question("他出现之后做了什么？", _mem())
    assert r["resolved_question"] == "乔治出现之后做了什么？"
    assert r["status"] == "resolved"
    print("[ok] '他' resolves to current male subject")


def test_coref_she() -> None:
    m = _mem()
    m.update_working_memory(current_subject="佩奇")
    r = context_resolver.resolve_question("她做了什么？", m)
    assert r["resolved_question"] == "佩奇做了什么？"
    print("[ok] '她' resolves to female subject")


def test_coref_ambiguous() -> None:
    m = ConversationMemory()  # no entities
    r = context_resolver.resolve_question("他在做什么？", m)
    assert r["status"] == "ambiguous"
    print("[ok] ambiguous pronoun does not guess")


if __name__ == "__main__":
    test_save_load()
    test_confirmed_event_dedup()
    test_tentative_vs_verified()
    test_coref_he()
    test_coref_she()
    test_coref_ambiguous()
    print("ALL CONVERSATION MEMORY TESTS PASSED")
