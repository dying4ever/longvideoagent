"""Conversation Memory: long-term multi-turn memory + short-term Working Memory.

Stores turns, entities, confirmed/tentative events, and a bounded Working
Memory that tracks the active context (StreamAgent-style). No hidden
chain-of-thought is stored — only facts, references, answers and evidence.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

WORKING_QUESTION_WINDOW = 4


class ConversationMemory:
    def __init__(self, session_id: Optional[str] = None, video_path: Optional[str] = None):
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.video_path = video_path
        self.turns: List[Dict[str, Any]] = []
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.confirmed_events: List[Dict[str, Any]] = []
        self.tentative_events: List[Dict[str, Any]] = []
        self.references: Dict[str, Any] = {}
        self.working_memory: Dict[str, Any] = {
            "active_entities": [],
            "current_subject": None,
            "reference_event": None,
            "temporal_constraint": None,
            "recent_intervals": [],
            "recent_questions": [],
            "unresolved_targets": [],
        }
        self.created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.updated_at = self.created_at

    # ---- serialization ----

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "video_path": self.video_path,
            "turns": self.turns,
            "entities": self.entities,
            "confirmed_events": self.confirmed_events,
            "tentative_events": self.tentative_events,
            "references": self.references,
            "working_memory": self.working_memory,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConversationMemory":
        m = cls(session_id=d.get("session_id"), video_path=d.get("video_path"))
        m.turns = d.get("turns", [])
        m.entities = d.get("entities", {})
        m.confirmed_events = d.get("confirmed_events", [])
        m.tentative_events = d.get("tentative_events", [])
        m.references = d.get("references", {})
        m.working_memory = d.get("working_memory", {})
        m.created_at = d.get("created_at", m.created_at)
        m.updated_at = d.get("updated_at", m.updated_at)
        return m

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> Optional["ConversationMemory"]:
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ---- turns ----

    def add_turn(self, turn: Dict[str, Any]) -> None:
        self.turns.append(turn)
        self._touch()

    # ---- entities ----

    def register_entity(self, name: str, gender: Optional[str] = None) -> None:
        if name not in self.entities:
            self.entities[name] = {"name": name, "gender": gender, "first_seen_turn": len(self.turns)}
        if gender and self.entities[name].get("gender") is None:
            self.entities[name]["gender"] = gender
        if name not in self.working_memory["active_entities"]:
            self.working_memory["active_entities"].append(name)
        self._touch()

    def known_entities(self) -> List[str]:
        return list(self.entities.keys())

    # ---- events ----

    def add_confirmed_event(self, event: Dict[str, Any]) -> None:
        event = dict(event)
        event.setdefault("event_id", f"event_{len(self.confirmed_events) + len(self.tentative_events) + 1:03d}")
        event["verification_status"] = "verified"
        key = (event.get("subject"), event.get("predicate"), round(float(event.get("timestamp", 0.0)), 3))
        if any((e.get("subject"), e.get("predicate"), round(float(e.get("timestamp", 0.0)), 3)) == key
               for e in self.confirmed_events):
            return
        self.confirmed_events.append(event)
        self._touch()

    def add_tentative_event(self, event: Dict[str, Any]) -> None:
        event = dict(event)
        event["verification_status"] = "tentative"
        self.tentative_events.append(event)
        self._touch()

    def lookup_reference_event(self, phrase: str) -> Optional[Dict[str, Any]]:
        for e in self.confirmed_events:
            subj = e.get("subject", "")
            pred = e.get("predicate", "")
            if subj and pred and subj in phrase and pred in phrase:
                return e
        return None

    def current_subject(self) -> Optional[str]:
        return self.working_memory.get("current_subject")

    # ---- working memory ----

    def update_working_memory(self, **updates: Any) -> None:
        self.working_memory.update(updates)
        self._trim_working_memory()
        self._touch()

    def _trim_working_memory(self) -> None:
        qs = self.working_memory.get("recent_questions", [])
        if len(qs) > WORKING_QUESTION_WINDOW:
            self.working_memory["recent_questions"] = qs[-WORKING_QUESTION_WINDOW:]
        ivs = self.working_memory.get("recent_intervals", [])
        if len(ivs) > WORKING_QUESTION_WINDOW:
            self.working_memory["recent_intervals"] = ivs[-WORKING_QUESTION_WINDOW:]

    def _touch(self) -> None:
        self.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
