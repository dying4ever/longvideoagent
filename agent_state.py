"""Agent state: unified per-question state with an execution trace."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class AgentState:
    """Tracks a single QA run: intervals searched, evidence accumulated,
    reasoning history, critic feedback, and a step-by-step trace."""

    def __init__(self, question: str, max_iterations: int = 5, video_duration: Optional[float] = None):
        self.question = question
        self.iteration = 0
        self.max_iterations = max_iterations
        self.video_duration = video_duration
        self.searched_intervals: List[Dict[str, float]] = []
        self.grounding_history: List[Dict[str, Any]] = []
        self.reasoning_history: List[Dict[str, Any]] = []
        self.evidence: List[Dict[str, Any]] = []
        self.current_answer: Optional[str] = None
        self.critic_feedback: Optional[Dict[str, Any]] = None
        self.status: str = "running"
        self.trace: List[Dict[str, Any]] = []
        self.temporal_type: str = "NORMAL"
        self.event_occurrences: List[Dict[str, Any]] = []
        self.violations: List[Dict[str, Any]] = []
        self.verified_absence_intervals: List[Dict[str, float]] = []
        self.final_timestamp: Optional[float] = None
        self._step = 0

    def _next_step(self) -> int:
        self._step += 1
        return self._step

    def add_trace(self, agent: str, **kwargs: Any) -> None:
        entry = {"step": self._next_step(), "agent": agent}
        entry.update(kwargs)
        self.trace.append(entry)

    def add_searched_interval(self, start: float, end: float) -> None:
        iv = {"start": round(float(start), 3), "end": round(float(end), 3)}
        if iv not in self.searched_intervals:
            self.searched_intervals.append(iv)

    def add_evidence(self, evidence_list: List[Dict[str, Any]]) -> None:
        for ev in evidence_list:
            try:
                ts = round(float(ev["timestamp"]), 3)
            except (KeyError, TypeError, ValueError):
                continue
            item = {"timestamp": ts, "description": str(ev.get("description", ""))}
            if item not in self.evidence:
                self.evidence.append(item)
        self.evidence.sort(key=lambda e: e["timestamp"])

    def add_occurrences(self, occurrences: List[Dict[str, Any]]) -> None:
        for occ in occurrences:
            try:
                ts = round(float(occ["timestamp"]), 3)
            except (KeyError, TypeError, ValueError):
                continue
            item = {
                "timestamp": ts,
                "description": str(occ.get("description", "")),
                "confidence": str(occ.get("confidence", "medium")),
            }
            if item not in self.event_occurrences:
                self.event_occurrences.append(item)
        self.event_occurrences.sort(key=lambda o: o["timestamp"])

    def add_verified_absence(self, start: float, end: float) -> None:
        iv = {"start": round(float(start), 3), "end": round(float(end), 3)}
        if iv not in self.verified_absence_intervals:
            self.verified_absence_intervals.append(iv)

    def add_violations(self, violations: List[Dict[str, Any]]) -> None:
        for v in violations:
            try:
                ts = round(float(v["timestamp"]), 3)
            except (KeyError, TypeError, ValueError):
                continue
            item = {
                "timestamp": ts,
                "description": str(v.get("description", "")),
                "confidence": str(v.get("confidence", "medium")),
            }
            if item not in self.violations:
                self.violations.append(item)
        self.violations.sort(key=lambda v: v["timestamp"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "video_duration": self.video_duration,
            "searched_intervals": self.searched_intervals,
            "grounding_history": self.grounding_history,
            "reasoning_history": self.reasoning_history,
            "evidence": self.evidence,
            "current_answer": self.current_answer,
            "critic_feedback": self.critic_feedback,
            "status": self.status,
            "trace": self.trace,
            "temporal_type": self.temporal_type,
            "event_occurrences": self.event_occurrences,
            "violations": self.violations,
            "verified_absence_intervals": self.verified_absence_intervals,
            "final_timestamp": self.final_timestamp,
        }
