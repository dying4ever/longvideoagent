"""LongVideoAgentSession: multi-turn interactive long-video QA.

Holds the video (built once), hierarchical video memory, conversation memory
and working memory. Each `ask()` resolves coreferences, reuses previously
verified facts, runs the agent (or a direct constrained reasoning for
BEFORE/AFTER with a known reference), and updates the memories.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import agent
from agents import grounding, reasoning
from memory import context_resolver, conversation_memory, video_memory
from temporal import parser as temporal_parser
from temporal.parser import AFTER, BEFORE, FIRST, LAST, NORMAL, REPEAT


class LongVideoAgentSession:
    def __init__(
        self,
        video_path: str,
        window_size: float = 60.0,
        frame_interval: float = 10.0,
        top_k: int = 2,
        fine_interval: float = 3.0,
        max_iterations: int = 6,
    ):
        self.video_path = video_path
        self.window_size = window_size
        self.frame_interval = frame_interval
        self.top_k = top_k
        self.fine_interval = fine_interval
        self.max_iterations = max_iterations
        self.conversation = conversation_memory.ConversationMemory(video_path=video_path)
        self.video_memory: Dict[str, Any] = self._load_or_build_memory()
        self.duration = float(self.video_memory.get("duration", 0.0))

    def _load_or_build_memory(self) -> Dict[str, Any]:
        mem_path = video_memory.default_memory_path(self.video_path)
        existing = video_memory.load_video_memory(mem_path)
        if existing is not None and video_memory.memory_matches(
            existing, self.video_path, self.window_size, 5.0
        ):
            return existing
        return video_memory.build_event_memory(
            self.video_path, window_size=self.window_size, frame_interval=5.0
        )

    # ---- main entry ----

    def ask(self, question: str) -> Dict[str, Any]:
        turn_id = len(self.conversation.turns) + 1

        resolved = context_resolver.resolve_question(question, self.conversation)
        resolved_question = resolved["resolved_question"]

        parsed = temporal_parser.parse_temporal_query(resolved_question)
        ttype = parsed["type"]

        tr = temporal_parser.extract_target_reference(resolved_question)
        target = tr.get("target")
        reference_event = tr.get("reference_event")

        if target and target.get("subject"):
            gender = target.get("gender") if target.get("gender") != "unknown" else None
            self.conversation.register_entity(target["subject"], gender)

        ref_ts = self._resolve_reference_timestamp(ttype, reference_event)

        result = self._execute(resolved_question, ttype, ref_ts, turn_id)
        result["question"] = question
        result["resolved_question"] = resolved_question
        result["temporal_type"] = ttype
        result["reference_timestamp"] = ref_ts

        self._update_memory(question, resolved_question, ttype, target, reference_event, ref_ts, result, turn_id)
        return result

    # ---- internal ----

    def _resolve_reference_timestamp(self, ttype: str, reference_event: Optional[Dict]) -> Optional[float]:
        if ttype not in (BEFORE, AFTER) or not reference_event:
            return None
        phrase = f"{reference_event.get('subject', '')}{reference_event.get('predicate', '')}"
        hit = self.conversation.lookup_reference_event(phrase)
        if hit:
            return float(hit["timestamp"])
        # fall back to a confirmed event whose subject matches the reference subject
        subj = reference_event.get("subject")
        if subj:
            for e in self.conversation.confirmed_events:
                if e.get("subject") == subj:
                    return float(e["timestamp"])
        return None

    def _execute(self, question: str, ttype: str, ref_ts: Optional[float], turn_id: int) -> Dict[str, Any]:
        if ttype in (BEFORE, AFTER) and ref_ts is not None:
            if ttype == BEFORE:
                rng = [0.0, ref_ts]
            else:
                rng = [ref_ts, self.duration]
            g = grounding.ground_video(
                question, self.video_memory, top_k=self.top_k,
                search_start=rng[0], search_end=rng[1],
            )
            candidates = g.get("candidates", [])
            if not candidates:
                candidates = [{"start": rng[0], "end": rng[1]}]
            res = reasoning.reason_over_candidates(
                self.video_path, question, candidates, fine_interval=self.frame_interval
            )
            res["status"] = "finished"
            res["current_answer"] = self._grounded_answer(res.get("evidence", []), ref_ts)
            res["final_timestamp"] = ref_ts
            res["trace"] = [
                {"step": 1, "agent": "context_resolver", "reference_timestamp": ref_ts},
                {"step": 2, "agent": "grounding", "candidates": candidates},
                {"step": 3, "agent": "reasoning", "interval": rng},
            ]
            res["inspected_intervals"] = [{"start": c["start"], "end": c["end"]} for c in candidates]
            return res
        return agent.run_agent(
            question, self.video_path, self.video_memory,
            max_iterations=self.max_iterations, top_k=self.top_k, fine_interval=self.fine_interval,
            initial_occurrences=self._known_occurrences(),
        )

    def _grounded_answer(self, evidence: List[Dict[str, Any]], ref_ts: float) -> str:
        if not evidence:
            return "该时间范围内未观察到相关画面"
        parts = [f"{ev['timestamp']:.0f}s {ev['description']}" for ev in evidence[:6]]
        return "；".join(parts)

    def _known_occurrences(self) -> List[Dict[str, Any]]:
        return [
            {"timestamp": float(e["timestamp"]),
             "description": f"{e.get('subject', '')}{e.get('predicate', '')}",
             "confidence": str(e.get("confidence", "high"))}
            for e in self.conversation.confirmed_events
        ]

    def _update_memory(
        self, question, resolved_question, ttype, target, reference_event, ref_ts, result, turn_id,
    ) -> None:
        answer = result.get("current_answer")
        timestamp = result.get("final_timestamp") or ref_ts

        self.conversation.add_turn({
            "turn_id": turn_id,
            "question": question,
            "resolved_question": resolved_question,
            "answer": answer,
            "temporal_type": ttype,
            "timestamp": timestamp,
            "evidence": result.get("evidence", []),
            "trace": result.get("trace", []),
        })

        if target and target.get("subject") and result.get("status") in ("finished", "verified", "max_iterations_reached"):
            subj = target["subject"]
            pred = target.get("predicate") or ""
            if ttype in (FIRST, LAST) and timestamp is not None:
                self.conversation.add_confirmed_event({
                    "subject": subj, "predicate": pred, "object": target.get("object"),
                    "timestamp": timestamp, "confidence": "high", "source_turn": turn_id,
                    "fact": f"{subj}{pred}于{timestamp}s",
                    "evidence": result.get("evidence", []),
                })

        self.conversation.update_working_memory(
            current_subject=(target.get("subject") if target else None),
            reference_event=({"event": f"{reference_event.get('subject','')}{reference_event.get('predicate','')}", "timestamp": ref_ts}
                             if reference_event else None),
            recent_questions=(self.conversation.working_memory.get("recent_questions", []) + [question]),
            recent_intervals=(self.conversation.working_memory.get("recent_intervals", [])
                              + result.get("inspected_intervals", result.get("searched_intervals", []))),
        )

    # ---- session control ----

    def reset(self) -> None:
        self.conversation = conversation_memory.ConversationMemory(video_path=self.video_path)

    def save(self, path: str) -> str:
        return self.conversation.save(path)

    def load(self, path: str) -> None:
        loaded = conversation_memory.ConversationMemory.load(path)
        if loaded is not None:
            self.conversation = loaded
