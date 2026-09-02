"""Visual Critic: judge whether the current answer + evidence are sufficient.

The Critic never re-reads the video. It judges sufficiency purely from the
question, answer, evidence, searched intervals and video duration. When in
doubt it returns insufficient rather than guessing.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from tools import vlm_tool


def _build_critic_prompt(
    question: str,
    answer: Any,
    evidence: List[Dict[str, Any]],
    searched_intervals: List[Dict[str, Any]],
    video_duration: float,
) -> str:
    lines = [
        "You are the Critic in a video QA agent.",
        "You judge whether the current answer is sufficiently supported by the evidence.",
        "You do NOT re-read the video; you only reason about the provided information.",
        "",
        f"Video duration: {video_duration}s",
        f"Question: {question}",
        f"Answer: {answer}",
        "",
        "Evidence collected so far (timestamp -> description):",
    ]
    if not evidence:
        lines.append("  (none)")
    for ev in evidence:
        lines.append(f"  {ev['timestamp']}s: {ev['description']}")
    lines += [
        "",
        "Intervals already searched (seconds):",
    ]
    if not searched_intervals:
        lines.append("  (none)")
    for iv in searched_intervals:
        lines.append(f"  [{iv['start']}s - {iv['end']}s]")
    lines += [
        "",
        "Check carefully:",
        "1. Does the answer DIRECTLY answer the question?",
        "2. Does the evidence actually support the answer?",
        "3. For 'first/last/before/after/always/again' questions, has the necessary time range been checked?",
        "4. For temporal localization, is there enough before/after evidence?",
        "5. Do multiple observations conflict with each other?",
        "6. Did the model infer beyond the visual evidence?",
        "",
        "If you cannot confirm sufficiency, return insufficient.",
        "",
        "Return ONLY valid JSON matching this schema:",
        json.dumps({
            "sufficient": False,
            "reason": "why insufficient / why sufficient",
            "missing_evidence": "what is still missing",
            "suggested_action": "search_earlier | search_later | inspect_specific | none",
            "suggested_range": {"start": 0.0, "end": 120.0},
        }, ensure_ascii=False),
    ]
    return "\n".join(lines)


def critique_answer(
    question: str,
    answer: Any,
    evidence: List[Dict[str, Any]],
    searched_intervals: List[Dict[str, Any]],
    video_duration: float,
) -> Dict[str, Any]:
    """Return {sufficient, reason, missing_evidence, suggested_action, suggested_range}."""
    model = vlm_tool.load_model()
    prompt = _build_critic_prompt(question, answer, evidence, searched_intervals, video_duration)
    raw = vlm_tool.generate_text(prompt, model=model.model, processor=model.processor)
    result = vlm_tool._extract_json(raw)

    sufficient = bool(result.get("sufficient", False))
    return {
        "sufficient": sufficient,
        "reason": str(result.get("reason", "")),
        "missing_evidence": str(result.get("missing_evidence", "")),
        "suggested_action": str(result.get("suggested_action", "none")),
        "suggested_range": result.get("suggested_range"),
    }
