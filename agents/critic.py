"""Visual Critic: rule-based temporal check + LLM semantic check.

Temporal sufficiency (first/last/always coverage) is decided by the Temporal
Verifier (pure Python rules), NOT by the LLM. The LLM is only consulted to
check visual-semantic validity: does the evidence actually support the claim,
is there hallucination, are there visual contradictions.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from temporal import verifier
from temporal.parser import NORMAL
from tools import vlm_tool
from utils import profiler


def _build_semantic_prompt(
    question: str,
    answer: Any,
    evidence: List[Dict[str, Any]],
) -> str:
    lines = [
        "You are checking whether a video-QA answer is visually plausible.",
        "You do NOT judge temporal coverage (first/last/always) — that is already handled.",
        "You only check whether the visual evidence genuinely supports the answer.",
        "",
        f"Question: {question}",
        f"Answer: {answer}",
        "",
        "Evidence (timestamp -> description):",
    ]
    if not evidence:
        lines.append("  (none)")
    for ev in evidence:
        lines.append(f"  {ev['timestamp']}s: {ev['description']}")
    lines += [
        "",
        "Check:",
        "1. Does the evidence support the answer, or is the answer hallucinated?",
        "2. Is there any visual contradiction?",
        "3. Is the confidence warranted by the evidence?",
        "4. NOTE: the answer is a GLOBAL claim about the WHOLE video. For a 'first' answer, "
        "LATER occurrences at other timestamps do NOT contradict an EARLIER 'first' timestamp; "
        "they are simply later appearances.",
        "",
        "Return ONLY valid JSON matching this schema:",
        json.dumps({
            "sufficient": True,
            "reason": "why the visual evidence is / is not trustworthy",
            "missing_evidence": "",
            "suggested_action": "reobserve | none",
            "suggested_range": None,
        }, ensure_ascii=False),
    ]
    return "\n".join(lines)


@profiler.timed("critic")
def critique_answer(
    question: str,
    answer: Any,
    evidence: List[Dict[str, Any]],
    searched_intervals: List[Dict[str, Any]],
    video_duration: float,
    temporal_type: str = NORMAL,
    occurrences: Optional[List[Dict[str, Any]]] = None,
    verified_absence: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Return {sufficient, reason, missing_evidence, suggested_action, suggested_range}."""
    occurrences = occurrences or []
    verified_absence = verified_absence or []

    if temporal_type != NORMAL:
        rule = verifier.verify_temporal_condition(
            temporal_type, occurrences, searched_intervals, verified_absence, video_duration
        )
        if not rule["sufficient"]:
            return {
                "sufficient": False,
                "reason": rule["reason"],
                "missing_evidence": f"需排查区间: {rule['missing_ranges']}",
                "suggested_action": "search",
                "suggested_range": rule["missing_ranges"][0] if rule["missing_ranges"] else None,
            }

    model = vlm_tool.load_model()
    prompt = _build_semantic_prompt(question, answer, evidence)
    raw = vlm_tool.generate_text(prompt, model=model.model, processor=model.processor)
    result = vlm_tool._extract_json(raw)
    return {
        "sufficient": bool(result.get("sufficient", False)),
        "reason": str(result.get("reason", "")),
        "missing_evidence": str(result.get("missing_evidence", "")),
        "suggested_action": str(result.get("suggested_action", "none")),
        "suggested_range": result.get("suggested_range"),
    }
