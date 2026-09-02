"""Visual Reasoning: observe a LOCAL interval and report local occurrences.

Reasoning only answers "what happened in THIS interval and when" — it must
never claim whether an event is the first/last/only/always occurrence across
the whole video. Global temporal semantics are handled by the Temporal
Verifier, not by this module.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from tools import video_tool, vlm_tool
from utils import profiler


def _build_reasoning_prompt(question: str, frames) -> str:
    lines = [
        "You are observing a LOCAL time interval of a video (NOT the whole video).",
        "Each frame has an EXPLICIT timestamp (in seconds) provided by the program; it is the REAL frame time.",
        "",
        f"Question: {question}",
        "",
        "Frames (chronological order):",
    ]
    for i, f in enumerate(frames, 1):
        lines.append(f"Frame {i}: timestamp = {f.timestamp} s")
    lines += [
        "",
        "Instructions:",
        "1. Observe the frames carefully in chronological order.",
        "2. Report the LOCAL occurrences of the subject/event with their timestamps.",
        "3. You ONLY see this local interval. Do NOT claim whether this is the FIRST, LAST, ONLY, or EVERY occurrence across the WHOLE video.",
        "4. Do NOT fabricate events you cannot see.",
        "5. If the subject is not observed in this interval, return an empty occurrences list.",
        "6. If the question asks whether something is 'always' true (e.g. '一直/始终'), also report "
        "'violations': times when the subject is PRESENT but the condition is NOT met.",
        "7. Return ONLY valid JSON matching this schema:",
        json.dumps({
            "answer": "one-sentence LOCAL observation",
            "occurrences": [
                {"timestamp": 0.0, "event": "what happened", "confidence": "high|medium|low"}
            ],
            "violations": [
                {"timestamp": 0.0, "event": "subject present but condition violated", "confidence": "high|medium|low"}
            ],
            "evidence": [{"timestamp": 0.0, "description": "what is observed at that time"}],
        }, ensure_ascii=False),
    ]
    return "\n".join(lines)


@profiler.timed("reasoning")
def reason_over_candidates(
    video_path: str,
    question: str,
    candidates: List[Dict[str, Any]],
    fine_interval: float = 2.0,
) -> Dict[str, Any]:
    """Observe each candidate interval and return local occurrences + evidence.

    Returns {"answer", "occurrences", "evidence", "inspected_intervals"}.
    `occurrences` are local events with timestamps; an empty list means the
    subject was not observed in that interval.
    """
    model = vlm_tool.load_model()
    occurrences: List[Dict[str, Any]] = []
    violations: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []
    inspected: List[Dict[str, float]] = []
    answers: List[str] = []

    for cand in candidates:
        start, end = float(cand["start"]), float(cand["end"])
        inspected.append({"start": start, "end": end})
        frames = video_tool.sample_frames(
            video_path, start_time=start, end_time=end, interval=fine_interval
        )
        if not frames:
            continue
        prompt = _build_reasoning_prompt(question, frames)
        result = vlm_tool.analyze_frames(
            frames, prompt, model=model.model, processor=model.processor
        )
        answers.append(str(result.get("answer", "")))
        for occ in result.get("occurrences", []):
            try:
                ts = float(occ.get("timestamp", 0.0))
            except (TypeError, ValueError):
                continue
            occurrences.append({
                "timestamp": ts,
                "description": str(occ.get("event", occ.get("description", ""))),
                "confidence": str(occ.get("confidence", "medium")),
            })
        for v in result.get("violations", []):
            try:
                ts = float(v.get("timestamp", 0.0))
            except (TypeError, ValueError):
                continue
            violations.append({
                "timestamp": ts,
                "description": str(v.get("event", v.get("description", ""))),
                "confidence": str(v.get("confidence", "medium")),
            })
        for ev in result.get("evidence", []):
            try:
                ts = float(ev.get("timestamp", 0.0))
            except (TypeError, ValueError):
                ts = 0.0
            evidence.append({"timestamp": ts, "description": str(ev.get("description", ""))})

    occurrences.sort(key=lambda o: o["timestamp"])
    violations.sort(key=lambda v: v["timestamp"])
    evidence.sort(key=lambda e: e["timestamp"])
    answer = "；".join(a for a in answers if a) or "无法确定（未获取到有效画面）"
    return {
        "answer": answer,
        "occurrences": occurrences,
        "violations": violations,
        "evidence": evidence,
        "inspected_intervals": inspected,
    }
