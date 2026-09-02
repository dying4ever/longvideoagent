"""Visual Reasoning: densely observe candidate intervals to answer a question.

Memory only locates candidate intervals; reasoning re-reads the ORIGINAL video
frames inside each candidate at a finer interval, collects timestamped visual
evidence, and synthesizes a final answer. It never trusts the memory summary
for the actual answer.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from tools import video_tool, vlm_tool


def _build_reasoning_prompt(question: str, frames) -> str:
    lines = [
        "You are answering a question about a video by closely observing the frames.",
        "Each frame has an EXPLICIT timestamp (in seconds) provided by the program;",
        "it is the REAL frame time, not something you should infer.",
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
        "2. Use the provided timestamps to locate when events happen.",
        "3. Do NOT fabricate events you cannot see in the frames.",
        "4. You only see a SUBSET of the video; do NOT claim something is the 'first/last/only' occurrence unless the evidence clearly supports it.",
        "5. If you cannot determine the answer, say so explicitly and set confidence to low.",
        "6. Return ONLY valid JSON matching this schema:",
        json.dumps({
            "answer": "your answer, or '不确定' if unknown",
            "confidence": "high|medium|low",
            "evidence": [{"timestamp": 0.0, "description": "what is observed at that time"}],
        }, ensure_ascii=False),
    ]
    return "\n".join(lines)


def _build_synthesis_prompt(question: str, partials: List[Dict[str, Any]]) -> str:
    lines = [
        "You are synthesizing a final answer from multiple observations of a video.",
        "",
        f"Question: {question}",
        "",
        "Observations from different time intervals:",
    ]
    for i, p in enumerate(partials, 1):
        lines.append(
            f"- Interval {i}: answer='{p.get('answer', '')}' confidence='{p.get('confidence', '')}'"
        )
    lines += [
        "",
        "Instructions:",
        "1. Combine the observations into ONE final answer.",
        "2. If the observations conflict or are insufficient, say so.",
        "3. Return ONLY valid JSON matching this schema:",
        json.dumps({"answer": "...", "confidence": "high|medium|low"}, ensure_ascii=False),
    ]
    return "\n".join(lines)


def reason_over_candidates(
    video_path: str,
    question: str,
    candidates: List[Dict[str, Any]],
    fine_interval: float = 2.0,
) -> Dict[str, Any]:
    """Densely observe each candidate interval and synthesize an answer."""
    model = vlm_tool.load_model()
    partials: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []
    inspected: List[Dict[str, float]] = []

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
        partials.append(result)
        for ev in result.get("evidence", []):
            try:
                ts = float(ev.get("timestamp", 0.0))
            except (TypeError, ValueError):
                ts = 0.0
            evidence.append({"timestamp": ts, "description": str(ev.get("description", ""))})

    if not partials:
        return {
            "answer": "无法确定（未获取到候选区间的有效画面）",
            "confidence": "low",
            "evidence": [],
            "inspected_intervals": inspected,
        }

    if len(partials) == 1:
        answer = partials[0].get("answer", "")
        confidence = partials[0].get("confidence", "low")
    else:
        synth_prompt = _build_synthesis_prompt(question, partials)
        raw = vlm_tool.generate_text(
            synth_prompt, model=model.model, processor=model.processor
        )
        synth = vlm_tool._extract_json(raw)
        answer = synth.get("answer", partials[0].get("answer", ""))
        confidence = synth.get("confidence", "low")

    evidence.sort(key=lambda e: e["timestamp"])
    return {
        "answer": answer,
        "confidence": confidence,
        "evidence": evidence,
        "inspected_intervals": inspected,
    }
