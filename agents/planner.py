"""Planner: decide the next action based on question + state.

The Planner never sees video frames and never produces visual facts. It only
picks the next step from a fixed action set, given the memory overview, current
evidence, searched intervals, critic feedback and iteration count.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from temporal import verifier
from temporal.parser import NORMAL
from tools import vlm_tool

VALID_ACTIONS = ("ground_video", "inspect_interval", "verify_answer", "finish")

_MAX_MEMORY_SUMMARY = 30


def _memory_overview(video_memory: Dict[str, Any]) -> List[Dict[str, Any]]:
    segs = video_memory.get("segments", [])
    return [
        {"start": s["start"], "end": s["end"], "summary": s.get("summary", "")}
        for s in segs[:_MAX_MEMORY_SUMMARY]
    ]


def _temporal_plan(question: str, state: Dict[str, Any], video_memory: Dict[str, Any]) -> Dict[str, Any]:
    duration = video_memory.get("duration", state.get("video_duration", 0.0))
    ver = verifier.verify_temporal_condition(
        state.get("temporal_type", NORMAL),
        state.get("event_occurrences", []),
        state.get("searched_intervals", []),
        state.get("verified_absence_intervals", []),
        duration,
    )
    if not state.get("event_occurrences"):
        return {"action": "ground_video", "query": question,
                "search_range": {"start": 0.0, "end": duration}, "reason": ver["reason"]}
    if ver["sufficient"]:
        return {"action": "verify_answer", "query": question,
                "search_range": None, "reason": ver["reason"]}
    if ver["missing_ranges"]:
        mr = ver["missing_ranges"][0]
        return {"action": "inspect_interval", "query": question,
                "search_range": {"start": mr[0], "end": mr[1]}, "reason": ver["reason"]}
    return {"action": "verify_answer", "query": question, "search_range": None, "reason": ver["reason"]}


def _build_planner_prompt(
    question: str,
    state: Dict[str, Any],
    video_memory: Dict[str, Any],
) -> str:
    duration = video_memory.get("duration", state.get("video_duration", 0.0))
    overview = _memory_overview(video_memory)

    lines = [
        "You are the Planner in a video QA agent.",
        "You decide the NEXT action only. You do NOT answer the question and do NOT see video frames.",
        "",
        f"Video duration: {duration}s",
        f"Question: {question}",
        "",
        "Video memory overview (coarse segment summaries):",
    ]
    for seg in overview:
        lines.append(f"  [{seg['start']}s - {seg['end']}s] {seg['summary']}")
    lines += [
        "",
        "Current state:",
        f"  iteration: {state.get('iteration')} / {state.get('max_iterations')}",
        f"  searched_intervals: {json.dumps(state.get('searched_intervals', []), ensure_ascii=False)}",
        f"  evidence_count: {len(state.get('evidence', []))}",
        f"  current_answer: {state.get('current_answer')}",
        f"  critic_feedback: {json.dumps(state.get('critic_feedback'), ensure_ascii=False)}",
        "",
        "Available actions:",
        "  - ground_video: search memory for relevant segments and inspect them (optionally restrict to search_range).",
        "  - inspect_interval: directly inspect ONE specific time interval (given by search_range).",
        "  - verify_answer: ask the Critic whether current evidence is sufficient.",
        "  - finish: stop and return the current answer.",
        "",
        "Decision rules:",
        "1. If there is no evidence yet, use ground_video over the full video range.",
        "2. If critic_feedback suggests a missing/uncertain time range, prioritize searching that range.",
        "3. Once evidence_count > 0, use verify_answer to let the Critic judge sufficiency; do NOT keep searching.",
        "4. Never search an interval already listed in searched_intervals.",
        "5. Use finish only when evidence is clearly sufficient.",
        "6. search_range must be within [0, duration].",
        "",
        "Return ONLY valid JSON matching this schema:",
        json.dumps({
            "action": "ground_video",
            "query": "what to look for (search intent)",
            "search_range": {"start": 0.0, "end": 300.0},
            "reason": "why this action",
        }, ensure_ascii=False),
    ]
    return "\n".join(lines)


def plan_next_action(
    question: str,
    state: Dict[str, Any],
    video_memory: Dict[str, Any],
) -> Dict[str, Any]:
    """Return the next action dict: {action, query, search_range, reason}."""
    if state.get("temporal_type", NORMAL) != NORMAL:
        return _temporal_plan(question, state, video_memory)

    model = vlm_tool.load_model()
    prompt = _build_planner_prompt(question, state, video_memory)
    raw = vlm_tool.generate_text(prompt, model=model.model, processor=model.processor)
    result = vlm_tool._extract_json(raw)

    action = result.get("action")
    if action not in VALID_ACTIONS:
        action = "verify_answer"

    return {
        "action": action,
        "query": str(result.get("query", question)),
        "search_range": result.get("search_range"),
        "reason": str(result.get("reason", "")),
    }
