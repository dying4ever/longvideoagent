"""Agent loop: Planner -> Grounding -> Reasoning -> Temporal Verifier -> Critic -> Answer.

For temporal questions (first/last/repeat/always/...), the loop drives
occurrence collection + temporal coverage verification with pure-Python rules,
so it terminates on verifiable coverage rather than max_iterations.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from agent_state import AgentState
from agents import critic, grounding, planner, reasoning
from temporal import parser as temporal_parser
from temporal import verifier as temporal_verifier
from temporal.parser import FIRST, LAST, NORMAL, REPEAT, ALWAYS
from tools import vlm_tool


def _valid_range(sr: Any, duration: float) -> Dict[str, float]:
    if not isinstance(sr, dict):
        return {"start": 0.0, "end": duration}
    try:
        start = float(sr.get("start", 0.0))
        end = float(sr.get("end", duration))
    except (TypeError, ValueError):
        return {"start": 0.0, "end": duration}
    start = max(0.0, min(start, duration))
    end = max(0.0, min(end, duration))
    if end <= start:
        end = duration
    return {"start": start, "end": end}


def _is_searched(start: float, end: float, searched: List[Dict[str, float]]) -> bool:
    iv = {"start": round(float(start), 3), "end": round(float(end), 3)}
    return iv in searched


def _update_occurrences(state: AgentState, result: Dict[str, Any], start: float, end: float) -> None:
    state.add_searched_interval(start, end)
    occurrences = result.get("occurrences", [])
    state.add_occurrences(occurrences)
    if not occurrences:
        state.add_verified_absence(start, end)
    state.add_evidence(result.get("evidence", []))
    state.reasoning_history.append({"interval": [start, end], "result": result})
    state.current_answer = result.get("answer")


def _trace_verifier(state: AgentState, duration: float) -> Dict[str, Any]:
    ver = temporal_verifier.verify_temporal_condition(
        state.temporal_type, state.event_occurrences,
        state.searched_intervals, state.verified_absence_intervals, duration,
    )
    state.add_trace(
        "temporal_verifier", sufficient=ver.get("sufficient"),
        candidate_timestamp=ver.get("candidate_timestamp"),
        missing_ranges=ver.get("missing_ranges"),
        reason=ver.get("reason"),
    )
    return ver


def _temporal_answer(ttype: str, ver: Dict[str, Any]):
    if ttype == FIRST and ver.get("candidate_timestamp") is not None:
        return f"首次出现时间约在 {ver['candidate_timestamp']}s"
    if ttype == LAST and ver.get("candidate_timestamp") is not None:
        return f"最后一次出现约在 {ver['candidate_timestamp']}s"
    if ttype == REPEAT:
        return "是（存在多次独立出现）" if ver.get("answer") else "否（未发现重复出现）"
    if ttype == ALWAYS:
        return "否（存在反例）" if ver.get("answer") is False else "是（未发现反例）"
    return None


def _run_critic(state: AgentState, question: str, duration: float) -> None:
    answer = state.current_answer
    if state.temporal_type != NORMAL:
        ver = temporal_verifier.verify_temporal_condition(
            state.temporal_type, state.event_occurrences,
            state.searched_intervals, state.verified_absence_intervals, duration,
        )
        tmp = _temporal_answer(state.temporal_type, ver)
        if tmp is not None:
            answer = tmp
    c = critic.critique_answer(
        question, answer, state.evidence,
        state.searched_intervals, duration,
        temporal_type=state.temporal_type,
        occurrences=state.event_occurrences,
        verified_absence=state.verified_absence_intervals,
    )
    state.critic_feedback = c
    state.add_trace(
        "critic", sufficient=c.get("sufficient"),
        reason=c.get("reason"),
        suggested_range=c.get("suggested_range"),
    )


def _build_synthesis_prompt(
    question: str,
    reasoning_history: List[Dict[str, Any]],
    evidence: List[Dict[str, Any]],
) -> str:
    lines = [
        "Synthesize a final answer to the question from all observations below.",
        f"Question: {question}",
        "",
        "Observations (per inspected interval):",
    ]
    for h in reasoning_history:
        iv = h.get("interval", [0.0, 0.0])
        res = h.get("result", {})
        lines.append(f"  [{iv[0]}s - {iv[1]}s] {res.get('answer', '')}")
    lines += ["", "Key evidence (timestamp -> description):"]
    for ev in evidence:
        lines.append(f"  {ev['timestamp']}s: {ev['description']}")
    lines += [
        "",
        "Return ONLY valid JSON matching this schema:",
        json.dumps({"answer": "...", "confidence": "high|medium|low"}, ensure_ascii=False),
    ]
    return "\n".join(lines)


def _build_final_answer(state: AgentState, question: str) -> None:
    ver = temporal_verifier.verify_temporal_condition(
        state.temporal_type, state.event_occurrences,
        state.searched_intervals, state.verified_absence_intervals,
        state.video_duration or 0.0,
    )
    tmp = _temporal_answer(state.temporal_type, ver)
    if tmp is not None:
        state.current_answer = tmp
        if ver.get("candidate_timestamp") is not None:
            state.final_timestamp = ver["candidate_timestamp"]
        return

    if len(state.reasoning_history) == 1:
        state.current_answer = state.reasoning_history[0].get("result", {}).get("answer", "无法回答")
    elif state.reasoning_history:
        model = vlm_tool.load_model()
        prompt = _build_synthesis_prompt(question, state.reasoning_history, state.evidence)
        raw = vlm_tool.generate_text(prompt, model=model.model, processor=model.processor)
        synth = vlm_tool._extract_json(raw)
        state.current_answer = synth.get("answer", state.current_answer or "无法回答")


def run_agent(
    question: str,
    video_path: str,
    video_memory: Dict[str, Any],
    max_iterations: int = 5,
    top_k: int = 3,
    fine_interval: float = 2.0,
) -> Dict[str, Any]:
    """Run the closed-loop QA agent and return the final serialized state."""
    duration = float(video_memory.get("duration", 0.0))
    state = AgentState(question, max_iterations=max_iterations, video_duration=duration)

    parsed = temporal_parser.parse_temporal_query(question)
    state.temporal_type = parsed["type"]
    state.add_trace("temporal_parser", type=parsed["type"], target=parsed.get("target"))

    while state.iteration < state.max_iterations:
        state.iteration += 1

        plan = planner.plan_next_action(question, state.to_dict(), video_memory)
        action = plan.get("action")
        if action not in planner.VALID_ACTIONS:
            action = "verify_answer"
        state.add_trace(
            "planner", action=action, reason=plan.get("reason", ""),
            search_range=plan.get("search_range"),
        )

        if action == "ground_video":
            sr = _valid_range(plan.get("search_range"), duration)
            g_result = grounding.ground_video(
                question, video_memory, top_k=top_k,
                search_start=sr["start"], search_end=sr["end"],
            )
            candidates = g_result.get("candidates", [])
            new_cands = [
                c for c in candidates
                if not _is_searched(c["start"], c["end"], state.searched_intervals)
            ]
            state.grounding_history.append({"iteration": state.iteration, "candidates": candidates})
            state.add_trace("grounding", candidates=candidates)
            if not new_cands:
                _run_critic(state, question, duration)
                if state.critic_feedback.get("sufficient"):
                    state.status = "finished"
                    break
                continue
            for cand in new_cands:
                result = reasoning.reason_over_candidates(
                    video_path, question, [cand], fine_interval=fine_interval
                )
                _update_occurrences(state, result, cand["start"], cand["end"])
                state.add_trace(
                    "reasoning", interval=[cand["start"], cand["end"]],
                    answer=result.get("answer"),
                    occurrences=result.get("occurrences"),
                )
            _trace_verifier(state, duration)

        elif action == "inspect_interval":
            sr = _valid_range(plan.get("search_range"), duration)
            if _is_searched(sr["start"], sr["end"], state.searched_intervals):
                _run_critic(state, question, duration)
                if state.critic_feedback.get("sufficient"):
                    state.status = "finished"
                    break
                continue
            result = reasoning.reason_over_candidates(
                video_path, question, [{"start": sr["start"], "end": sr["end"]}],
                fine_interval=fine_interval,
            )
            _update_occurrences(state, result, sr["start"], sr["end"])
            state.add_trace(
                "reasoning", interval=[sr["start"], sr["end"]],
                answer=result.get("answer"),
                occurrences=result.get("occurrences"),
            )
            _trace_verifier(state, duration)

        elif action == "verify_answer":
            _run_critic(state, question, duration)
            if state.critic_feedback.get("sufficient"):
                state.status = "finished"
                break

        elif action == "finish":
            state.status = "finished"
            break
    else:
        state.status = "max_iterations_reached"

    _build_final_answer(state, question)
    return state.to_dict()
