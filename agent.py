"""Agent loop: Planner -> Grounding -> Reasoning -> Critic -> (Replan) -> Answer.

Runs a closed loop over a question, accumulating evidence across iterations,
using the Critic to decide when the evidence is sufficient, and the Planner to
re-scope the search when it is not.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from agent_state import AgentState
from agents import critic, grounding, planner, reasoning
from tools import vlm_tool


def _valid_range(sr: Any, duration: float) -> Dict[str, float]:
    """Clamp a search range into [0, duration]."""
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


def _run_critic(state: AgentState, question: str, duration: float) -> None:
    c = critic.critique_answer(
        question, state.current_answer, state.evidence,
        state.searched_intervals, duration,
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
        "Instructions:",
        "1. Use the evidence timestamps to answer temporal questions.",
        "2. For 'first / earliest / first time' questions, report the EARLIEST timestamp where the subject clearly appears.",
        "3. If different observations claim different 'first' times, trust the earliest timestamp.",
        "4. Do NOT invent a time that is not in the evidence.",
        "5. Return ONLY valid JSON matching this schema:",
        json.dumps({"answer": "...", "confidence": "high|medium|low"}, ensure_ascii=False),
    ]
    return "\n".join(lines)


def _final_synthesis(
    question: str,
    reasoning_history: List[Dict[str, Any]],
    evidence: List[Dict[str, Any]],
    current_answer: Any,
) -> str:
    if not reasoning_history:
        return current_answer or "无法回答（没有获得任何有效画面）"
    if len(reasoning_history) == 1:
        return reasoning_history[0].get("result", {}).get("answer", current_answer or "")

    model = vlm_tool.load_model()
    prompt = _build_synthesis_prompt(question, reasoning_history, evidence)
    raw = vlm_tool.generate_text(prompt, model=model.model, processor=model.processor)
    synth = vlm_tool._extract_json(raw)
    return synth.get("answer", current_answer or "")


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
                state.add_searched_interval(cand["start"], cand["end"])
                state.add_evidence(result.get("evidence", []))
                state.reasoning_history.append(
                    {"interval": [cand["start"], cand["end"]], "result": result}
                )
                state.current_answer = result.get("answer")
                state.add_trace(
                    "reasoning", interval=[cand["start"], cand["end"]],
                    answer=result.get("answer"),
                    evidence_count=len(result.get("evidence", [])),
                )

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
            state.add_searched_interval(sr["start"], sr["end"])
            state.add_evidence(result.get("evidence", []))
            state.reasoning_history.append(
                {"interval": [sr["start"], sr["end"]], "result": result}
            )
            state.current_answer = result.get("answer")
            state.add_trace(
                "reasoning", interval=[sr["start"], sr["end"]],
                answer=result.get("answer"),
                evidence_count=len(result.get("evidence", [])),
            )

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

    if state.reasoning_history:
        state.current_answer = _final_synthesis(
            question, state.reasoning_history, state.evidence, state.current_answer
        )

    return state.to_dict()
