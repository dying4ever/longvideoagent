"""Lightweight tests for the Agent loop (Planner/Critic/State/Loop).

Model-dependent calls are monkeypatched; no 17.5GB checkpoint is loaded.

Run:  python tests/test_agent_loop.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import agent  # noqa: E402
from agent_state import AgentState  # noqa: E402
from agents import critic, grounding, planner  # noqa: E402
from tools import vlm_tool  # noqa: E402

_FAKE_MODEL = vlm_tool._VLM(None, None)

_MEM = {
    "duration": 120.0,
    "segments": [
        {"segment_id": 0, "start": 0.0, "end": 60.0, "summary": "森林，没有兔子", "events": []},
        {"segment_id": 1, "start": 60.0, "end": 120.0, "summary": "兔子出现", "events": []},
    ],
}


def _plan(action, range_=None, query="q", reason="r"):
    return {"action": action, "query": query, "search_range": range_, "reason": reason}


# --- Planner ---

def test_planner_valid_action_and_fallback() -> None:
    with mock.patch.object(vlm_tool, "load_model", return_value=_FAKE_MODEL), \
         mock.patch.object(vlm_tool, "generate_text", return_value='{"action": "ground_video", "reason": "r"}'):
        r = planner.plan_next_action("q", {"iteration": 0, "max_iterations": 5}, _MEM)
    assert r["action"] in planner.VALID_ACTIONS

    with mock.patch.object(vlm_tool, "load_model", return_value=_FAKE_MODEL), \
         mock.patch.object(vlm_tool, "generate_text", return_value='{"action": "bogus_action"}'):
        r = planner.plan_next_action("q", {"iteration": 0, "max_iterations": 5}, _MEM)
    assert r["action"] == "verify_answer"  # undefined action falls back
    print("[ok] planner: valid action + fallback for undefined action")


def test_search_range_clamped() -> None:
    assert agent._valid_range({"start": -10, "end": 999}, 120.0) == {"start": 0.0, "end": 120.0}
    assert agent._valid_range({"start": 100, "end": 50}, 120.0) == {"start": 100.0, "end": 120.0}
    assert agent._valid_range(None, 120.0) == {"start": 0.0, "end": 120.0}
    print("[ok] _valid_range clamps into [0, duration]")


# --- Grounding range restriction ---

def test_grounding_range_restriction() -> None:
    with mock.patch.object(vlm_tool, "load_model", return_value=_FAKE_MODEL), \
         mock.patch.object(vlm_tool, "generate_text", return_value='{"candidates": [{"segment_id": 0, "score": 0.9, "reason": "r"}]}'):
        r = grounding.ground_video("兔子什么时候出现", _MEM, top_k=3, search_start=0.0, search_end=60.0)
    assert len(r["candidates"]) == 1
    assert r["candidates"][0]["start"] == 0.0 and r["candidates"][0]["end"] == 60.0
    print("[ok] grounding restricted to [0, 60) returns only segment 0")


# --- Critic ---

def test_critic_fields_complete() -> None:
    raw = ('{"sufficient": false, "reason": "r", "missing_evidence": "m", '
           '"suggested_action": "search_earlier", "suggested_range": {"start": 0, "end": 60}}')
    with mock.patch.object(vlm_tool, "load_model", return_value=_FAKE_MODEL), \
         mock.patch.object(vlm_tool, "generate_text", return_value=raw):
        r = critic.critique_answer("q", "a", [], [], 120.0)
    assert set(r.keys()) == {"sufficient", "reason", "missing_evidence", "suggested_action", "suggested_range"}
    assert r["sufficient"] is False
    print("[ok] critic returns all required fields")


# --- Agent loop ---

def _run_loop(plans, critic_results, reasoning_results, grounding_cands, max_iterations=5):
    with mock.patch.object(agent.planner, "plan_next_action", side_effect=plans), \
         mock.patch.object(agent.grounding, "ground_video", side_effect=grounding_cands), \
         mock.patch.object(agent.reasoning, "reason_over_candidates", side_effect=reasoning_results), \
         mock.patch.object(agent.critic, "critique_answer", side_effect=critic_results), \
         mock.patch.object(vlm_tool, "load_model", return_value=_FAKE_MODEL), \
         mock.patch.object(vlm_tool, "generate_text", return_value='{"answer": "综合答案", "confidence": "high"}'):
        return agent.run_agent("兔子第一次什么时候出现？", "/tmp/fake.mp4", _MEM, max_iterations=max_iterations)


def test_loop_sufficient_finish() -> None:
    plans = [_plan("ground_video"), _plan("verify_answer")]
    cands = [{"query": "q", "candidates": [{"start": 60.0, "end": 120.0, "score": 0.9, "reason": "r"}]}]
    reason_res = [{"answer": "兔子在 72 秒出现", "confidence": "high",
                   "evidence": [{"timestamp": 72.0, "description": "兔子出现"}]}]
    critics = [{"sufficient": True, "reason": "ok", "missing_evidence": "", "suggested_action": "none", "suggested_range": None}]
    r = _run_loop(plans, critics, reason_res, cands)

    assert r["status"] == "finished"
    agents = [e["agent"] for e in r["trace"]]
    assert agents == ["planner", "grounding", "reasoning", "planner", "critic"]
    assert r["searched_intervals"] == [{"start": 60.0, "end": 120.0}]
    print("[ok] loop finishes when critic says sufficient")


def test_loop_insufficient_replan_then_finish() -> None:
    plans = [
        _plan("ground_video"),
        _plan("verify_answer"),
        _plan("inspect_interval", {"start": 0.0, "end": 60.0}),
        _plan("verify_answer"),
    ]
    cands = [{"query": "q", "candidates": [{"start": 60.0, "end": 120.0, "score": 0.9, "reason": "r"}]}]
    reason_res = [
        {"answer": "兔子在 72 秒出现", "confidence": "high",
         "evidence": [{"timestamp": 72.0, "description": "兔子出现"}]},
        {"answer": "0-60 秒没有兔子", "confidence": "high",
         "evidence": [{"timestamp": 30.0, "description": "只有森林"}]},
    ]
    critics = [
        {"sufficient": False, "reason": "无法证明是第一次", "missing_evidence": "需要检查 0-60 秒",
         "suggested_action": "search_earlier", "suggested_range": {"start": 0.0, "end": 60.0}},
        {"sufficient": True, "reason": "已检查 0-120 秒", "missing_evidence": "", "suggested_action": "none", "suggested_range": None},
    ]
    r = _run_loop(plans, critics, reason_res, cands)

    assert r["status"] == "finished"
    assert r["searched_intervals"] == [{"start": 60.0, "end": 120.0}, {"start": 0.0, "end": 60.0}]
    assert len([e for e in r["trace"] if e["agent"] == "critic"]) == 2  # replan happened
    assert r["current_answer"] == "综合答案"
    print("[ok] loop replans after insufficient critic, then finishes")


def test_loop_max_iterations() -> None:
    plans = [_plan("ground_video")] * 10  # never verifies/finishes
    cands = [{"query": "q", "candidates": [{"start": 60.0, "end": 120.0, "score": 0.9, "reason": "r"}]}] * 10
    reason_res = [{"answer": "兔子出现", "confidence": "high",
                   "evidence": [{"timestamp": 72.0, "description": "兔子出现"}]}] * 10
    critics = [{"sufficient": False, "reason": "x", "missing_evidence": "y", "suggested_action": "none", "suggested_range": None}] * 10
    r = _run_loop(plans, critics, reason_res, cands, max_iterations=3)

    assert r["status"] == "max_iterations_reached"
    print("[ok] loop stops at max_iterations")


def test_state_dedupe_and_trace() -> None:
    s = AgentState("q", max_iterations=5)
    s.add_searched_interval(0.0, 60.0)
    s.add_searched_interval(0.0, 60.0)  # duplicate
    s.add_searched_interval(60.0, 120.0)
    assert s.searched_intervals == [{"start": 0.0, "end": 60.0}, {"start": 60.0, "end": 120.0}]
    s.add_trace("planner", action="x")
    s.add_trace("critic", sufficient=False)
    assert [e["agent"] for e in s.trace] == ["planner", "critic"]
    assert [e["step"] for e in s.trace] == [1, 2]
    print("[ok] state dedupes intervals + trace step order correct")


if __name__ == "__main__":
    test_planner_valid_action_and_fallback()
    test_search_range_clamped()
    test_grounding_range_restriction()
    test_critic_fields_complete()
    test_loop_sufficient_finish()
    test_loop_insufficient_replan_then_finish()
    test_loop_max_iterations()
    test_state_dedupe_and_trace()
    print("ALL AGENT-LOOP TESTS PASSED")
