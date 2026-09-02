"""Lightweight tests for temporal reasoning (no model weights).

Covers query classification, occurrence clustering/dedup, interval utilities,
FIRST/LAST/REPEAT/ALWAYS verification rules, and planner missing_range usage.

Run:  python tests/test_temporal_reasoning.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent_state import AgentState  # noqa: E402
from agents import planner  # noqa: E402
from temporal import parser as tparser  # noqa: E402
from temporal import verifier as tverifier  # noqa: E402
from utils import intervals  # noqa: E402
from tools import vlm_tool  # noqa: E402

_FAKE = vlm_tool._VLM(None, None)


# --- classification ---

def test_classify_first_last_always() -> None:
    assert tparser.parse_temporal_query("乔治第一次什么时候出现？")["type"] == "FIRST"
    assert tparser.parse_temporal_query("乔治最后一次什么时候出现？")["type"] == "LAST"
    assert tparser.parse_temporal_query("乔治一直在草地上吗？")["type"] == "ALWAYS"
    print("[ok] classify FIRST/LAST/ALWAYS")


def test_classify_before_after() -> None:
    assert tparser.parse_temporal_query("乔治出现之前发生了什么？")["type"] == "BEFORE"
    assert tparser.parse_temporal_query("乔治出现之后发生了什么？")["type"] == "AFTER"
    print("[ok] classify BEFORE/AFTER")


def test_classify_normal() -> None:
    with mock.patch.object(vlm_tool, "load_model", return_value=_FAKE), \
         mock.patch.object(vlm_tool, "generate_text", return_value='{"type": "NORMAL"}'):
        assert tparser.parse_temporal_query("这段视频里有什么？")["type"] == "NORMAL"
    print("[ok] classify NORMAL (LLM fallback)")


# --- occurrence / interval utils ---

def test_occurrence_dedup() -> None:
    s = AgentState("q")
    s.add_occurrences([
        {"timestamp": 6.0, "description": "乔治出现", "confidence": "high"},
        {"timestamp": 6.0, "description": "乔治出现", "confidence": "high"},
        {"timestamp": 78.0, "description": "乔治出现", "confidence": "high"},
    ])
    assert len(s.event_occurrences) == 2
    print("[ok] occurrence dedup by timestamp")


def test_cluster_merge() -> None:
    occs = [{"timestamp": 78.0}, {"timestamp": 80.0}, {"timestamp": 82.0}, {"timestamp": 168.0}]
    clusters = tverifier.cluster_occurrences(occs, threshold=5.0)
    assert len(clusters) == 2  # [78..82] merged + [168]
    print("[ok] cluster_occurrences merges adjacent frames")


def test_interval_merge() -> None:
    assert intervals.merge_intervals([[0, 30], [30, 60], [60, 78]]) == [[0, 78]]
    print("[ok] merge_intervals")


def test_interval_coverage() -> None:
    assert intervals.is_range_covered([0, 78], [[0, 30], [30, 60], [60, 78]]) is True
    assert intervals.is_range_covered([0, 100], [[0, 78]]) is False
    print("[ok] is_range_covered")


# --- verification rules ---

def test_first_prefix_uncovered() -> None:
    v = tverifier.verify_temporal_condition("FIRST", [{"timestamp": 78.0}], [], [], 285.0)
    assert v["sufficient"] is False
    assert v["candidate_timestamp"] == 78.0
    assert v["missing_ranges"] == [[0.0, 78.0]]
    print("[ok] FIRST prefix uncovered -> insufficient")


def test_first_prefix_covered() -> None:
    v = tverifier.verify_temporal_condition("FIRST", [{"timestamp": 6.0}], [], [[0.0, 6.0]], 285.0)
    assert v["sufficient"] is True
    assert v["candidate_timestamp"] == 6.0
    print("[ok] FIRST prefix covered -> sufficient")


def test_last_suffix_uncovered() -> None:
    v = tverifier.verify_temporal_condition("LAST", [{"timestamp": 168.0}], [], [], 285.0)
    assert v["sufficient"] is False
    assert v["missing_ranges"] == [[168.0, 285.0]]
    print("[ok] LAST suffix uncovered -> insufficient")


def test_last_suffix_covered() -> None:
    v = tverifier.verify_temporal_condition("LAST", [{"timestamp": 168.0}], [], [[168.0, 285.0]], 285.0)
    assert v["sufficient"] is True
    print("[ok] LAST suffix covered -> sufficient")


def test_repeat_single_insufficient() -> None:
    v = tverifier.verify_temporal_condition("REPEAT", [{"timestamp": 6.0}], [], [], 285.0)
    assert v["sufficient"] is False
    print("[ok] REPEAT single occurrence -> insufficient")


def test_repeat_multiple_sufficient() -> None:
    v = tverifier.verify_temporal_condition("REPEAT", [{"timestamp": 6.0}, {"timestamp": 78.0}], [], [], 285.0)
    assert v["sufficient"] is True
    assert v["answer"] is True
    print("[ok] REPEAT multiple occurrences -> sufficient")


def test_always_counterexample() -> None:
    v = tverifier.verify_temporal_condition("ALWAYS", [{"timestamp": 6.0}], [], [[10.0, 30.0]], 285.0)
    assert v["sufficient"] is True
    assert v["answer"] is False
    print("[ok] ALWAYS counterexample -> false + sufficient")


def test_repeat_cluster_merge() -> None:
    v = tverifier.verify_temporal_condition(
        "REPEAT", [{"timestamp": 6.0}, {"timestamp": 8.0}, {"timestamp": 10.0}], [], [], 285.0)
    assert v["sufficient"] is False
    assert len(v.get("clusters", [])) == 1
    print("[ok] REPEAT: consecutive occurrences merge into 1 cluster")


def test_repeat_distinct_clusters_timestamps() -> None:
    v = tverifier.verify_temporal_condition(
        "REPEAT", [{"timestamp": 6.0}, {"timestamp": 78.0}, {"timestamp": 177.0}], [], [], 285.0)
    assert v["sufficient"] is True
    assert v["answer"] is True
    assert len(v["clusters"]) == 3
    assert v["clusters"][1]["start"] == 78.0
    print("[ok] REPEAT: distinct clusters -> sufficient + repeat timestamps")


def test_always_predicate_violation() -> None:
    v = tverifier.verify_temporal_condition(
        "ALWAYS", [{"timestamp": 6.0}], [], [], 285.0,
        violations=[{"timestamp": 30.0, "description": "乔治在屋里不在草地上"}])
    assert v["sufficient"] is True
    assert v["answer"] is False
    print("[ok] ALWAYS: predicate violation -> false + sufficient")


# --- planner integration ---

def _mem(duration=285.0):
    return {"duration": duration, "segments": []}


def test_planner_uses_missing_range() -> None:
    state = {
        "temporal_type": "FIRST",
        "event_occurrences": [{"timestamp": 78.0, "description": "乔治", "confidence": "high"}],
        "searched_intervals": [{"start": 60.0, "end": 120.0}],
        "verified_absence_intervals": [],
        "video_duration": 285.0,
    }
    plan = planner.plan_next_action("乔治第一次什么时候出现？", state, _mem())
    assert plan["action"] == "inspect_interval"
    assert plan["search_range"] == {"start": 0.0, "end": 78.0}
    print("[ok] planner uses missing_range -> inspect_interval")


def test_planner_no_grounding_when_sufficient() -> None:
    state = {
        "temporal_type": "FIRST",
        "event_occurrences": [{"timestamp": 6.0, "description": "乔治", "confidence": "high"}],
        "searched_intervals": [{"start": 0.0, "end": 60.0}],
        "verified_absence_intervals": [{"start": 0.0, "end": 6.0}],
        "video_duration": 285.0,
    }
    plan = planner.plan_next_action("乔治第一次什么时候出现？", state, _mem())
    assert plan["action"] == "verify_answer"
    print("[ok] verifier sufficient -> planner returns verify_answer (no more grounding)")


def test_planner_large_range_uses_grounding() -> None:
    state = {
        "temporal_type": "REPEAT",
        "event_occurrences": [{"timestamp": 6.0, "description": "乔治", "confidence": "high"}],
        "searched_intervals": [],
        "verified_absence_intervals": [],
        "video_duration": 285.0,
    }
    plan = planner.plan_next_action("乔治后来有没有再次出现？", state, _mem())
    assert plan["action"] == "ground_video"
    assert plan["search_range"] == {"start": 6.0, "end": 285.0}
    print("[ok] large missing range -> ground_video (avoid OOM)")


if __name__ == "__main__":
    test_classify_first_last_always()
    test_classify_before_after()
    test_classify_normal()
    test_occurrence_dedup()
    test_cluster_merge()
    test_interval_merge()
    test_interval_coverage()
    test_first_prefix_uncovered()
    test_first_prefix_covered()
    test_last_suffix_uncovered()
    test_last_suffix_covered()
    test_repeat_single_insufficient()
    test_repeat_multiple_sufficient()
    test_repeat_cluster_merge()
    test_repeat_distinct_clusters_timestamps()
    test_always_counterexample()
    test_always_predicate_violation()
    test_planner_uses_missing_range()
    test_planner_no_grounding_when_sufficient()
    test_planner_large_range_uses_grounding()
    print("ALL TEMPORAL REASONING TESTS PASSED")
