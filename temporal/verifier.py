"""Temporal condition verifier: decide whether global temporal semantics hold.

Pure Python, no model calls. Works on occurrence timestamps + coverage
intervals to decide FIRST/LAST/REPEAT/ALWAYS/BEFORE/AFTER sufficiency, and
reports the missing ranges that must still be inspected.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import config
from temporal.parser import AFTER, ALWAYS, BEFORE, FIRST, LAST, REPEAT
from utils import intervals


def cluster_occurrences(
    occurrences: Sequence[Dict[str, Any]],
    threshold: float = config.TEMPORAL_MERGE_THRESHOLD,
) -> List[Dict[str, float]]:
    """Group occurrences whose timestamps are within `threshold` into clusters.

    Returns a list of {"start", "end"} where each cluster is one independent
    event, so adjacent frames (78s/80s/82s) collapse into a single cluster.
    """
    if not occurrences:
        return []
    ts = sorted({round(float(o["timestamp"]), 3) for o in occurrences})
    clusters: List[Dict[str, float]] = []
    start = prev = ts[0]
    for t in ts[1:]:
        if t - prev > threshold:
            clusters.append({"start": start, "end": prev})
            start = t
        prev = t
    clusters.append({"start": start, "end": prev})
    return clusters


def _fmt_missing(missing: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[round(m[0], 3), round(m[1], 3)] for m in missing]


def _verify_first(occurrences, verified_absence, duration):
    if not occurrences:
        return {"sufficient": False, "candidate_timestamp": None,
                "missing_ranges": [[0.0, duration]],
                "reason": "尚未发现任何 occurrence，需全范围搜索", "answer": None}
    candidate = min(float(o["timestamp"]) for o in occurrences)
    if candidate <= config.TEMPORAL_MERGE_THRESHOLD:
        return {"sufficient": True, "candidate_timestamp": candidate, "missing_ranges": [],
                "reason": f"候选 {candidate}s 距视频开头不足 merge 阈值，无空间存在更早独立 occurrence", "answer": candidate}
    missing = intervals.subtract_intervals([0.0, candidate], verified_absence)
    if not missing:
        return {"sufficient": True, "candidate_timestamp": candidate, "missing_ranges": [],
                "reason": f"前缀 [0,{candidate}) 已确认不存在更早 occurrence", "answer": candidate}
    return {"sufficient": False, "candidate_timestamp": candidate,
            "missing_ranges": _fmt_missing(missing),
            "reason": f"候选 {candidate}s，但 [0,{candidate}) 尚未排查", "answer": None}


def _verify_last(occurrences, verified_absence, duration):
    if not occurrences:
        return {"sufficient": False, "candidate_timestamp": None,
                "missing_ranges": [[0.0, duration]],
                "reason": "尚未发现任何 occurrence，需全范围搜索", "answer": None}
    candidate = max(float(o["timestamp"]) for o in occurrences)
    if duration - candidate <= config.TEMPORAL_MERGE_THRESHOLD:
        return {"sufficient": True, "candidate_timestamp": candidate, "missing_ranges": [],
                "reason": f"候选 {candidate}s 距视频末尾不足 merge 阈值，无空间存在更晚独立 occurrence", "answer": candidate}
    missing = intervals.subtract_intervals([candidate, duration], verified_absence)
    if not missing:
        return {"sufficient": True, "candidate_timestamp": candidate, "missing_ranges": [],
                "reason": f"后缀 [{candidate},{duration}) 已确认不存在更晚 occurrence", "answer": candidate}
    return {"sufficient": False, "candidate_timestamp": candidate,
            "missing_ranges": _fmt_missing(missing),
            "reason": f"候选 {candidate}s，但 [{candidate},{duration}) 尚未排查", "answer": None}


def _verify_repeat(occurrences, verified_absence, duration, threshold):
    clusters = cluster_occurrences(occurrences, threshold)
    if len(clusters) >= 2:
        return {"sufficient": True, "candidate_timestamp": None, "missing_ranges": [],
                "reason": f"发现 {len(clusters)} 个独立 occurrence cluster，REPEAT 成立", "answer": True}
    if not clusters:
        missing = [[0.0, duration]]
    else:
        missing = intervals.subtract_intervals([clusters[0]["end"], duration], verified_absence)
    return {"sufficient": False, "candidate_timestamp": None,
            "missing_ranges": _fmt_missing(missing),
            "reason": "独立 occurrence 不足 2 个，无法证明 repeat", "answer": None}


def _verify_always(occurrences, verified_absence, searched, duration):
    if verified_absence:
        return {"sufficient": True, "candidate_timestamp": None, "missing_ranges": [],
                "reason": "发现反例：目标在部分区间不存在", "answer": False}
    coverage = intervals.merge_intervals(searched)
    missing = intervals.subtract_intervals([0.0, duration], coverage)
    if missing:
        return {"sufficient": False, "candidate_timestamp": None,
                "missing_ranges": _fmt_missing(missing),
                "reason": "未发现反例，但覆盖不足", "answer": None}
    return {"sufficient": True, "candidate_timestamp": None, "missing_ranges": [],
            "reason": "全范围覆盖且未发现反例", "answer": True}


def _verify_before(occurrences, verified_absence, duration):
    if not occurrences:
        return {"sufficient": False, "candidate_timestamp": None,
                "missing_ranges": [[0.0, duration]],
                "reason": "需先定位 reference event", "answer": None}
    ref = min(float(o["timestamp"]) for o in occurrences)
    missing = intervals.subtract_intervals([0.0, ref], verified_absence)
    if not missing:
        return {"sufficient": True, "candidate_timestamp": ref, "missing_ranges": [],
                "reason": f"reference event ({ref}s) 之前范围已覆盖", "answer": None}
    return {"sufficient": False, "candidate_timestamp": ref,
            "missing_ranges": _fmt_missing(missing),
            "reason": "需覆盖 reference 之前范围", "answer": None}


def _verify_after(occurrences, verified_absence, duration):
    if not occurrences:
        return {"sufficient": False, "candidate_timestamp": None,
                "missing_ranges": [[0.0, duration]],
                "reason": "需先定位 reference event", "answer": None}
    ref = max(float(o["timestamp"]) for o in occurrences)
    missing = intervals.subtract_intervals([ref, duration], verified_absence)
    if not missing:
        return {"sufficient": True, "candidate_timestamp": ref, "missing_ranges": [],
                "reason": f"reference event ({ref}s) 之后范围已覆盖", "answer": None}
    return {"sufficient": False, "candidate_timestamp": ref,
            "missing_ranges": _fmt_missing(missing),
            "reason": "需覆盖 reference 之后范围", "answer": None}


def verify_temporal_condition(
    temporal_type: str,
    occurrences: Sequence[Dict[str, Any]],
    searched_intervals: Sequence[Sequence[float]],
    verified_absence_intervals: Sequence[Sequence[float]],
    video_duration: float,
) -> Dict[str, Any]:
    """Return {sufficient, candidate_timestamp, missing_ranges, reason, answer}."""
    handlers = {
        FIRST: _verify_first,
        LAST: _verify_last,
        BEFORE: _verify_before,
        AFTER: _verify_after,
    }
    if temporal_type in handlers:
        return handlers[temporal_type](occurrences, verified_absence_intervals, video_duration)
    if temporal_type == REPEAT:
        return _verify_repeat(occurrences, verified_absence_intervals, video_duration,
                              config.TEMPORAL_MERGE_THRESHOLD)
    if temporal_type == ALWAYS:
        return _verify_always(occurrences, verified_absence_intervals, searched_intervals, video_duration)
    return {"sufficient": True, "candidate_timestamp": None, "missing_ranges": [],
            "reason": "普通问答，无时序约束", "answer": None}
