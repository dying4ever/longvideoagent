"""Minimal interval utilities for temporal coverage reasoning.

All intervals are [start, end) in seconds. A small epsilon absorbs floating
point boundary error so that [0, 30] + [30, 60] merges into [0, 60].
"""
from __future__ import annotations

from typing import List, Sequence

EPS = 1e-6


def _pairs(intervals: Sequence[Sequence[float]]) -> List[List[float]]:
    result: List[List[float]] = []
    for iv in intervals:
        if isinstance(iv, dict):
            result.append([float(iv["start"]), float(iv["end"])])
        else:
            result.append([float(iv[0]), float(iv[1])])
    return result


def merge_intervals(intervals: Sequence[Sequence[float]]) -> List[List[float]]:
    """Merge overlapping/adjacent intervals into disjoint, sorted intervals."""
    if not intervals:
        return []
    sorted_ivs = sorted(_pairs(intervals), key=lambda x: (x[0], x[1]))
    merged: List[List[float]] = [sorted_ivs[0][:]]
    for s, e in sorted_ivs[1:]:
        if s <= merged[-1][1] + EPS:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def subtract_intervals(
    total: Sequence[float],
    subtracted: Sequence[Sequence[float]],
) -> List[List[float]]:
    """Return the sub-intervals of `total` not covered by `subtracted`."""
    total = [float(total[0]), float(total[1])]
    result: List[List[float]] = []
    cur = total[0]
    for s, e in merge_intervals(subtracted):
        if e <= total[0] or s >= total[1]:
            continue
        s = max(s, total[0])
        e = min(e, total[1])
        if s > cur + EPS:
            result.append([cur, s])
        cur = max(cur, e)
    if cur < total[1] - EPS:
        result.append([cur, total[1]])
    return result


def is_range_covered(
    target: Sequence[float],
    covered: Sequence[Sequence[float]],
) -> bool:
    """Return True if `target` is fully covered by `covered` intervals."""
    return len(subtract_intervals(target, covered)) == 0
