"""Lightweight stage profiler for the LongVideoAgent pipeline.

Accumulates per-stage wall-clock time and call counts so a full run can be
broken down into Model Load / Frame Sampling / Event Segmentation / Memory
Build / Grounding / Reasoning / Critic etc.
"""
from __future__ import annotations

import time
from typing import Dict, List

_timings: Dict[str, List[float]] = {}
_frame_count = 0


def count_frames(n: int) -> None:
    global _frame_count
    _frame_count += n


def get_frame_count() -> int:
    return _frame_count


def record(stage: str, duration: float) -> None:
    t = _timings.setdefault(stage, [0.0, 0.0])
    t[0] += duration
    t[1] += 1


class timeit:
    def __init__(self, stage: str):
        self.stage = stage

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        record(self.stage, time.perf_counter() - self._t0)


def get_timings() -> Dict[str, Dict[str, float]]:
    return {
        k: {"total": round(v[0], 3), "calls": int(v[1])}
        for k, v in sorted(_timings.items())
    }


def reset() -> None:
    global _frame_count
    _timings.clear()
    _frame_count = 0


def timed(stage: str):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            with timeit(stage):
                return fn(*args, **kwargs)
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper
    return decorator
