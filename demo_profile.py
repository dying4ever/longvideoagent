"""Profile the LongVideoAgent pipeline: per-stage timing + VLM calls + frames.

Usage:
  python demo_profile.py --video data/videos/test.mp4
"""
from __future__ import annotations

import argparse
import time

from session import LongVideoAgentSession
from tools import vlm_tool
from utils import profiler


def _dump(title: str, dt: float) -> None:
    c = vlm_tool.get_call_counts()
    print(f"\n{title}  ({dt:.1f}s)")
    print(f"  image VLM calls: {c['analyze_frames']}  text LLM calls: {c['generate_text']}  "
          f"frames viewed: {profiler.get_frame_count()}")
    for stage, v in profiler.get_timings().items():
        print(f"  {stage:<20} {v['total']:>7.1f}s  ({v['calls']} calls)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    args = p.parse_args()

    profiler.reset()
    vlm_tool.reset_call_counts()
    t0 = time.perf_counter()
    session = LongVideoAgentSession(args.video)
    _dump("[Memory] session init (load/build)", time.perf_counter() - t0)

    questions = [
        "乔治第一次什么时候出现？",
        "他出现之后做了什么？",
        "他后来有没有再次出现？",
    ]
    for q in questions:
        profiler.reset()
        vlm_tool.reset_call_counts()
        t0 = time.perf_counter()
        result = session.ask(q)
        print(f"\n[Q] {q}")
        print(f"  answer: {result.get('current_answer')} ({result.get('status')})")
        _dump("[Stages]", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
