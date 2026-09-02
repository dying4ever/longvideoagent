"""Long-video QA: Memory -> Grounding -> Reasoning -> Answer.

Usage:
  python demo_long_video.py --video data/videos/example.mp4 \
      --question "兔子什么时候第一次出现？"
"""
from __future__ import annotations

import argparse

from agents import grounding, reasoning
from memory import video_memory


def main() -> int:
    p = argparse.ArgumentParser(description="LongVideoAgent long-video QA pipeline")
    p.add_argument("--video", required=True, help="path to the input video")
    p.add_argument("--question", required=True, help="question about the video")
    p.add_argument("--window-size", type=float, default=60.0, help="memory window (seconds)")
    p.add_argument("--frame-interval", type=float, default=10.0, help="memory sampling interval")
    p.add_argument("--top-k", type=int, default=3, help="number of grounding candidates")
    p.add_argument("--fine-interval", type=float, default=2.0, help="reasoning sampling interval")
    args = p.parse_args()

    mem_path = video_memory.default_memory_path(args.video)
    existing = video_memory.load_video_memory(mem_path)
    if existing is not None:
        print(f"[Memory] found existing memory ({len(existing.get('segments', []))} segments), loading...")
    else:
        print("[Memory] building coarse video memory...")

    memory = video_memory.build_video_memory(
        args.video,
        window_size=args.window_size,
        frame_interval=args.frame_interval,
    )
    n = len(memory["segments"])
    print(f"[Memory] {n} segments loaded (window={args.window_size}s).")

    g = grounding.ground_video(args.question, memory, top_k=args.top_k)
    print("[Grounding]")
    if not g["candidates"]:
        print("  no candidates found")
    for i, c in enumerate(g["candidates"], 1):
        print(f"  Candidate {i}: {c['start']}s - {c['end']}s (score={c['score']:.2f})")
        print(f"    Reason: {c['reason']}")

    r = reasoning.reason_over_candidates(
        args.video, args.question, g["candidates"], fine_interval=args.fine_interval
    )

    print("[Reasoning]")
    for iv in r["inspected_intervals"]:
        print(f"  Inspecting {iv['start']}s - {iv['end']}s...")

    print(f"\n[Answer]")
    print(f"  {r['answer']} (confidence={r['confidence']})")

    print("\n[Evidence]")
    if not r["evidence"]:
        print("  (none)")
    for ev in r["evidence"]:
        print(f"  {ev['timestamp']}s: {ev['description']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
