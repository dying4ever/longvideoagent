"""Closed-loop long-video QA demo: Planner -> Grounding -> Reasoning -> Critic -> Answer.

Usage:
  python demo_agent.py --video data/videos/test.mp4 --question "兔子第一次什么时候出现？"
"""
from __future__ import annotations

import argparse
import json

import agent
from memory import video_memory


def main() -> int:
    p = argparse.ArgumentParser(description="LongVideoAgent closed-loop QA")
    p.add_argument("--video", required=True, help="path to the input video")
    p.add_argument("--question", required=True, help="question about the video")
    p.add_argument("--window-size", type=float, default=10.0)
    p.add_argument("--frame-interval", type=float, default=2.0)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--fine-interval", type=float, default=2.0)
    p.add_argument("--max-iterations", type=int, default=5)
    args = p.parse_args()

    mem_path = video_memory.default_memory_path(args.video)
    existing = video_memory.load_video_memory(mem_path)
    if existing is not None:
        print(f"[Memory] found existing memory ({len(existing.get('segments', []))} segments), loading...")
    else:
        print("[Memory] building coarse video memory...")
    memory = video_memory.build_video_memory(
        args.video, window_size=args.window_size, frame_interval=args.frame_interval,
    )
    print(f"[Memory] {len(memory['segments'])} segments (window={args.window_size}s).\n")

    result = agent.run_agent(
        args.question, args.video, memory,
        max_iterations=args.max_iterations, top_k=args.top_k,
        fine_interval=args.fine_interval,
    )

    print("=" * 60)
    for entry in result["trace"]:
        agent_name = entry["agent"]
        if agent_name == "planner":
            print(f"[Planner] step {entry['step']}")
            print(f"  Action: {entry['action']}")
            print(f"  Reason: {entry['reason']}")
            if entry.get("search_range"):
                print(f"  Range: {entry['search_range']}")
        elif agent_name == "grounding":
            print(f"[Grounding] step {entry['step']}")
            for c in entry.get("candidates", []):
                print(f"  Candidate: {c['start']}s - {c['end']}s (score={c['score']:.2f})")
        elif agent_name == "reasoning":
            print(f"[Reasoning] step {entry['step']}")
            print(f"  Inspecting {entry['interval'][0]}s - {entry['interval'][1]}s")
            print(f"  Answer: {entry['answer']}")
        elif agent_name == "critic":
            print(f"[Critic] step {entry['step']}")
            print(f"  Sufficient: {entry['sufficient']}")
            print(f"  Reason: {entry['reason']}")
        print()

    print("=" * 60)
    print(f"[Final Answer] ({result['status']})")
    print(f"  {result['current_answer']}")
    print("\n[Evidence]")
    for ev in result["evidence"]:
        print(f"  {ev['timestamp']}s: {ev['description']}")
    print("\n[Searched Intervals]")
    for iv in result["searched_intervals"]:
        print(f"  {iv['start']}s - {iv['end']}s")

    print("\n[Full Trace (JSON)]")
    print(json.dumps(result["trace"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
