"""Interactive multi-turn chat demo for LongVideoAgent.

Usage:
  python demo_chat.py --video data/videos/test.mp4

Commands: /memory (print memory), /trace (last turn trace), /reset, /quit.
"""
from __future__ import annotations

import argparse
import json

from session import LongVideoAgentSession


def _print_memory(session: LongVideoAgentSession) -> None:
    cm = session.conversation
    print("\n=== Conversation Memory ===")
    print(f"turns: {len(cm.turns)}")
    print(f"entities: {list(cm.entities.keys())}")
    print(f"confirmed_events: {len(cm.confirmed_events)}")
    for e in cm.confirmed_events:
        print(f"  - {e.get('subject')}{e.get('predicate')} @ {e.get('timestamp')}s (turn {e.get('source_turn')})")
    print("working_memory:")
    wm = cm.working_memory
    print(f"  current_subject: {wm.get('current_subject')}")
    print(f"  reference_event: {wm.get('reference_event')}")
    print(f"  active_entities: {wm.get('active_entities')}")
    print()


def _print_trace(session: LongVideoAgentSession) -> None:
    if not session.conversation.turns:
        print("(no turns yet)")
        return
    last = session.conversation.turns[-1]
    print(f"\n=== Turn {last['turn_id']} Trace ===")
    print(f"Q: {last['question']}")
    print(f"resolved: {last['resolved_question']}")
    print(f"answer: {last['answer']}")
    print(f"temporal_type: {last['temporal_type']}, timestamp: {last['timestamp']}")
    for entry in last.get("trace", []):
        print(f"  [{entry.get('agent')}] {json.dumps({k: v for k, v in entry.items() if k != 'agent'}, ensure_ascii=False)[:120]}")
    print()


def main() -> int:
    p = argparse.ArgumentParser(description="LongVideoAgent interactive chat")
    p.add_argument("--video", required=True)
    p.add_argument("--window-size", type=float, default=60.0)
    p.add_argument("--frame-interval", type=float, default=10.0)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--fine-interval", type=float, default=3.0)
    p.add_argument("--max-iterations", type=int, default=6)
    args = p.parse_args()

    print("Loading video memory...")
    session = LongVideoAgentSession(
        args.video, window_size=args.window_size, frame_interval=args.frame_interval,
        top_k=args.top_k, fine_interval=args.fine_interval, max_iterations=args.max_iterations,
    )
    print(f"Video memory ready ({len(session.video_memory.get('segments', []))} segments).")
    print("\nLongVideoAgent Chat. Commands: /memory /trace /reset /quit\n")

    while True:
        try:
            q = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q == "/quit":
            break
        if q == "/memory":
            _print_memory(session)
            continue
        if q == "/trace":
            _print_trace(session)
            continue
        if q == "/reset":
            session.reset()
            print("(conversation memory cleared)")
            continue

        result = session.ask(q)
        print(f"Agent: {result.get('current_answer')}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
