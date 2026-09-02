"""Minimal end-to-end demo: local video -> timestamped frames -> Qwen3-VL -> JSON.

Usage:
  python demo_basic.py --video data/videos/test_10s.mp4 --start 0 --end 30 \\
      --interval 2 --question "这段视频发生了什么？"
"""
from __future__ import annotations

import argparse
import json
import sys

import config
from tools import video_tool


def main() -> int:
    p = argparse.ArgumentParser(description="LongVideoAgent minimal pipeline")
    p.add_argument("--video", required=True, help="path to the input video")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=None)
    p.add_argument("--interval", type=float, default=config.DEFAULT_INTERVAL)
    p.add_argument("--question", default="这段视频发生了什么？")
    p.add_argument("--model", default=None, help="model dir override")
    args = p.parse_args()

    # 1. sample timestamped frames
    try:
        frames = video_tool.sample_frames(
            args.video, start_time=args.start, end_time=args.end,
            interval=args.interval,
        )
    except video_tool.VideoError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1
    if not frames:
        print("[error] no frames sampled", file=sys.stderr)
        return 1
    print(f"[info] sampled {len(frames)} frames "
          f"({frames[0].timestamp}s .. {frames[-1].timestamp}s)")

    # 2. load model once (cached across calls)
    from tools import vlm_tool
    try:
        vlm_tool.load_model(model_path=args.model)
    except vlm_tool.VLMError as e:
        print(f"[error] model load failed: {e}", file=sys.stderr)
        return 1

    # 3. analyze
    prompt = vlm_tool.build_frames_prompt(frames, args.question)
    try:
        result = vlm_tool.analyze_frames(frames, prompt)
    except vlm_tool.VLMError as e:
        print(f"[error] analysis failed: {e}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
