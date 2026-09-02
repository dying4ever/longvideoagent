"""Minimal self-contained test for the MVP pipeline (no model weights needed).

Run:  python tests/test_basic_pipeline.py

Part 1 (video_tool): synthesizes a 10s test video with ffmpeg, then checks
duration, frame sampling and timestamp correctness.

Part 2 (vlm_tool, pure-python only): verifies LLM-output JSON parsing and
timestamp injection, so the end-to-end run doesn't fail at the parse stage
after a long model load.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools import video_tool  # noqa: E402
from tools import vlm_tool  # noqa: E402


def _make_test_video(path: str, duration: int = 10, fps: int = 25) -> str:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i",
         f"testsrc=duration={duration}:size=640x360:rate={fps}",
         "-pix_fmt", "yuv420p", path],
        check=True,
    )
    return path


def test_duration() -> None:
    with tempfile.TemporaryDirectory() as d:
        v = _make_test_video(os.path.join(d, "t.mp4"))
        dur = video_tool.get_video_duration(v)
        assert abs(dur - 10.0) < 0.5, f"duration {dur} != ~10"
        print(f"[ok] duration = {dur:.2f}s")


def test_sample_frames_timestamps() -> None:
    with tempfile.TemporaryDirectory() as d:
        v = _make_test_video(os.path.join(d, "t.mp4"))
        frames = video_tool.sample_frames(v, start_time=0, end_time=6, interval=2.0)
        assert len(frames) == 3, f"expected 3 frames, got {len(frames)}"
        ts = [f.timestamp for f in frames]
        for expect in (0.0, 2.0, 4.0):
            assert any(abs(t - expect) < 0.1 for t in ts), f"missing ts ~{expect} in {ts}"
        assert all(f.image.size == (640, 360) for f in frames), \
            f"unexpected frame size {[f.image.size for f in frames]}"
        assert all(f.image.mode == "RGB" for f in frames)
        print(f"[ok] sampled timestamps = {ts}")


def test_errors() -> None:
    assert _raises(video_tool.get_video_duration, "/no/such/video.mp4")
    with tempfile.TemporaryDirectory() as d:
        v = _make_test_video(os.path.join(d, "t.mp4"))
        assert _raises(video_tool.sample_frames, v, start_time=100)
        assert _raises(video_tool.sample_frames, v, end_time=999)
        assert _raises(video_tool.sample_frames, v, interval=0)
    print("[ok] error handling works")


def _raises(fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
    except Exception:
        return True
    return False


def test_vlm_json_extraction() -> None:
    cases = {
        "clean json": '{"summary": "a", "events": [{"timestamp": 1.0, "description": "x"}]}',
        "fenced json": '```json\n{"summary": "a", "events": []}\n```',
        "trailing text": 'Here is the result:\n{"summary": "a", "events": []}\nHope that helps!',
        "chinese json": '{"summary": "男子进入房间", "events": [{"timestamp": 12.0, "description": "男子拿起杯子"}]}',
        "array output": '[{"timestamp": 1.0, "description": "x"}]',
        "prose + fence": 'Sure! ```json\n{"a": 1}\n``` done',
    }
    for name, text in cases.items():
        r = vlm_tool._extract_json(text)
        assert isinstance(r, (dict, list)), f"{name}: unexpected type {type(r)}"
    for bad in ("", "no json here at all"):
        assert _raises(vlm_tool._extract_json, bad), f"should raise for {bad!r}"
    print("[ok] vlm_tool JSON extraction robust")


def test_vlm_prompt_timestamps() -> None:
    @dataclass
    class F:
        timestamp: float
        image: Image.Image

    frames = [F(12.0, Image.new("RGB", (8, 8))), F(14.0, Image.new("RGB", (8, 8)))]
    p = vlm_tool.build_frames_prompt(frames, "发生了什么？")
    assert "timestamp = 12.0 s" in p
    assert "timestamp = 14.0 s" in p
    assert "Frame 1" in p and "Frame 2" in p
    assert "JSON" in p
    print("[ok] vlm_tool prompt injects timestamps + JSON schema")


if __name__ == "__main__":
    test_duration()
    test_sample_frames_timestamps()
    test_errors()
    test_vlm_json_extraction()
    test_vlm_prompt_timestamps()
    print("ALL TESTS PASSED")
