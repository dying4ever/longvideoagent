"""Lightweight tests for Memory / Grounding / Reasoning (no model weights).

All model-dependent calls are monkeypatched so these tests run fast and do not
load the 17.5GB Qwen3-VL checkpoint.

Run:  python tests/test_long_video.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents import grounding, reasoning  # noqa: E402
from memory import video_memory  # noqa: E402
from tools import video_tool, vlm_tool  # noqa: E402


@dataclass
class _FakeFrame:
    timestamp: float
    image: object = None


_FAKE_MODEL = vlm_tool._VLM(None, None)


def _sample_mem(duration=120.0, window_size=60.0):
    return {
        "video_path": "/tmp/fake.mp4",
        "duration": duration,
        "window_size": window_size,
        "frame_interval": 10.0,
        "segments": [
            {"segment_id": 0, "start": 0.0, "end": 60.0, "summary": "森林和草地", "events": []},
            {"segment_id": 1, "start": 60.0, "end": 120.0, "summary": "兔子从洞里出来", "events": []},
        ],
    }


def test_split_segments() -> None:
    segs = video_memory._split_segments(125.0, 60.0)
    assert [(s["start"], s["end"]) for s in segs] == [(0.0, 60.0), (60.0, 120.0), (120.0, 125.0)]
    assert segs[-1]["end"] - segs[-1]["start"] == 5.0  # last segment partial
    assert video_memory._split_segments(60.0, 60.0) == [{"segment_id": 0, "start": 0.0, "end": 60.0}]
    print("[ok] split_segments: windows + partial last segment correct")


def test_memory_save_load() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mem.json")
        mem = _sample_mem()
        video_memory.save_video_memory(mem, path)
        loaded = video_memory.load_video_memory(path)
        assert loaded == mem
        assert video_memory.load_video_memory(os.path.join(d, "nope.json")) is None
    print("[ok] memory save/load round-trip + missing-file returns None")


def test_build_memory_resume() -> None:
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "mem.json")
        partial = _sample_mem()
        partial["segments"] = partial["segments"][:1]  # only segment 0 built
        video_memory.save_video_memory(partial, out)

        calls = []

        def fake_sample(video_path, start_time=None, end_time=None, interval=2.0):
            calls.append((start_time, end_time))
            return [_FakeFrame(start_time)]

        with mock.patch.object(video_tool, "get_video_duration", return_value=120.0), \
             mock.patch.object(vlm_tool, "load_model", return_value=_FAKE_MODEL), \
             mock.patch.object(video_tool, "sample_frames", side_effect=fake_sample), \
             mock.patch.object(vlm_tool, "analyze_frames", return_value={
                 "summary": "s", "events": [{"timestamp": 60.0, "description": "e"}],
             }):
            result = video_memory.build_video_memory("/tmp/fake.mp4", window_size=60.0, output_path=out)

        assert len(calls) == 1 and calls[0] == (60.0, 120.0), f"should only build segment 1, got {calls}"
        assert len(result["segments"]) == 2
    print("[ok] build_memory resumes: already-built segment 0 skipped, only segment 1 built")


def test_coarse_filter() -> None:
    segs = _sample_mem()["segments"]
    top = grounding.coarse_filter("兔子什么时候出现", segs, top_n=1)
    assert len(top) == 1
    assert top[0]["segment_id"] == 1  # segment mentioning 兔子 ranks higher
    print("[ok] coarse_filter picks the segment mentioning 兔子")


def test_grounding_valid_ranges() -> None:
    mem = _sample_mem()
    raw = '{"candidates": [{"segment_id": 1, "score": 0.95, "reason": "兔子出现"}]}'
    with mock.patch.object(vlm_tool, "load_model", return_value=_FAKE_MODEL), \
         mock.patch.object(vlm_tool, "generate_text", return_value=raw):
        result = grounding.ground_video("兔子什么时候出现？", mem, top_k=2)

    assert result["query"] == "兔子什么时候出现？"
    assert len(result["candidates"]) == 1
    c = result["candidates"][0]
    assert c["start"] == 60.0 and c["end"] == 120.0
    assert 0.0 <= c["score"] <= 1.0
    assert c["reason"]
    print("[ok] grounding maps segment_id -> valid [start, end) ranges")


def test_reasoning_bounds_and_evidence() -> None:
    calls = []

    def fake_sample(video_path, start_time=None, end_time=None, interval=2.0):
        calls.append((start_time, end_time, interval))
        return [_FakeFrame(start_time + i * interval) for i in range(3)]

    def fake_analyze(frames, prompt, model=None, processor=None, max_new_tokens=1024):
        return {
            "answer": "兔子约在 12 秒出现",
            "confidence": "high",
            "evidence": [{"timestamp": 12.0, "description": "兔子从洞口出现"}],
        }

    with mock.patch.object(vlm_tool, "load_model", return_value=_FAKE_MODEL), \
         mock.patch.object(video_tool, "sample_frames", side_effect=fake_sample), \
         mock.patch.object(vlm_tool, "analyze_frames", side_effect=fake_analyze):
        result = reasoning.reason_over_candidates(
            "/tmp/fake.mp4", "兔子什么时候出现？",
            [{"start": 10.0, "end": 20.0}], fine_interval=2.0,
        )

    assert len(calls) == 1
    assert calls[0] == (10.0, 20.0, 2.0)  # frames read ONLY within candidate interval
    assert result["inspected_intervals"] == [{"start": 10.0, "end": 20.0}]
    assert result["answer"] == "兔子约在 12 秒出现"
    assert result["evidence"][0]["timestamp"] == 12.0
    print("[ok] reasoning samples only within candidate interval + returns evidence")


if __name__ == "__main__":
    test_split_segments()
    test_memory_save_load()
    test_build_memory_resume()
    test_coarse_filter()
    test_grounding_valid_ranges()
    test_reasoning_bounds_and_evidence()
    print("ALL LONG-VIDEO TESTS PASSED")
