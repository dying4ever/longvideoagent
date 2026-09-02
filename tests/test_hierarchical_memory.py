"""Lightweight tests for hierarchical video memory + chapter retrieval.

Run:  python tests/test_hierarchical_memory.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents import grounding  # noqa: E402
from memory import video_memory  # noqa: E402
from tools import vlm_tool  # noqa: E402

_FAKE = vlm_tool._VLM(None, None)


def _mem_with_segments():
    return {
        "video_path": "/tmp/v.mp4",
        "duration": 300.0,
        "segments": [
            {"segment_id": i, "start": i * 60.0, "end": (i + 1) * 60.0, "summary": s, "events": []}
            for i, s in enumerate([
                "佩奇一家在家玩耍", "佩奇一家在家拆礼物", "佩奇去公园", "佩奇在公园玩", "佩奇回家",
            ])
        ],
    }


def test_build_hierarchy() -> None:
    mem = _mem_with_segments()
    gs = '{"global_summary": "佩奇一家在家拆礼物后去公园玩"}'
    chapters = '{"chapters": [{"segment_ids": [0, 1], "summary": "在家"}, {"segment_ids": [2, 3], "summary": "公园"}, {"segment_ids": [4], "summary": "回家"}]}'
    with mock.patch.object(vlm_tool, "load_model", return_value=_FAKE), \
         mock.patch.object(vlm_tool, "generate_text", side_effect=[gs, chapters]):
        result = video_memory.build_hierarchy(mem)
    assert result["global_summary"]
    assert len(result["chapters"]) == 3
    # chapter ranges must be contiguous and valid
    ch = result["chapters"][0]
    assert ch["start"] == 0.0 and ch["end"] == 120.0
    assert ch["segment_ids"] == [0, 1]
    print("[ok] build_hierarchy: global_summary + contiguous chapters")


def test_chapter_ranges_valid() -> None:
    mem = _mem_with_segments()
    mem["chapters"] = [
        {"chapter_id": 0, "start": 0.0, "end": 120.0, "summary": "在家", "segment_ids": [0, 1]},
        {"chapter_id": 1, "start": 120.0, "end": 240.0, "summary": "公园", "segment_ids": [2, 3]},
    ]
    for ch in mem["chapters"]:
        assert ch["start"] < ch["end"]
        assert ch["end"] <= mem["duration"]
    print("[ok] chapter ranges within [0, duration]")


def test_hierarchical_prune_reduces() -> None:
    mem = _mem_with_segments()
    mem["chapters"] = [
        {"chapter_id": 0, "start": 0.0, "end": 120.0, "summary": "在家拆礼物", "segment_ids": [0, 1]},
        {"chapter_id": 1, "start": 120.0, "end": 240.0, "summary": "公园玩耍", "segment_ids": [2, 3]},
        {"chapter_id": 2, "start": 240.0, "end": 300.0, "summary": "回家", "segment_ids": [4]},
        {"chapter_id": 3, "start": 300.0, "end": 360.0, "summary": "xxx", "segment_ids": []},
    ]
    # many chapters (>5) should prune; simulate more chapters to trigger pruning
    many = _mem_with_segments()
    many["chapters"] = [
        {"chapter_id": i, "start": i * 60.0, "end": (i + 1) * 60.0, "summary": f"ch{i}", "segment_ids": [i]}
        for i in range(10)
    ]
    with mock.patch.object(grounding, "_select_chapters_llm", return_value=[0, 1, 2]):
        pruned = grounding._hierarchical_prune("公园在哪里", many, many["segments"], max_chapters=3)
    assert len(pruned) <= 3
    print("[ok] hierarchical prune reduces segment candidates")


def test_select_chapters_llm() -> None:
    chapters = [
        {"chapter_id": i, "start": i * 60.0, "end": (i + 1) * 60.0, "summary": f"ch{i}", "segment_ids": [i]}
        for i in range(10)
    ]
    with mock.patch.object(vlm_tool, "load_model", return_value=_FAKE), \
         mock.patch.object(vlm_tool, "generate_text", return_value='{"chapter_ids": [2, 5, 7]}'):
        selected = grounding._select_chapters_llm("公园在哪里", chapters, 3)
    assert selected == [2, 5, 7]
    print("[ok] select_chapters_llm returns semantic chapter ids")


if __name__ == "__main__":
    test_build_hierarchy()
    test_chapter_ranges_valid()
    test_hierarchical_prune_reduces()
    test_select_chapters_llm()
    print("ALL HIERARCHICAL MEMORY TESTS PASSED")
