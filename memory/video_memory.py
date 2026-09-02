"""Video Memory: coarse-grained long-video memory construction.

Splits a long video into fixed-size time windows, samples a few frames per
window, describes each window with Qwen3-VL, and persists the result as a
structured JSON file.

The memory is resumable: already-built segments are skipped on the next run,
so a crash mid-build does not restart from zero.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional

import config
from tools import video_tool, vlm_tool

MEMORY_QUESTION = "这段视频里发生了什么？请描述画面中的场景、人物、物体和动作。"


def _split_segments(duration: float, window_size: float) -> List[Dict[str, float]]:
    """Split [0, duration) into consecutive windows of `window_size` seconds.

    The final window is shorter than window_size when duration is not an exact
    multiple. The end is floored to 3 decimals so it never exceeds duration.
    Returns a list of {"segment_id", "start", "end"}.
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")
    segments: List[Dict[str, float]] = []
    start = 0.0
    sid = 0
    while start < duration:
        end = min(start + window_size, duration)
        segments.append({
            "segment_id": sid,
            "start": round(start, 3),
            "end": math.floor(end * 1000) / 1000,
        })
        start = end
        sid += 1
    return segments


def default_memory_path(video_path: str) -> str:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return str(config.DATA_DIR / "memory" / f"{stem}_memory.json")


def load_video_memory(path: str) -> Optional[Dict[str, Any]]:
    """Load a memory JSON file, or return None if it does not exist."""
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_video_memory(memory: Dict[str, Any], path: str) -> str:
    """Persist memory as UTF-8 JSON. Creates parent directories as needed."""
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)
    return path


def build_video_memory(
    video_path: str,
    window_size: float = 60,
    frame_interval: float = 10,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Build coarse video memory, resuming from an existing file if present.

    Loads the model once, processes each segment independently (frames are
    released after each segment), and saves incrementally so a failure can be
    resumed without redoing finished segments.
    """
    duration = video_tool.get_video_duration(video_path)
    out_path = output_path or default_memory_path(video_path)

    existing = load_video_memory(out_path)
    if (
        existing is not None
        and existing.get("duration") == duration
        and existing.get("window_size") == window_size
        and existing.get("frame_interval") == frame_interval
    ):
        memory = existing
        built_ids = {seg["segment_id"] for seg in memory.get("segments", [])}
    else:
        memory = {
            "video_path": os.path.abspath(video_path),
            "duration": duration,
            "window_size": window_size,
            "frame_interval": frame_interval,
            "segments": [],
        }
        built_ids = set()

    segments = _split_segments(duration, window_size)
    model = vlm_tool.load_model()

    for seg in segments:
        sid = seg["segment_id"]
        if sid in built_ids:
            continue
        frames = video_tool.sample_frames(
            video_path,
            start_time=seg["start"],
            end_time=seg["end"],
            interval=frame_interval,
        )
        prompt = vlm_tool.build_frames_prompt(frames, MEMORY_QUESTION)
        result = vlm_tool.analyze_frames(
            frames, prompt, model=model.model, processor=model.processor
        )
        memory["segments"].append({
            "segment_id": sid,
            "start": seg["start"],
            "end": seg["end"],
            "summary": result.get("summary", ""),
            "events": result.get("events", []),
        })
        memory["segments"].sort(key=lambda s: s["segment_id"])
        save_video_memory(memory, out_path)

    save_video_memory(memory, out_path)
    return memory


def build_hierarchy(memory: Dict[str, Any], output_path: Optional[str] = None) -> Dict[str, Any]:
    """Add global_summary + chapters (MemDreamer-style hierarchy) to a memory.

    Chapters are contiguous groups of adjacent segments with semantic
    continuity. If already present, returns the memory unchanged.
    """
    if memory.get("global_summary") and memory.get("chapters"):
        return memory

    segments = memory.get("segments", [])
    model = vlm_tool.load_model()

    if not memory.get("global_summary"):
        lines = ["Summarize the overall content of this video from its segment summaries:", ""]
        for s in segments:
            lines.append(f"[{s['start']}s-{s['end']}s] {s.get('summary', '')}")
        lines += ['', 'Return ONLY JSON: {"global_summary": "one-paragraph summary"}']
        raw = vlm_tool.generate_text("\n".join(lines), model=model.model, processor=model.processor)
        memory["global_summary"] = vlm_tool._extract_json(raw).get("global_summary", "")

    if not memory.get("chapters") and segments:
        lines = ["Group these CONSECUTIVE segments into chapters (scenes). Only group ADJACENT segments.", ""]
        for s in segments:
            lines.append(f"segment {s['segment_id']}: [{s['start']}s-{s['end']}s] {s.get('summary', '')}")
        lines += ['', 'Return ONLY JSON: {"chapters": [{"segment_ids": [0, 1], "summary": "..."}]}']
        raw = vlm_tool.generate_text("\n".join(lines), model=model.model, processor=model.processor)
        parsed = vlm_tool._extract_json(raw)
        seg_by_id = {s["segment_id"]: s for s in segments}
        chapters = []
        for i, c in enumerate(parsed.get("chapters", [])):
            ids = [sid for sid in c.get("segment_ids", []) if sid in seg_by_id]
            if not ids:
                continue
            ids.sort()
            chapters.append({
                "chapter_id": i,
                "start": seg_by_id[ids[0]]["start"],
                "end": seg_by_id[ids[-1]]["end"],
                "summary": c.get("summary", ""),
                "segment_ids": ids,
            })
        memory["chapters"] = chapters

    if output_path:
        save_video_memory(memory, output_path)
    return memory
