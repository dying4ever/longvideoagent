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
from typing import Any, Callable, Dict, List, Optional

import config
from tools import video_tool, vlm_tool
from utils import profiler

MEMORY_QUESTION = "这段视频里发生了什么？请描述画面中的场景、人物、物体和动作。"


def video_fingerprint(video_path: str, chunk_size: int = 1 << 20) -> str:
    """Return a content hash of the video (first 16 sha256 hex chars)."""
    import hashlib

    h = hashlib.sha256()
    with open(video_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


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
    fp = video_fingerprint(video_path)
    return str(config.DATA_DIR / "memory" / f"{fp}_memory.json")


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


def _build_event_prompt(frames, window_start: float, window_end: float, carry: Optional[Dict]) -> str:
    lines = [
        "You are segmenting a video into semantic EVENTS.",
        f"You observe a window from {window_start}s to {window_end}s.",
        "Each frame has an EXPLICIT timestamp (seconds) provided by the program; it is the REAL frame time.",
        "",
        "An EVENT is a continuous scene/action/topic (e.g. '佩奇拆礼物盒', '乔治进门', '全家吃甜甜圈').",
    ]
    if carry:
        lines.append(
            f"NOTE: an ongoing event started at {carry['start']}s and continues into this window. "
            f"Merge it with the first event you detect (use {carry['start']}s as its start)."
        )
    lines += ["", "Frames (chronological order):"]
    for i, f in enumerate(frames, 1):
        lines.append(f"Frame {i}: timestamp = {f.timestamp} s")
    lines += [
        "",
        "Instructions:",
        "1. Identify each COMPLETE event with its boundary (start/end timestamp in seconds).",
        "2. For each event, give a short summary + entities (people/objects) + actions.",
        "3. If the LAST event is still ongoing at the last frame, do NOT list it as a complete event; "
        "put it in continuing_event with its start timestamp.",
        "4. Boundaries must be within the window.",
        "5. Output ALL text (summary/entities/actions) in Chinese.",
        "6. Return ONLY valid JSON matching this schema:",
        json.dumps({
            "events": [
                {"start": 0.0, "end": 25.0, "summary": "...", "entities": ["..."], "actions": ["..."]}
            ],
            "continuing_event": {"start": 55.0, "summary": "..."},
        }, ensure_ascii=False),
    ]
    return "\n".join(lines)


@profiler.timed("event_segmentation")
def segment_events(
    video_path: str,
    window_size: float = 60,
    frame_interval: float = 5,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """Adaptive event segmentation.

    The fixed `window_size` only bounds a single observation; the VLM detects
    adaptive event boundaries WITHIN each window. An event that is still
    ongoing at the window edge is carried into the next window (merged), so a
    semantic event is never split by the fixed window boundary.

    `progress_cb`, when provided, receives progress dicts: {"phase": "loading"},
    then {"phase": "segmenting", "window": i, "total": n} per window.
    """
    if progress_cb:
        progress_cb({"phase": "loading"})
    duration = video_tool.get_video_duration(video_path)
    model = vlm_tool.load_model()
    events: List[Dict[str, Any]] = []
    carry: Optional[Dict] = None
    window_start = 0.0
    eid = 0
    total_windows = max(1, int(math.ceil(duration / window_size)))
    window_index = 0

    while window_start < duration:
        window_end = min(window_start + window_size, duration)
        if progress_cb:
            progress_cb({"phase": "segmenting", "window": window_index, "total": total_windows})
        frames = video_tool.sample_frames(
            video_path, start_time=window_start, end_time=window_end, interval=frame_interval
        )
        if not frames:
            break
        prompt = _build_event_prompt(frames, window_start, window_end, carry)
        result = vlm_tool.analyze_frames(
            frames, prompt, model=model.model, processor=model.processor
        )
        for ev in result.get("events", []):
            try:
                s = float(ev["start"])
                e = float(ev["end"])
            except (TypeError, ValueError):
                continue
            s = max(s, window_start)
            e = min(e, window_end)
            if e <= s:
                continue
            events.append({
                "event_id": eid,
                "start": round(s, 3),
                "end": round(e, 3),
                "summary": str(ev.get("summary", "")),
                "entities": ev.get("entities", []),
                "actions": ev.get("actions", []),
            })
            eid += 1
        cont = result.get("continuing_event")
        if isinstance(cont, dict) and cont.get("start") is not None:
            carry = {"start": float(cont["start"]), "summary": str(cont.get("summary", ""))}
        else:
            carry = None
        window_start = window_end
        window_index += 1

    return events


def _events_to_segments(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for e in events:
        text = " ".join(
            [e["summary"]] + list(e.get("entities", [])) + list(e.get("actions", []))
        )
        result.append({
            "segment_id": e["event_id"],
            "start": e["start"],
            "end": e["end"],
            "summary": e["summary"],
            "events": [{"timestamp": e["start"], "description": text}],
        })
    return result


@profiler.timed("hierarchy_build")
def _add_event_hierarchy(memory: Dict[str, Any]) -> None:
    events = memory.get("events", [])
    model = vlm_tool.load_model()

    lines = ["Summarize the overall content of this video from its event summaries:", ""]
    for e in events:
        lines.append(f"[{e['start']}s-{e['end']}s] {e.get('summary', '')}")
    lines += ['', 'Return ONLY JSON: {"global_summary": "one-paragraph summary"}']
    raw = vlm_tool.generate_text("\n".join(lines), model=model.model, processor=model.processor)
    memory["global_summary"] = vlm_tool._extract_json(raw).get("global_summary", "")

    lines = ["Group these CONSECUTIVE events into chapters (scenes). Only group ADJACENT events.", ""]
    for e in events:
        lines.append(f"event {e['event_id']}: [{e['start']}s-{e['end']}s] {e.get('summary', '')}")
    lines += ['', 'Return ONLY JSON: {"chapters": [{"event_ids": [0, 1], "summary": "..."}]}']
    raw = vlm_tool.generate_text("\n".join(lines), model=model.model, processor=model.processor)
    parsed = vlm_tool._extract_json(raw)
    ev_by_id = {e["event_id"]: e for e in events}
    chapters = []
    for i, c in enumerate(parsed.get("chapters", [])):
        ids = [eid for eid in c.get("event_ids", []) if eid in ev_by_id]
        if not ids:
            continue
        ids.sort()
        chapters.append({
            "chapter_id": i,
            "start": ev_by_id[ids[0]]["start"],
            "end": ev_by_id[ids[-1]]["end"],
            "summary": c.get("summary", ""),
            "event_ids": ids,
            "segment_ids": ids,
        })
    memory["chapters"] = chapters


def build_event_memory(
    video_path: str,
    window_size: float = 60,
    frame_interval: float = 5,
    output_path: Optional[str] = None,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Build adaptive-event video memory (events + hierarchy).

    `events` hold adaptive boundaries; `segments` is mapped from events for
    backward compatibility with grounding (which reads `segments`).
    """
    duration = video_tool.get_video_duration(video_path)
    out_path = output_path or default_memory_path(video_path)
    events = segment_events(
        video_path, window_size=window_size, frame_interval=frame_interval,
        progress_cb=progress_cb,
    )
    if progress_cb:
        progress_cb({"phase": "hierarchy"})
    memory = {
        "video_path": os.path.abspath(video_path),
        "duration": duration,
        "fingerprint": video_fingerprint(video_path),
        "memory_version": config.MEMORY_VERSION,
        "window_size": window_size,
        "frame_interval": frame_interval,
        "events": events,
        "segments": _events_to_segments(events),
    }
    _add_event_hierarchy(memory)
    save_video_memory(memory, out_path)
    return memory


def memory_matches(
    memory: Dict[str, Any],
    video_path: str,
    window_size: float,
    frame_interval: float,
) -> bool:
    """Return True if a cached memory is reusable for the given video + config."""
    return (
        memory.get("events") is not None
        and memory.get("memory_version") == config.MEMORY_VERSION
        and memory.get("fingerprint") == video_fingerprint(video_path)
        and memory.get("window_size") == window_size
        and memory.get("frame_interval") == frame_interval
    )
