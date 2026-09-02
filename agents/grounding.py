"""Visual Grounding: locate candidate time intervals for a user question.

Strategy (no dedicated model trained):
  1. Collect text (summary + events) of every memory segment.
  2. Lightweight coarse filter (token overlap) to shrink the candidate set
     for long videos, so hundreds of segments are never dumped to the model.
  3. Ask Qwen3-VL (text-only) to pick the most relevant segments.
  4. Map segment ids back to real [start, end) time ranges.

Grounding only answers "where to look"; it does not produce a final answer.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from tools import vlm_tool
from utils import profiler

MAX_SEGMENTS_FOR_MODEL = 8


def _tokens(text: str) -> set:
    """Extract a lightweight token set (English words + CJK char bigrams)."""
    text = text.lower()
    words = set(re.findall(r"[a-z0-9]+", text))
    clean = re.sub(r"[^\u4e00-\u9fffa-z0-9]", "", text)
    bigrams = {clean[i:i + 2] for i in range(len(clean) - 1)} if len(clean) >= 2 else set()
    return words | bigrams


def _coarse_score(question: str, segment_text: str) -> float:
    q = _tokens(question)
    s = _tokens(segment_text)
    if not q:
        return 0.0
    return len(q & s) / len(q)


def _segment_text(seg: Dict[str, Any]) -> str:
    parts = [seg.get("summary", "")]
    for ev in seg.get("events", []):
        parts.append(ev.get("description", ""))
    return " ".join(parts)


def coarse_filter(question: str, segments: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    """Return up to top_n segments ranked by token-overlap with the question."""
    scored = [(_coarse_score(question, _segment_text(s)), s) for s in segments]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [seg for _, seg in scored[:top_n]]


def _hierarchical_prune(
    question: str,
    video_memory: Dict[str, Any],
    segments: List[Dict[str, Any]],
    max_chapters: int = 5,
) -> List[Dict[str, Any]]:
    """If hierarchical chapters exist, first pick relevant chapters, then keep
    only segments within those chapters (MemDreamer-style coarse-to-fine)."""
    chapters = video_memory.get("chapters")
    if not chapters or len(chapters) <= max_chapters:
        return segments
    scored = sorted(
        chapters, key=lambda c: _coarse_score(question, c.get("summary", "")), reverse=True
    )
    kept_ids = {sid for c in scored[:max_chapters] for sid in c.get("segment_ids", [])}
    return [s for s in segments if s.get("segment_id") in kept_ids]


def _build_grounding_prompt(question: str, segments: List[Dict[str, Any]], top_k: int) -> str:
    lines = [
        "You are a video temporal grounding assistant.",
        "Given a user question and a list of video segment summaries (with time ranges in seconds),",
        "select the segments most likely to contain the answer.",
        "",
        f"Question: {question}",
        "",
        "Segments:",
    ]
    for seg in segments:
        lines.append(
            f"- segment_id={seg['segment_id']}: [{seg['start']}s - {seg['end']}s] {seg.get('summary', '')}"
        )
    lines += [
        "",
        "Instructions:",
        f"1. Return at most {top_k} segments, ordered by relevance (highest first).",
        "2. score is a 0-1 relevance score.",
        "3. reason is a short justification based on the summary.",
        "4. Return ONLY valid JSON matching this schema:",
        json.dumps({"candidates": [{"segment_id": 0, "score": 0.9, "reason": "..."}]}),
    ]
    return "\n".join(lines)


@profiler.timed("grounding")
def ground_video(
    question: str,
    video_memory: Dict[str, Any],
    top_k: int = 3,
    search_start: float = None,
    search_end: float = None,
) -> Dict[str, Any]:
    """Return candidate time intervals (with scores/reasons) for a question.

    If search_start / search_end are given, only memory segments overlapping
    [search_start, search_end) are considered.
    """
    segments = video_memory.get("segments", [])
    if search_start is not None or search_end is not None:
        s = 0.0 if search_start is None else float(search_start)
        e = float("inf") if search_end is None else float(search_end)
        segments = [
            seg for seg in segments
            if float(seg["end"]) > s and float(seg["start"]) < e
        ]
    if not segments:
        return {"query": question, "candidates": []}

    segments = _hierarchical_prune(question, video_memory, segments)

    if len(segments) > MAX_SEGMENTS_FOR_MODEL:
        candidates = coarse_filter(question, segments, MAX_SEGMENTS_FOR_MODEL)
    else:
        candidates = segments

    model = vlm_tool.load_model()
    prompt = _build_grounding_prompt(question, candidates, top_k)
    raw = vlm_tool.generate_text(prompt, model=model.model, processor=model.processor)
    parsed = vlm_tool._extract_json(raw)

    seg_by_id = {s["segment_id"]: s for s in segments}
    out: List[Dict[str, Any]] = []
    for c in parsed.get("candidates", []):
        sid = c.get("segment_id")
        seg = seg_by_id.get(sid)
        if seg is None:
            continue
        out.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "score": float(c.get("score", 0.0)),
            "reason": str(c.get("reason", "")),
        })
    out.sort(key=lambda x: x["score"], reverse=True)
    return {"query": question, "candidates": out[:top_k]}
