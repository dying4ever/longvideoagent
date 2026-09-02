"""Video utilities for the LongVideoAgent MVP.

Responsibilities:
  * get_video_duration(video_path) -> float (seconds)
  * sample_frames(video_path, start_time, end_time, interval) -> list[VideoFrame]
  * cut_clip(video_path, start_time, end_time, output_path) -> output_path

Conventions:
  * All time values are in SECONDS.
  * Frames are decoded one at a time (a long video is never fully loaded into RAM).
  * This module has NO dependency on the VLM, so it can be used and swapped
    independently of the model backend.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
from PIL import Image


class VideoError(Exception):
    """Raised when a video operation fails (missing file, bad range, ...)."""


@dataclass
class VideoFrame:
    """One sampled video frame with its timestamp in seconds."""

    timestamp: float       # seconds, as precise as the source allows
    frame_index: int       # source frame number
    image: Image.Image     # RGB PIL image


def _require_ffprobe() -> None:
    if shutil.which("ffprobe") is None:
        raise VideoError("ffprobe not found; install ffmpeg/ffprobe first.")


def get_video_duration(video_path: str) -> float:
    """Return video duration in seconds. Raises VideoError on failure."""
    if not os.path.isfile(video_path):
        raise VideoError(f"video not found: {video_path}")

    _require_ffprobe()
    proc = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1",
         video_path],
        capture_output=True, text=True,
    )
    if proc.returncode == 0:
        try:
            duration = float(proc.stdout.strip())
            if duration > 0:
                return duration
        except ValueError:
            pass

    # Fallback via OpenCV: frame_count / fps
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoError(f"could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps <= 0:
        raise VideoError(f"invalid fps for video: {video_path}")
    return n_frames / fps


def _resolve_range(
    video_path: str,
    start_time: Optional[float],
    end_time: Optional[float],
) -> Tuple[float, float]:
    """Validate and resolve (start, end) in seconds, clamping end to duration."""
    duration = get_video_duration(video_path)
    start = 0.0 if start_time is None else float(start_time)
    end = duration if end_time is None else float(end_time)
    if start < 0:
        raise VideoError(f"start_time must be >= 0, got {start}")
    if end <= start:
        raise VideoError(f"end_time ({end}) must be > start_time ({start})")
    if start >= duration:
        raise VideoError(
            f"start_time ({start}) is out of range: video duration is {duration:.3f}s")
    if end > duration:
        raise VideoError(
            f"end_time ({end}) is out of range: video duration is {duration:.3f}s")
    return start, end


def sample_frames(
    video_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    interval: float = 2.0,
) -> List[VideoFrame]:
    """Sample frames every `interval` seconds within [start_time, end_time).

    Returns a list of VideoFrame ordered by timestamp. Frames are decoded one
    at a time so memory stays bounded for long videos.
    """
    if interval <= 0:
        raise VideoError(f"interval must be > 0, got {interval}")
    start, end = _resolve_range(video_path, start_time, end_time)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoError(f"could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release()
        raise VideoError(f"invalid fps for video: {video_path}")

    frames: List[VideoFrame] = []
    try:
        i = 0
        while True:
            t = start + i * interval
            if t >= end:
                break
            frame_index = int(round(t * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, bgr = cap.read()
            if not ok:
                break  # reached EOF early; stop instead of erroring
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(VideoFrame(
                timestamp=round(frame_index / fps, 3),
                frame_index=frame_index,
                image=Image.fromarray(rgb),
            ))
            i += 1
    finally:
        cap.release()
    return frames


def cut_clip(
    video_path: str,
    start_time: float,
    end_time: float,
    output_path: str,
) -> str:
    """Cut [start_time, end_time) into a new file via ffmpeg stream copy.

    Fast and lossless, but seeks to the nearest keyframe, so the exact start
    frame is approximate. Returns output_path on success.
    """
    if not os.path.isfile(video_path):
        raise VideoError(f"video not found: {video_path}")
    if shutil.which("ffmpeg") is None:
        raise VideoError("ffmpeg not found; install ffmpeg first.")
    start, end = _resolve_range(video_path, start_time, end_time)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-i", video_path, "-c", "copy", output_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise VideoError(f"cut_clip failed: {proc.stderr.strip()}")
    return output_path
