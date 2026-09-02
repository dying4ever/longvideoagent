"""LongVideoAgent MVP — central configuration.

Every path and inference setting lives here so no other module hardcodes the
model path or device. The local Qwen3-VL model path is auto-detected.

Detection order:
  1. env var LONGVIDEO_MODEL_PATH
  2. <project_root>/model
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos"
FRAME_DIR = DATA_DIR / "frames"

# --- Model -----------------------------------------------------------------
MODEL_REPO_ID = "Qwen/Qwen3-VL-8B-Instruct"
ENV_MODEL_PATH = "LONGVIDEO_MODEL_PATH"

_MODEL_CANDIDATES = [
    PROJECT_ROOT / "model",
]


def _looks_like_model(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def detect_model_path() -> str | None:
    """Return the local Qwen3-VL model directory, or None if not present."""
    explicit = os.environ.get(ENV_MODEL_PATH)
    if explicit:
        p = Path(explicit).expanduser()
        if _looks_like_model(p):
            return str(p.resolve())
    for cand in _MODEL_CANDIDATES:
        if _looks_like_model(cand):
            return str(cand.resolve())
    return None


# --- Inference defaults ------------------------------------------------------
DEVICE_MAP = "auto"       # single-GPU friendly; "cuda:0" forces one GPU
DTYPE = "auto"            # Qwen3-VL uses `dtype` (not `torch_dtype`); "auto" derives bf16 from weights
MAX_NEW_TOKENS = 1024
DEFAULT_INTERVAL = 2.0
