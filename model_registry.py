"""Model registry + backend runtime status for the UI/API.

Reports which models are actually available (only Qwen3-VL-8B local right now)
without pretending remote API providers are configured.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import config

MODELS: List[Dict[str, Any]] = [
    {
        "id": "qwen3-vl-8b-local",
        "name": "Qwen3-VL-8B (Local)",
        "kind": "local",
        "available": True,
        "description": "本地 Qwen3-VL-8B，视觉 + 语言推理（已配置）",
    },
    {"id": "qwen-api", "name": "Qwen API", "kind": "api", "available": False, "description": "未配置"},
    {"id": "openai-api", "name": "OpenAI API", "kind": "api", "available": False, "description": "未配置"},
    {"id": "gemini-api", "name": "Gemini API", "kind": "api", "available": False, "description": "未配置"},
]


def get_models() -> List[Dict[str, Any]]:
    return MODELS


def get_backend_status() -> Dict[str, Any]:
    mem_dir = config.DATA_DIR / "memory"
    cached = [f for f in os.listdir(mem_dir) if f.endswith("_memory.json")] if mem_dir.is_dir() else []
    return {
        "runtime": "native-python",
        "model": "qwen3-vl-8b-local",
        "model_loaded": bool(_model_loaded()),
        "memory_cache": {"dir": str(mem_dir), "n_cached": len(cached)},
    }


def _model_loaded() -> bool:
    try:
        from tools import vlm_tool
        return vlm_tool._loaded is not None
    except Exception:
        return False
