"""Thread-safe in-memory progress registry for memory building.

The memory build runs inside the synchronous ``POST /sessions`` endpoint
(executed on FastAPI's threadpool), while ``GET /progress/{video_id}`` reads
from another thread, so access is guarded by a lock.
"""
from __future__ import annotations

from threading import Lock
from typing import Any, Dict, Optional

_lock = Lock()
_registry: Dict[str, Dict[str, Any]] = {}


def set_progress(video_id: str, info: Dict[str, Any]) -> None:
    with _lock:
        _registry[video_id] = info


def get_progress(video_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        return _registry.get(video_id)


def clear_progress(video_id: str) -> None:
    with _lock:
        _registry.pop(video_id, None)
