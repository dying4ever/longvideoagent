"""Lightweight API tests (no model loading).

Uses FastAPI TestClient with mocked session_manager, so no Qwen checkpoint is
loaded.

Run:  python tests/test_api.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

import api.app as app_module  # noqa: E402
from api import session_manager  # noqa: E402


class _FakeConversation:
    def __init__(self):
        self.working_memory = {"current_subject": "乔治"}
        self.entities = {"乔治": {"gender": "male"}}
        self.confirmed_events = []
        self.tentative_events = []
        self.turns = []


class _FakeSession:
    def __init__(self):
        self.duration = 285.0
        self.video_memory = {"segments": [], "global_summary": "s", "chapters": []}
        self.conversation = _FakeConversation()

    def ask(self, question):
        return {
            "current_answer": "首次出现时间约在 6.0s", "status": "finished",
            "final_timestamp": 6.0, "resolved_question": question,
            "temporal_type": "FIRST", "reference_timestamp": None,
            "evidence": [], "trace": [],
        }

    def reset(self):
        self.conversation = _FakeConversation()


def test_health() -> None:
    client = TestClient(app_module.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    print("[ok] GET /health")


def test_video_upload() -> None:
    client = TestClient(app_module.app)
    r = client.post("/videos", files={"file": ("t.mp4", io.BytesIO(b"fake-video"), "video/mp4")})
    assert r.status_code == 200
    assert "video_id" in r.json()
    print("[ok] POST /videos upload")


def test_session_ask_memory_trace_reset() -> None:
    client = TestClient(app_module.app)
    app_module._videos["vid1"] = "/tmp/vid1.mp4"
    with mock.patch.object(session_manager, "create_session", return_value=("sid1", _FakeSession())), \
         mock.patch.object(session_manager, "get_session", return_value=_FakeSession()):
        r = client.post("/sessions", json={"video_id": "vid1"})
        assert r.status_code == 200
        assert r.json()["session_id"] == "sid1"

        r = client.post("/sessions/sid1/ask", json={"question": "乔治第一次什么时候出现？"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "首次出现时间约在 6.0s"
        assert body["status"] == "finished"

        r = client.get("/sessions/sid1/memory")
        assert r.status_code == 200
        assert "working_memory" in r.json()

        r = client.get("/sessions/sid1/trace")
        assert r.status_code == 200

        r = client.post("/sessions/sid1/reset")
        assert r.status_code == 200
        assert r.json()["status"] == "reset"
    print("[ok] session create / ask / memory / trace / reset")


if __name__ == "__main__":
    test_health()
    test_video_upload()
    test_session_ask_memory_trace_reset()
    print("ALL API TESTS PASSED")
