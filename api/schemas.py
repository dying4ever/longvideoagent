"""Pydantic schemas for the LongVideoAgent API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class SessionCreateRequest(BaseModel):
    video_id: str


class AskResponse(BaseModel):
    answer: str
    status: str
    timestamp: Optional[float] = None
    resolved_question: Optional[str] = None
    temporal_type: Optional[str] = None
    reference_timestamp: Optional[float] = None
    evidence: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []
    working_memory: Dict[str, Any] = {}
    conversation_context: Dict[str, Any] = {}
