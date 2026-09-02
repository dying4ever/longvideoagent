"""Temporal query parser: classify the global temporal intent of a question.

Rule-based keyword matching first; LLM is only consulted as a fallback for
ambiguous phrasings. This never loads a separate model — it reuses
`vlm_tool.generate_text` when the rules cannot decide.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from tools import vlm_tool

FIRST = "FIRST"
LAST = "LAST"
REPEAT = "REPEAT"
BEFORE = "BEFORE"
AFTER = "AFTER"
ALWAYS = "ALWAYS"
NORMAL = "NORMAL"

_KEYWORDS = {
    ALWAYS: ["一直", "始终", "全程", "总是", "always"],
    FIRST: ["第一次", "首次", "最早", "first", "earliest"],
    LAST: ["最后一次", "最晚", "last", "latest"],
    BEFORE: ["之前", "以前", "before"],
    AFTER: ["之后", "后来", "after"],
    REPEAT: ["再次", "又一次", "有没有再", "again"],
}

_LLM_FALLBACK_PROMPT = (
    "Classify the temporal intent of this video question into ONE type:\n"
    "FIRST (第一次/首次/最早), LAST (最后一次/最晚), REPEAT (再次/又), "
    "BEFORE (之前), AFTER (之后), ALWAYS (一直/始终), or NORMAL.\n"
    "Question: {question}\n"
    'Return ONLY JSON: {{"type": "FIRST"}}'
)


def _rule_match(question_lower: str) -> Optional[str]:
    for ttype in (ALWAYS, FIRST, LAST, BEFORE, AFTER, REPEAT):
        for kw in _KEYWORDS[ttype]:
            if kw in question_lower:
                return ttype
    return None


def parse_temporal_query(
    question: str,
    model=None,
    processor=None,
) -> Dict[str, Any]:
    """Return {"type", "target", "reference_event"} for a question."""
    ttype = _rule_match(question.lower())
    if ttype is not None:
        return {"type": ttype, "target": None, "reference_event": None}

    if model is None or processor is None:
        vlm = vlm_tool.load_model()
        model, processor = vlm.model, vlm.processor

    raw = vlm_tool.generate_text(
        _LLM_FALLBACK_PROMPT.format(question=question),
        model=model, processor=processor, max_new_tokens=64,
    )
    parsed = vlm_tool._extract_json(raw)
    ttype = str(parsed.get("type", NORMAL)).upper()
    if ttype not in (FIRST, LAST, REPEAT, BEFORE, AFTER, ALWAYS):
        ttype = NORMAL
    return {"type": ttype, "target": None, "reference_event": None}
