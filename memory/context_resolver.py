"""Context resolver: pronoun/coreference resolution using rules + memory.

Resolves 他/她/它/他们/那个人/那个物体 etc. against the Conversation Memory's
known entities and current subject. When ambiguous, it does NOT guess — it
returns resolution_status = "ambiguous" with candidates.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

_PRONOUNS = {
    "他们": {"kind": "group"},
    "那个人": {"kind": "person"},
    "那个女人": {"kind": "female"},
    "那个男人": {"kind": "male"},
    "那个物体": {"kind": "object"},
    "刚才的人": {"kind": "person"},
    "他": {"kind": "male"},
    "她": {"kind": "female"},
    "它": {"kind": "object"},
}


def _replace_all(text: str, pronoun: str, entity: str) -> str:
    return text.replace(pronoun, entity)


def resolve_question(
    question: str,
    memory: Any,
) -> Dict[str, Any]:
    """Return {resolved_question, status, replacements, candidates}."""
    resolved = question
    replacements: Dict[str, str] = []
    candidates: List[str] = []
    entities = memory.known_entities()
    current = memory.current_subject()

    for pronoun, info in _PRONOUNS.items():
        if pronoun not in resolved:
            continue
        if info["kind"] == "male":
            entity = _pick_male(entities, current, memory)
        elif info["kind"] == "female":
            entity = _pick_female(entities, memory)
        elif info["kind"] == "object":
            entity = _pick_object(memory)
        else:  # person / group
            entity = current or (entities[0] if len(entities) == 1 else None)

        if entity is None:
            return {
                "resolved_question": resolved,
                "status": "ambiguous",
                "replacements": replacements,
                "candidates": candidates or entities,
            }
        resolved = _replace_all(resolved, pronoun, entity)
        replacements.append((pronoun, entity))

    status = "resolved" if replacements else "none"
    return {"resolved_question": resolved, "status": status, "replacements": replacements, "candidates": candidates}


def _pick_male(entities: List[str], current: Optional[str], memory: Any) -> Optional[str]:
    males = [e for e in entities if memory.entities.get(e, {}).get("gender") == "male"]
    if males:
        return males[-1]
    if current:
        return current
    if len(entities) == 1:
        return entities[0]
    return None


def _pick_female(entities: List[str], memory: Any) -> Optional[str]:
    females = [e for e in entities if memory.entities.get(e, {}).get("gender") == "female"]
    if females:
        return females[-1]
    return None


def _pick_object(memory: Any) -> Optional[str]:
    for e in reversed(memory.confirmed_events):
        if e.get("object"):
            return e["object"]
    return None
