# Architecture

LongVideoAgent pipeline:

```
Video
  → Hierarchical Video Memory (global_summary → chapters → segments → events)
  → Conversation Session
     Q → Context Resolver → Temporal Parser → Planner → Grounding → Reasoning
       → Temporal Verifier → Visual Critic → (Replan) → Answer
       → Conversation Memory Update → Working Memory Update
```

## Components

| Module | Role |
|---|---|
| `tools/video_tool.py` | frame sampling (with timestamps), clip cutting |
| `tools/vlm_tool.py` | Qwen3-VL-8B image understanding + text generation (singleton) |
| `memory/video_memory.py` | coarse + hierarchical video memory |
| `memory/conversation_memory.py` | multi-turn conversation + working memory |
| `memory/context_resolver.py` | pronoun / coreference resolution |
| `temporal/parser.py` | temporal intent + target/reference_event extraction |
| `temporal/verifier.py` | FIRST/LAST/REPEAT/ALWAYS/BEFORE/AFTER coverage verification |
| `agents/planner.py` | next-action decision (deterministic for temporal types) |
| `agents/grounding.py` | hierarchical candidate retrieval |
| `agents/reasoning.py` | local visual observation (occurrences + violations) |
| `agents/critic.py` | rule + LLM evidence sufficiency check |
| `session.py` | multi-turn orchestration (LongVideoAgentSession) |

## Key principles

- Model loads once per process (singleton), shared across turns.
- Memory is built once, shared across turns.
- Temporal semantics (first/last/repeat/always) are decided by deterministic
  coverage rules, NOT by the VLM claiming "first" locally.
- Confirmed facts are reused; new visual facts always go back to raw frames.
