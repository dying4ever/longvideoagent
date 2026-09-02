# API

The OpenClaw skill delegates to `openclaw_adapter`, which wraps
`LongVideoAgentSession`. The same session powers the CLI, OpenClaw and the Web API.

## Python adapter

```python
import openclaw_adapter as lva

s = lva.create_session("/path/to/video.mp4")   # → {"session_id": "...", ...}
r = lva.ask(s["session_id"], "乔治第一次什么时候出现？")
# r = {"current_answer": "首次出现时间约在 6.0s", "status": "finished",
#      "final_timestamp": 6.0, "evidence": [...], "trace": [...], ...}

lva.get_memory(s["session_id"])    # working + conversation + video memory
lva.get_trace(s["session_id"])     # per-turn trace
lva.reset_session(s["session_id"]) # clear conversation memory only
```

## ask() result fields

| field | meaning |
|---|---|
| `current_answer` | natural-language answer |
| `status` | `finished` / `max_iterations_reached` |
| `final_timestamp` | key timestamp (FIRST/LAST/REPEAT) |
| `reference_timestamp` | reused reference event time (BEFORE/AFTER) |
| `evidence` | list of `{timestamp, description}` |
| `trace` | step-by-step agent decisions (no hidden CoT) |
| `resolved_question` | question after coreference resolution |

## Runtime note

This bundle is ready to be discovered by an OpenClaw runtime. It was authored
without an installed OpenClaw CLI on the build machine; the SKILL.md format
follows the frontmatter convention (name + description).
