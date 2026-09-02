---
name: long-video-agent
description: Interactive long-video understanding with planning, hierarchical memory, visual grounding, temporal reasoning, critic verification and multi-turn memory. Use when the user uploads/specifies a long video, asks about video content, asks to locate when an event happened, asks consecutive questions about the same video, or asks first/last/before/after/repeat/always temporal relations.
---

# Long Video Agent

Interactive long-video understanding agent. It plans, grounds, reasons, verifies
and remembers across multiple turns of conversation about a single video.

## When to use

Trigger when the user:

1. Uploads or specifies a long video.
2. Asks about video content ("这段视频里发生了什么").
3. Asks to locate when an event happened ("乔治第一次什么时候出现").
4. Asks consecutive questions about the same video.
5. Asks first / last / before / after / repeat / always temporal relations.

## How to invoke

Do NOT re-implement the agent. Run the adapter CLI — it wraps the single
`LongVideoAgentSession` and persists the conversation across turns:

```bash
/home/ps/.miniconda3/envs/zjx_openvla/bin/python /mnt/sda/zjx_space/agent/openclaw_adapter.py \
  --video <path-to-video> --question "<user question>"
```

The command prints JSON with `answer`, `status`, `timestamp`, `temporal_type`,
`resolved_question` and `evidence`.

- Use the FULL conda python path (zjx_openvla has torch + Qwen3-VL).
- The conversation memory is persisted next to the video memory, so asking the
  same video again automatically reuses confirmed entities / reference events /
  occurrences (multi-turn).
- Do NOT pre-load the model or rebuild memory yourself — the adapter handles it.

## Workflow (internal, for context)

```
Session → Conversation Memory → Context Resolve → Temporal Parse
→ Hierarchical Memory Retrieve → Planner → Visual Grounding → Visual Reasoning
→ Temporal Verification → Visual Critic → (Replan if needed) → Final Answer
→ Memory Update
```

## Prohibited

- Do NOT generate facts without visual evidence.
- Do NOT treat a local "first" as a global "first".
- Do NOT write tentative events into confirmed memory.
- Do NOT re-search an already-confirmed reference event.
- Do NOT claim ALWAYS when the video is not fully covered.
- Do NOT count consecutive frames as multiple REPEAT occurrences.

See `references/architecture.md` and `references/api.md` for details.
