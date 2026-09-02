"""Qwen3-VL frame understanding: timestamped frames -> structured JSON.

Model loading is centralized and cached so it happens ONCE per process, then
`analyze_frames` can be called repeatedly.

This module is decoupled from video_tool: it accepts any iterable of objects
with `.image` (PIL.Image) and `.timestamp` (float). Swapping the backend
(Qwen3-VL-8B -> 32B -> remote API VLM) does NOT require touching video_tool.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import config


class VLMError(Exception):
    """Raised on model load / generation / output-parsing failures."""


class _VLM:
    """Thin wrapper holding a loaded model + processor."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor


# module-level cache: model + processor loaded exactly once per process
_loaded: Optional[_VLM] = None

_CALL_COUNTS = {"analyze_frames": 0, "generate_text": 0}


def get_call_counts() -> Dict[str, int]:
    return dict(_CALL_COUNTS)


def reset_call_counts() -> None:
    _CALL_COUNTS["analyze_frames"] = 0
    _CALL_COUNTS["generate_text"] = 0


def _require_transformers() -> None:
    try:
        import transformers
    except ImportError as e:
        raise VLMError(
            "transformers is not installed. Install it to use Qwen3-VL."
        ) from e
    try:
        from packaging.version import Version
    except ImportError:
        return
    if Version(transformers.__version__) < Version("4.51.0"):
        raise VLMError(
            f"transformers {transformers.__version__} is too old for Qwen3-VL; "
            "need >= 4.51.0 (4.57.x recommended)."
        )


def load_model(
    model_path: Optional[str] = None,
    device_map: str = config.DEVICE_MAP,
    dtype: str = config.DTYPE,
) -> "_VLM":
    """Load Qwen3-VL once and cache it. Subsequent calls return the cache."""
    global _loaded
    if _loaded is not None:
        return _loaded
    _require_transformers()

    path = model_path or config.detect_model_path()
    if path is None:
        raise VLMError(
            "No local Qwen3-VL model found. Set LONGVIDEO_MODEL_PATH or place "
            "the model under <project_root>/model."
        )

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, dtype=dtype, device_map=device_map, local_files_only=True
        )
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    except (FileNotFoundError, OSError) as e:
        raise VLMError(
            f"Could not load model from {path}. The weights may still be "
            f"downloading or incomplete: {e}"
        ) from e
    model.eval()
    _loaded = _VLM(model, processor)
    return _loaded


def build_frames_prompt(frames, question: str) -> str:
    """Compose the prompt, injecting explicit per-frame timestamps."""
    lines = [
        "You are analyzing a long video. Below are frames sampled in "
        "chronological order.",
        "Each frame has an EXPLICIT timestamp (in seconds) provided by the "
        "program; treat it as ground truth for WHEN an event happens. Do NOT "
        "infer any real-world clock time.",
        "",
        "Frames:",
    ]
    for i, f in enumerate(frames, 1):
        lines.append(f"Frame {i}: timestamp = {f.timestamp} s")
    lines += [
        "",
        f"Question: {question}",
        "",
        "Instructions:",
        "1. Understand the frames in chronological order.",
        "2. Pay attention to people, objects, actions and events.",
        "3. Use the explicitly provided timestamps to place events in time.",
        "4. Do NOT fabricate any information without visual evidence.",
        "5. If uncertain about something, say so explicitly.",
        "6. Respond with ONLY valid JSON matching this schema:",
        json.dumps({
            "summary": "one-sentence overall description",
            "events": [
                {"timestamp": 0.0, "description": "what happens at that time"}
            ],
        }, ensure_ascii=False),
    ]
    return "\n".join(lines)


def _extract_json(text: str) -> Any:
    """Best-effort extraction of a JSON object/array from model output."""
    if not text:
        raise VLMError("model returned empty output")

    # strip markdown code fences
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    text = text.strip()

    start = -1
    for opener in ("{", "["):
        idx = text.find(opener)
        if idx != -1 and (start == -1 or idx < start):
            start = idx
    if start == -1:
        raise VLMError(f"no JSON found in model output: {text[:200]!r}")

    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text[start:])
        return obj
    except json.JSONDecodeError:
        pass

    # fallback: scan to the matching closing bracket
    pairs = {"{": "}", "[": "]"}
    stack: List[str] = []
    close = None
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch in pairs:
            stack.append(pairs[ch])
        elif ch in ("}", "]"):
            if stack and stack[-1] == ch:
                stack.pop()
                if not stack:
                    close = idx + 1
                    break
    if close is not None:
        try:
            return json.loads(text[start:close])
        except json.JSONDecodeError as e:
            raise VLMError(f"could not parse JSON from model output: {e}") from e
    raise VLMError(f"could not parse JSON from model output: {text[:200]!r}")


def analyze_frames(
    frames,
    prompt: str,
    model=None,
    processor=None,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
) -> Any:
    """Run Qwen3-VL over a list of timestamped frames and return parsed JSON.

    `frames`: iterable of objects with `.image` (PIL.Image) and `.timestamp`.
    `model` / `processor`: optionally pass already-loaded HF objects; if either
        is None, the module-level cached model is used (loaded once).
    """
    import torch

    if model is None or processor is None:
        vlm = load_model()
        model, processor = vlm.model, vlm.processor

    images = [f.image for f in frames]
    if not images:
        raise VLMError("no frames provided to analyze_frames")

    messages = [{
        "role": "user",
        "content": [
            *[{"type": "image", "image": img} for img in images],
            {"type": "text", "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=images, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    _CALL_COUNTS["analyze_frames"] += 1
    generated = generated[:, inputs["input_ids"].shape[1]:]
    output_text = processor.batch_decode(generated, skip_special_tokens=True)[0]

    return _extract_json(output_text)


def generate_text(
    prompt: str,
    model=None,
    processor=None,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
) -> str:
    """Text-only generation (no images). Returns the raw decoded string.

    Used by Grounding (segment selection over summaries) and Reasoning
    (synthesizing multiple observations), which reason over text rather than
    image frames.
    """
    import torch

    if model is None or processor is None:
        vlm = load_model()
        model, processor = vlm.model, vlm.processor

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    _CALL_COUNTS["generate_text"] += 1
    generated = generated[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0]
