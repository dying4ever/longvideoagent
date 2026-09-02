# LongTVQA Evaluation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, resumable, GT-leakage-free LongTVQA mini-evaluation pipeline that compares Direct Qwen3-VL, Grounding+Reasoning, and the Full Agent on the same 20 validation questions while preserving normal MP4 use.

**Architecture:** Add a virtual episode frame source that maps official 3-FPS clip directories onto one continuous episode timeline, then place dataset loading, answer selection, metrics, mode orchestration, persistence, and CLI behavior under `evaluation/`. Extend existing memory/reasoning functions only through optional source parameters so current MP4 callers remain unchanged.

**Tech Stack:** Python 3.10, standard library, Pillow, existing OpenCV/ffmpeg tools, existing Qwen3-VL/transformers stack, unittest-style executable tests, JSON/JSONL.

## Global Constraints

- Use Conda environment `zjx_openvla`; do not create a new environment.
- Do not change torch, CUDA, transformers, tokenizers, Qwen weights, or model-loading behavior.
- Do not train or fine-tune any model.
- Do not add a new Agent module or duplicate three Agent implementations.
- Preserve `python demo_agent.py --video <video.mp4> --question "..."` behavior.
- Do not expose `answer`, `ts`, or `occur_clip` to any inference prompt or Agent state.
- Treat official TVQA frames as 3 FPS with one-based numeric names: `(frame_number - 1) / 3` seconds.
- Do not accept IoU results until clip-local `ts` mapping has been checked against actual downloaded frames.
- Do not run the official one-shot download script unchanged.
- Before any frame archive larger than a few tens of GB, report exact contents, size, and target path and obtain approval.
- Save all LongTVQA data under `data/longtvqa/` and all benchmark code under `evaluation/` except the shared source adapter.
- Work in an isolated Git worktree if the shared branch remains dirty; preserve unrelated `config.py`, `agents/planner.py`, and memory-file changes.

---

## File map

- Create `tools/frame_source.py`: common MP4/frame-directory source API and virtual episode timeline.
- Create `evaluation/__init__.py`: package marker.
- Create `evaluation/longtvqa_dataset.py`: annotation parsing and inference/GT separation.
- Create `evaluation/multiple_choice.py`: prompt formatting and strict `a0`-`a4` mapping.
- Create `evaluation/metrics.py`: temporal and aggregate metrics.
- Create `evaluation/instrumentation.py`: evaluation-scoped call/frame counters.
- Create `evaluation/modes.py`: three shared evaluation modes.
- Create `evaluation/result_store.py`: append-safe JSONL persistence and resume keys.
- Create `evaluation/evaluate_longtvqa.py`: CLI, subset selection, episode cache, orchestration, summaries.
- Create `evaluation/download_longtvqa_minimal.py`: metadata-first minimal annotation downloader; never downloads frames implicitly.
- Modify `memory/video_memory.py`: optional frame source for duration and sampling.
- Modify `agents/reasoning.py`: optional frame source for candidate inspection.
- Modify `agent.py`: optional frame source passed only to reasoning.
- Create `tests/test_longtvqa_dataset.py`.
- Create `tests/test_frame_source.py`.
- Create `tests/test_multiple_choice.py`.
- Create `tests/test_evaluation_metrics.py`.
- Create `tests/test_evaluation_modes.py`.
- Create `tests/test_evaluation_runner.py`.
- Modify `README.md`: benchmark commands and mini-benchmark warning.

---

### Task 1: LongTVQA annotation loader and leakage boundary

**Files:**
- Create: `evaluation/__init__.py`
- Create: `evaluation/longtvqa_dataset.py`
- Create: `tests/test_longtvqa_dataset.py`

**Interfaces:**
- Produces: `load_longtvqa(path: str) -> list[LongTVQARecord]`
- Produces: `LongTVQARecord.inference_item() -> LongTVQAItem`
- Produces: `LongTVQARecord.ground_truth -> LongTVQAGroundTruth`
- `LongTVQAItem` contains `qid`, `question`, `options`, `episode_name`, `show_name` only.

- [ ] **Step 1: Write loader tests for JSON array, JSONL, normalization, and leakage separation**

```python
def test_record_separates_inference_from_ground_truth(tmp_path):
    raw = [{
        "qid": 7, "q": "What is held?", "a0": "cup", "a1": "book",
        "a2": "pen", "a3": "phone", "a4": "plate", "answer": "a2",
        "ts": [4.0, 8.0], "episode_name": "s01e01",
        "occur_clip": "s01e01_seg01_clip_02", "show_name": "bbt",
    }]
    path = tmp_path / "val.jsonl"
    path.write_text(json.dumps(raw), encoding="utf-8")
    record = load_longtvqa(str(path))[0]
    item = dataclasses.asdict(record.inference_item())
    assert item["options"] == {"a0": "cup", "a1": "book", "a2": "pen", "a3": "phone", "a4": "plate"}
    assert not {"answer", "ts", "occur_clip"} & set(item)
    assert record.ground_truth.answer == "a2"
    assert record.ground_truth.ts == (4.0, 8.0)

def test_loader_accepts_true_jsonl(tmp_path):
    rows = [make_row(1), make_row(2)]
    path = tmp_path / "val.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
    assert [x.item.qid for x in load_longtvqa(str(path))] == [1, 2]
```

- [ ] **Step 2: Run the tests and verify the intended failure**

Run: `conda run -n zjx_openvla python tests/test_longtvqa_dataset.py`

Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.longtvqa_dataset'`.

- [ ] **Step 3: Implement immutable public and GT dataclasses plus strict parsing**

```python
@dataclass(frozen=True)
class LongTVQAItem:
    qid: int
    question: str
    options: Dict[str, str]
    episode_name: str
    show_name: str

@dataclass(frozen=True)
class LongTVQAGroundTruth:
    answer: str
    ts: Tuple[float, float]
    occur_clip: str

@dataclass(frozen=True)
class LongTVQARecord:
    item: LongTVQAItem
    ground_truth: LongTVQAGroundTruth
    def inference_item(self) -> LongTVQAItem:
        return self.item

def load_longtvqa(path: str) -> List[LongTVQARecord]:
    raw = _load_json_array_or_jsonl(path)
    return [_parse_record(row) for row in raw]
```

Validation must reject missing options, non-`a0`-`a4` answers, non-two-element spans, reversed spans, missing episode names, and missing `occur_clip`.

- [ ] **Step 4: Run loader tests**

Run: `conda run -n zjx_openvla python tests/test_longtvqa_dataset.py`

Expected: all loader and leakage tests pass.

- [ ] **Step 5: Commit**

```bash
git add evaluation/__init__.py evaluation/longtvqa_dataset.py tests/test_longtvqa_dataset.py
git commit -m "feat: add leakage-safe LongTVQA loader"
```

---

### Task 2: Virtual episode frame source

**Files:**
- Create: `tools/frame_source.py`
- Create: `tests/test_frame_source.py`

**Interfaces:**
- Produces: `VideoFileSource(video_path: str)`
- Produces: `FrameDirectorySource(frame_root: str, episode_name: str, fps: float = 3.0)`
- Produces: `.duration -> float`
- Produces: `.sample_frames(start_time=None, end_time=None, interval=2.0) -> list[VideoFrame]`
- Produces: `.uniform_sample(count: int) -> list[VideoFrame]`
- Produces: `.gt_episode_interval(occur_clip: str, ts: tuple[float, float]) -> tuple[float, float]`

- [ ] **Step 1: Write synthetic clip-directory tests**

Create clip folders out of lexical order, write small colored JPEGs named `00001.jpg`, `00002.jpg`, `00004.jpg`, and assert numeric segment/clip sorting, missing-frame time preservation, one-based timestamps, and GT conversion.

```python
def test_virtual_timeline_uses_numeric_clip_order_and_3fps(tmp_path):
    make_clip(tmp_path, "s01e01_seg02_clip_00", [1, 4])
    make_clip(tmp_path, "s01e01_seg01_clip_10", [1, 2, 3])
    source = FrameDirectorySource(str(tmp_path), "s01e01", fps=3.0)
    assert [c.name for c in source.clips] == [
        "s01e01_seg01_clip_10", "s01e01_seg02_clip_00"
    ]
    assert source.clips[0].duration == 1.0
    assert source.clips[1].offset == 1.0
    assert source.gt_episode_interval("s01e01_seg02_clip_00", (0.0, 1.0)) == (1.0, 2.0)

def test_missing_frame_number_does_not_collapse_time(tmp_path):
    make_clip(tmp_path, "s01e01_seg01_clip_00", [1, 4])
    source = FrameDirectorySource(str(tmp_path), "s01e01")
    frames = source.sample_frames(0.0, source.duration, interval=0.1)
    assert [round(f.timestamp, 3) for f in frames] == [0.0, 1.0]
```

- [ ] **Step 2: Run and verify failure**

Run: `conda run -n zjx_openvla python tests/test_frame_source.py`

Expected: FAIL because `tools.frame_source` does not exist.

- [ ] **Step 3: Implement source classes and strict parsers**

```python
CLIP_RE = re.compile(r"^(?P<prefix>.+?s\d+e\d+)_seg(?P<segment>\d+)_clip_(?P<clip>\d+)$")
FRAME_RE = re.compile(r"^(?P<number>\d+)\.(?:jpg|jpeg|png)$", re.I)

@dataclass(frozen=True)
class ClipIndex:
    name: str
    path: Path
    segment_number: int
    clip_number: int
    offset: float
    duration: float
    frames: Tuple[Tuple[int, Path], ...]

class FrameDirectorySource:
    def __init__(self, frame_root: str, episode_name: str, fps: float = 3.0):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.fps = float(fps)
        self.episode_name = episode_name
        self.clips = self._index_episode(Path(frame_root))
        self.duration = sum(c.duration for c in self.clips)
```

For a clip with highest frame number `N`, duration is `N / fps`; frame timestamp is `clip.offset + (N - 1) / fps`. Sampling chooses the first indexed frame at or after each requested time, deduplicates paths, and preserves chronological order.

- [ ] **Step 4: Run source tests and the existing video-tool tests**

Run:

```bash
conda run -n zjx_openvla python tests/test_frame_source.py
conda run -n zjx_openvla python tests/test_basic_pipeline.py
```

Expected: both suites pass.

- [ ] **Step 5: Commit**

```bash
git add tools/frame_source.py tests/test_frame_source.py
git commit -m "feat: add virtual episode frame source"
```

---

### Task 3: Source-compatible memory, reasoning, and full Agent

**Files:**
- Modify: `memory/video_memory.py`
- Modify: `agents/reasoning.py`
- Modify: `agent.py`
- Create: `tests/test_frame_source_agent_compat.py`

**Interfaces:**
- Consumes: source objects from Task 2.
- Extends: `build_video_memory(video_path, ..., source=None)`.
- Extends: `reason_over_candidates(video_path, ..., source=None)`.
- Extends: `run_agent(question, video_path, video_memory, ..., source=None)`.

- [ ] **Step 1: Write compatibility tests with a fake source**

```python
class FakeSource:
    duration = 120.0
    def sample_frames(self, start_time=None, end_time=None, interval=2.0):
        return [FakeFrame(start_time or 0.0)]

def test_memory_uses_optional_source_without_video_file(monkeypatch, tmp_path):
    stub_vlm(monkeypatch)
    memory = build_video_memory("virtual://s01e01", window_size=60,
                                output_path=str(tmp_path / "m.json"),
                                source=FakeSource())
    assert [(x["start"], x["end"]) for x in memory["segments"]] == [(0.0, 60.0), (60.0, 120.0)]
```

Also assert existing calls without `source` still invoke `video_tool` and preserve signatures.

- [ ] **Step 2: Run and verify failure**

Run: `conda run -n zjx_openvla python tests/test_frame_source_agent_compat.py`

Expected: FAIL with unexpected keyword argument `source`.

- [ ] **Step 3: Implement minimal optional-source routing**

```python
duration = source.duration if source is not None else video_tool.get_video_duration(video_path)
sample = source.sample_frames if source is not None else (
    lambda start_time=None, end_time=None, interval=2.0:
        video_tool.sample_frames(video_path, start_time, end_time, interval)
)
```

Use the routed sampler in memory and reasoning. In `agent.py`, pass `source=source` only to reasoning calls. Do not change Planner, Critic, Temporal Verifier, or model code.

- [ ] **Step 4: Run compatibility and all existing lightweight suites**

Run the new test plus `test_basic_pipeline.py`, `test_long_video.py`, `test_agent_loop.py`, `test_temporal_reasoning.py`, `test_conversation_memory.py`, `test_hierarchical_memory.py`, and `test_multiturn.py`.

Expected: all pass without loading model weights.

- [ ] **Step 5: Commit**

```bash
git add memory/video_memory.py agents/reasoning.py agent.py tests/test_frame_source_agent_compat.py
git commit -m "feat: support frame sources in agent pipeline"
```

---

### Task 4: Multiple-choice formatting and answer mapping

**Files:**
- Create: `evaluation/multiple_choice.py`
- Create: `tests/test_multiple_choice.py`

**Interfaces:**
- Produces: `format_multiple_choice_question(item: LongTVQAItem) -> str`
- Produces: `parse_choice(value: object) -> str | None`
- Produces: `select_answer_option(question, options, agent_answer, evidence, model=None, processor=None) -> dict`

- [ ] **Step 1: Write parser and leakage tests**

```python
def test_parse_choice_is_strict_but_handles_wrappers():
    assert parse_choice("a2") == "a2"
    assert parse_choice('{"choice":"a4"}') == "a4"
    assert parse_choice("The answer is A. cup") == "a0"
    assert parse_choice("option 9") is None

def test_prompt_contains_options_but_no_ground_truth(sample_item):
    text = format_multiple_choice_question(sample_item)
    assert "a0:" in text and "a4:" in text
    assert "occur_clip" not in text and "Ground Truth" not in text
```

- [ ] **Step 2: Run and verify failure**

Run: `conda run -n zjx_openvla python tests/test_multiple_choice.py`

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement deterministic parsing before text-only fallback**

`select_answer_option` first accepts an explicit `a0`-`a4` from structured/free output. Only if absent, call `vlm_tool.generate_text` with question, options, final answer, and evidence. The fallback prompt must request only:

```json
{"choice":"a0","answer_text":"...","confidence":"high|medium|low"}
```

Validate the returned key and emit `choice=None` with low confidence on failure.

- [ ] **Step 4: Run tests**

Run: `conda run -n zjx_openvla python tests/test_multiple_choice.py`

Expected: all pass with the fallback model mocked.

- [ ] **Step 5: Commit**

```bash
git add evaluation/multiple_choice.py tests/test_multiple_choice.py
git commit -m "feat: add multiple-choice answer adapter"
```

---

### Task 5: Metrics and scoped instrumentation

**Files:**
- Create: `evaluation/metrics.py`
- Create: `evaluation/instrumentation.py`
- Create: `tests/test_evaluation_metrics.py`

**Interfaces:**
- Produces: `temporal_iou(pred, gt) -> float`
- Produces: `score_intervals(predicted, gt) -> dict`
- Produces: `summarize_results(rows) -> dict`
- Produces: `EvaluationCounters` with `vlm_calls`, `llm_calls`, `grounding_calls`, `reasoning_calls`, `critic_calls`, `replans`, `frames_viewed`.

- [ ] **Step 1: Write exact metric tests**

```python
def test_temporal_iou():
    assert math.isclose(temporal_iou((0, 10), (5, 15)), 5 / 15)
    assert temporal_iou((0, 4), (5, 9)) == 0.0

def test_top1_and_best_of_k():
    score = score_intervals([(0, 5), (10, 20)], (12, 18))
    assert score["top1_iou"] == 0.0
    assert math.isclose(score["best_iou"], 0.6)
    assert score["recall_iou_03"] is True
    assert score["recall_iou_05"] is True

def test_summary_ignores_unavailable_direct_iou():
    summary = summarize_results([{"correct": True, "top1_iou": None}, {"correct": False, "top1_iou": None}])
    assert summary["accuracy"] == 0.5
    assert summary["mean_top1_iou"] is None
```

- [ ] **Step 2: Run and verify failure**

Run: `conda run -n zjx_openvla python tests/test_evaluation_metrics.py`

Expected: FAIL because metrics do not exist.

- [ ] **Step 3: Implement pure metrics and a context-local counter**

Use `contextvars.ContextVar` so counters do not leak between questions. Counting wrappers increment only while an evaluation context is active. Summaries report accuracy over all selected rows, failed count separately, and averages over rows where the field exists.

- [ ] **Step 4: Run tests**

Run: `conda run -n zjx_openvla python tests/test_evaluation_metrics.py`

Expected: all metric and counter-isolation tests pass.

- [ ] **Step 5: Commit**

```bash
git add evaluation/metrics.py evaluation/instrumentation.py tests/test_evaluation_metrics.py
git commit -m "feat: add evaluation metrics and counters"
```

---

### Task 6: Shared three-mode evaluator

**Files:**
- Create: `evaluation/modes.py`
- Create: `tests/test_evaluation_modes.py`

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: `EvaluationConfig(mode, direct_frame_count, top_k, fine_interval, max_iterations)`.
- Produces: `evaluate_item(item, source, memory, config) -> ModeResult`.

- [ ] **Step 1: Write mode-isolation tests with mocks**

```python
def test_direct_calls_no_agent_modules(monkeypatch):
    calls = fake_calls(monkeypatch)
    result = evaluate_item(item(), source(), None, EvaluationConfig(mode="direct", direct_frame_count=16))
    assert calls == {"direct_vlm": 1, "grounding": 0, "reasoning": 0, "agent": 0}
    assert result.choice in {"a0", "a1", "a2", "a3", "a4"}

def test_grounding_reasoning_disables_full_loop(monkeypatch):
    calls = fake_calls(monkeypatch)
    evaluate_item(item(), source(), memory(), EvaluationConfig(mode="grounding_reasoning"))
    assert calls["grounding"] == 1 and calls["reasoning"] == 1 and calls["agent"] == 0

def test_full_agent_uses_existing_loop(monkeypatch):
    calls = fake_calls(monkeypatch)
    evaluate_item(item(), source(), memory(), EvaluationConfig(mode="full_agent"))
    assert calls["agent"] == 1
```

- [ ] **Step 2: Run and verify failure**

Run: `conda run -n zjx_openvla python tests/test_evaluation_modes.py`

Expected: FAIL because `evaluation.modes` does not exist.

- [ ] **Step 3: Implement shared mode dispatch**

Build the question once with `format_multiple_choice_question`. `direct` uses `source.uniform_sample(count)` and `vlm_tool.analyze_frames`. `grounding_reasoning` invokes existing Grounding and source-compatible Reasoning. `full_agent` invokes `run_agent(..., source=source)`. Every path finishes through `select_answer_option` and returns intervals, evidence, status, elapsed time, and counter snapshot.

- [ ] **Step 4: Run mode tests**

Run: `conda run -n zjx_openvla python tests/test_evaluation_modes.py`

Expected: all pass with no real model load.

- [ ] **Step 5: Commit**

```bash
git add evaluation/modes.py tests/test_evaluation_modes.py
git commit -m "feat: add shared LongTVQA evaluation modes"
```

---

### Task 7: Append-safe result store and resumable runner

**Files:**
- Create: `evaluation/result_store.py`
- Create: `evaluation/evaluate_longtvqa.py`
- Create: `tests/test_evaluation_runner.py`

**Interfaces:**
- Produces: `ResultStore(path).append(row)`, `.completed_keys()`, `.rows()`.
- Produces: `select_subset(records, max_samples, show_name, episode_name, max_episodes) -> list[LongTVQARecord]`.
- Produces CLI arguments specified in the design.

- [ ] **Step 1: Write persistence, resume, and deterministic-subset tests**

```python
def test_result_store_flushes_and_resume_skips_completed(tmp_path):
    store = ResultStore(tmp_path / "results.jsonl")
    store.append({"mode": "direct", "qid": 1, "config_hash": "abc", "correct": True})
    reopened = ResultStore(tmp_path / "results.jsonl")
    assert ("direct", 1, "abc") in reopened.completed_keys()

def test_subset_prefers_few_episodes_and_is_deterministic():
    selected1 = select_subset(records(), max_samples=20, max_episodes=5)
    selected2 = select_subset(records(), max_samples=20, max_episodes=5)
    assert [x.item.qid for x in selected1] == [x.item.qid for x in selected2]
    assert len({x.item.episode_name for x in selected1}) <= 5
```

Also simulate a failure after row 3, rerun with `--resume`, and assert only rows 4-5 are evaluated.

- [ ] **Step 2: Run and verify failure**

Run: `conda run -n zjx_openvla python tests/test_evaluation_runner.py`

Expected: FAIL because result store and runner do not exist.

- [ ] **Step 3: Implement durable JSONL and config hashing**

`append` writes one compact JSON line, flushes, and calls `os.fsync`. The config hash is SHA-256 over canonical JSON containing mode, source FPS/mapping version, model id, frame budgets, memory configuration, prompt/schema versions, and subtitle flag.

- [ ] **Step 4: Implement runner orchestration**

For each selected episode, construct one `FrameDirectorySource` and load/build one memory cache. For each QA, create an inference item before entering evaluation; keep ground truth outside the mode call; after evaluation returns, convert GT interval via source, compute metrics, append the row, and regenerate `summary.json` from persisted rows.

- [ ] **Step 5: Run runner tests**

Run: `conda run -n zjx_openvla python tests/test_evaluation_runner.py`

Expected: all persistence, resume, subset, cache-reuse, and leakage tests pass.

- [ ] **Step 6: Commit**

```bash
git add evaluation/result_store.py evaluation/evaluate_longtvqa.py tests/test_evaluation_runner.py
git commit -m "feat: add resumable LongTVQA benchmark runner"
```

---

### Task 8: Metadata-first minimal data acquisition

**Files:**
- Create: `evaluation/download_longtvqa_minimal.py`
- Create: `tests/test_longtvqa_download.py`

**Interfaces:**
- Produces: CLI that downloads only named annotation files after printing size and destination.
- Never downloads frame archives; only reports their metadata.

- [ ] **Step 1: Write mocked metadata/download tests**

```python
def test_default_download_requests_only_validation_annotation(monkeypatch, tmp_path):
    calls = stub_hf_download(monkeypatch)
    main(["--data-root", str(tmp_path)])
    assert calls == ["LongTVQA_val.jsonl"]

def test_existing_matching_file_is_not_downloaded(monkeypatch, tmp_path):
    annotation = tmp_path / "annotations" / "LongTVQA_val.jsonl"
    annotation.parent.mkdir(parents=True)
    annotation.write_text("[]", encoding="utf-8")
    calls = stub_hf_download(monkeypatch)
    main(["--data-root", str(tmp_path)])
    assert calls == []
```

- [ ] **Step 2: Run and verify failure**

Run: `conda run -n zjx_openvla python tests/test_longtvqa_download.py`

Expected: FAIL because downloader does not exist.

- [ ] **Step 3: Implement explicit minimal acquisition**

Use the already installed `huggingface_hub` only if present. Otherwise print an exact `hf download` command and stop without changing dependencies. Verify the downloaded annotation is parseable before reporting success. Frame metadata lookup prints repository, filename, size, and target but requires a separate `--download-frames` flag plus explicit confirmation handled outside this script.

- [ ] **Step 4: Run tests**

Run: `conda run -n zjx_openvla python tests/test_longtvqa_download.py`

Expected: all pass without network access.

- [ ] **Step 5: Commit**

```bash
git add evaluation/download_longtvqa_minimal.py tests/test_longtvqa_download.py
git commit -m "feat: add minimal LongTVQA data preparation"
```

---

### Task 9: Full lightweight verification and documentation

**Files:**
- Modify: `README.md`
- Create: `evaluation/README.md`

**Interfaces:**
- Documents exact Step 1-6 commands, output schema, leakage boundary, and the mini-benchmark warning.

- [ ] **Step 1: Add benchmark documentation**

Document these commands exactly:

```bash
conda activate zjx_openvla
python evaluation/download_longtvqa_minimal.py --data-root data/longtvqa
python evaluation/evaluate_longtvqa.py --data-root data/longtvqa --split val --mode direct --max-samples 1 --output evaluation/results/direct.jsonl
python evaluation/evaluate_longtvqa.py --data-root data/longtvqa --split val --mode full_agent --max-samples 20 --resume --output evaluation/results/full_agent.jsonl
```

State that `answer`, `ts`, and `occur_clip` are evaluation-only and that 20 samples are not comparable with full validation results.

- [ ] **Step 2: Run every lightweight test**

Run each executable test under `tests/`, including all existing suites and the eight new benchmark suites.

Expected: every suite passes and none loads the 17.5GB model.

- [ ] **Step 3: Verify legacy MP4 CLI parsing**

Run: `conda run -n zjx_openvla python demo_agent.py --help`

Expected: exit 0 with existing `--video` and `--question` arguments.

- [ ] **Step 4: Inspect diff and run static checks**

Run:

```bash
python -m compileall evaluation tools/frame_source.py memory agents agent.py
git diff --check
git status --short
```

Expected: compile succeeds, no whitespace errors, only intended files modified.

- [ ] **Step 5: Commit**

```bash
git add README.md evaluation/README.md
git commit -m "docs: document LongTVQA mini evaluation"
```

---

### Task 10: Execute the staged mini benchmark

**Files:**
- Generate: `data/longtvqa/annotations/LongTVQA_val.jsonl`
- Generate: `data/longtvqa/frames/` only after size approval
- Generate: `data/longtvqa/cache/`
- Generate: `evaluation/results/longtvqa_subset_manifest.json`
- Generate: `evaluation/results/direct.jsonl`
- Generate: `evaluation/results/grounding_reasoning.jsonl`
- Generate: `evaluation/results/full_agent.jsonl`
- Generate: `evaluation/results/summary.json`

**Interfaces:**
- Consumes the completed CLI and official data.
- Produces actual 20-question results and case analysis.

- [ ] **Step 1: Download and validate only the 7.44MB validation annotation**

Print three records with qid, question, options, answer, ts, and episode name. Confirm JSON-array handling.

- [ ] **Step 2: Choose and save the deterministic 20-question manifest**

Select 2-5 episodes with at least 20 validation questions and varied wording (where/what/who/when/why/action). Do not select by answer or GT timestamp.

- [ ] **Step 3: Report frame download size before transfer**

Resolve the exact official archive filename and byte size. If only a monolithic show archive is available, stop and request approval with source, size, target, and expected extracted size.

- [ ] **Step 4: Validate one episode timeline**

After approved data preparation, print clip count, frame count, first/last clip, first/last timestamp, and three `occur_clip + ts` conversions. Manually inspect representative frames around those spans before enabling IoU.

- [ ] **Step 5: Run one full-Agent question**

Confirm valid `a0`-`a4`, no GT fields in captured inference input, predicted intervals in episode coordinates, and one persisted result row.

- [ ] **Step 6: Run five questions with interruption/resume**

Interrupt after at least two rows, resume, and confirm no duplicate qids and one memory build per episode.

- [ ] **Step 7: Run all three modes on the same 20 questions**

Use the same manifest and shared Qwen3-VL model. Run `direct`, `grounding_reasoning`, then `full_agent`, with `--resume` enabled.

- [ ] **Step 8: Verify summaries from raw rows**

Recompute Accuracy, mean Top-1/Best-of-K IoU, recalls, calls, time, failures, and termination counts independently from JSONL and compare with `summary.json`.

- [ ] **Step 9: Analyze cases and report**

Show at least three successes and three failures. Attribute each failure to grounding, perception, reasoning, option mapping, temporal verification, or data/timestamp mapping. Explicitly state whether scaling to 50/100 questions is justified.

- [ ] **Step 10: Commit only code/docs/manifests suitable for Git**

Do not commit model weights, frames, caches, or result files unless repository policy explicitly tracks small manifests. Verify `.gitignore` before commit.
