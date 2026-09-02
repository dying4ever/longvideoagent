# LongTVQA Evaluation Pipeline Design

Date: 2026-09-02

## 1. Objective

Add a reproducible, resumable, leakage-free LongTVQA zero-shot evaluation pipeline to the existing LongVideoAgent project. The pipeline must compare three modes on the same Qwen3-VL model and the same 20-question validation subset:

1. `direct`: fixed-budget uniform episode sampling and direct multiple-choice prediction.
2. `grounding_reasoning`: Video Memory, Grounding, and Reasoning only.
3. `full_agent`: the current Memory, Planner, Grounding, Reasoning, Temporal Verifier, Critic, and Replan loop.

The first milestone is a 20-question mini benchmark across 2-5 episodes. It is a smoke evaluation, not a paper-level result.

## 2. Non-goals

- No model training, fine-tuning, LoRA, or reinforcement learning.
- No new Agent modules.
- No torch, CUDA, transformers, tokenizer, Qwen weight, or Conda environment changes.
- No full-dataset or full-show frame download before the minimal pipeline is verified.
- No use of ground-truth answer or temporal location during question answering.
- No large changes to `agent.py`, Planner, Reasoning, or other current Agent modules.

## 3. Official LongTVQA Findings

- `LongTVQA_val.jsonl` has a `.jsonl` suffix but is stored as one JSON array. The loader will also accept true line-delimited JSONL.
- Each QA item contains `qid`, `q`, `a0`-`a4`, `answer`, `ts`, `episode_name`, `occur_clip`, and `show_name`.
- The TVQA frames are stored as numbered JPEGs under clip directories such as `s04e10_seg01_clip_03/00001.jpg`.
- Official evaluation code treats the images as 3 FPS and uses one-based names. The clip-local mapping is `clip_time = (frame_number - 1) / 3`.
- The released `ts` values behave as clip-local spans even though the paper describes episode re-indexing. The evaluation pipeline will verify this against actual frames before any IoU result is accepted.
- Official answer accuracy is exact option-key correctness: predicted `a0`-`a4` compared with the ground truth, divided by the selected sample count.
- Official LongVideoAgent uses full episode subtitles for text grounding, a symbolic localized clip, and sampled frames from that clip for visual inspection. The Master Agent consumes textual observations rather than raw images.
- The first benchmark run is vision-only. Subtitle loading is designed but disabled by default.

## 4. Architecture

### 4.1 Source abstraction

Add `tools/frame_source.py` with a small common interface:

- `VideoFileSource`: delegates to the existing `video_tool` for normal MP4 input.
- `FrameDirectorySource`: reads an episode represented by ordered clip directories containing numbered frames.

Both return the existing `VideoFrame` shape:

```python
VideoFrame(timestamp=episode_seconds, frame_index=episode_frame_index, image=image)
```

`FrameDirectorySource` may keep private metadata for debugging, including clip name, clip-local frame number, and clip-local timestamp, but downstream Agent code receives the common fields only.

### 4.2 Virtual episode timeline

For one episode:

1. Select all clip directories belonging to `episode_name`.
2. Sort them by parsed numeric `(segment_number, clip_number)`, never plain filename assumptions alone.
3. Sort frames within each clip by their numeric filename.
4. Map one-based frame number `N` to clip-local time `(N - 1) / 3` seconds.
5. Set each clip offset to the cumulative duration of preceding clips.
6. Map each frame to `episode_timestamp = clip_offset + clip_local_timestamp`.

Clip duration is derived from the official 3 FPS sequence and the highest valid frame number. Missing frame numbers do not collapse time. Duplicate or malformed names fail validation rather than silently changing time.

The scoring-only GT span is:

```text
episode_gt = clip_offset(occur_clip) + ts
```

This conversion is never exposed to the Agent.

### 4.3 Dataset model and leakage boundary

Add `evaluation/longtvqa_dataset.py` with:

- `LongTVQAItem`: public question fields used to build the Agent prompt.
- `LongTVQAGroundTruth`: `answer`, `ts`, and `occur_clip`, retained by the evaluator only.
- JSON-array and JSONL loading.
- strict validation of five options and answer keys.

The Agent-facing request contains only:

- `qid` for logging;
- question;
- five options;
- episode identifier needed to load the whole episode;
- optional episode subtitles when explicitly enabled.

It does not contain `answer`, `ts`, or `occur_clip`.

### 4.4 Multiple-choice answer adapter

Add `evaluation/multiple_choice.py`.

All modes receive the question and options before reasoning. Prompts request structured output:

```json
{
  "choice": "a2",
  "answer_text": "...",
  "confidence": "high|medium|low",
  "evidence": []
}
```

If the current Agent returns free text, a lightweight text-only adapter maps the final answer and accumulated evidence to `a0`-`a4`. It cannot inspect video or ground-truth fields.

### 4.5 Evaluation modes

One configuration object controls the mode. The modes share the same dataset loader, frame source, model instance, option format, result writer, and metrics.

#### `direct`

- Uniformly sample a fixed number of frames across the full virtual episode.
- Send frames, question, and options directly to Qwen3-VL.
- Do not use Memory, Grounding, Planner, Reasoning, Critic, Replan, or Temporal Verifier.

#### `grounding_reasoning`

- Load or build the episode Video Memory.
- Run Grounding once and Reasoning over its candidates.
- Do not call Planner, Critic, Replan, or Temporal Verifier.

#### `full_agent`

- Load or build the episode Video Memory.
- Run the current full Agent loop with minimal source-interface compatibility changes.

### 4.6 Episode memory cache

Memory is cached per episode and memory configuration, never per question answer.

The cache key includes:

- episode name;
- source fingerprint;
- 3 FPS mapping version;
- memory window size;
- memory frame interval;
- model identifier;
- prompt/schema version.

Segment and chapter summaries are persisted incrementally. A cache with a mismatched key is not reused.

## 5. Runner and persistence

Add `evaluation/evaluate_longtvqa.py` supporting:

- `--data-root`
- `--split`
- `--mode`
- `--max-samples`
- `--start-index`
- `--show-name`
- `--episode-name`
- `--output`
- `--resume`
- `--use-subtitles`

Selection order is deterministic and saved in a subset manifest containing qids, episode names, and a selection-policy version. The same manifest is used by all three modes.

After every question:

1. Append one complete JSON object to the results JSONL.
2. Flush and synchronize the file.
3. Update summary statistics from completed rows.

Resume identifies completed work by `(mode, qid, evaluation_config_hash)` and never recomputes matching rows.

## 6. Metrics and instrumentation

Add `evaluation/metrics.py`.

Recorded per question:

- prediction and exact correctness;
- GT span and predicted intervals;
- Top-1 temporal IoU;
- Best-of-K temporal IoU;
- recall indicators for IoU >= 0.3 and IoU >= 0.5;
- elapsed time;
- VLM image calls;
- LLM text calls;
- grounding calls;
- reasoning calls;
- critic calls;
- replan count;
- frames viewed;
- termination state: `finished`, `max_iterations_reached`, or `failed`.

Direct mode reports temporal metrics as unavailable unless it explicitly produces a search interval. Aggregates ignore unavailable IoU values rather than treating them as zero.

Instrumentation will use an evaluation-scoped counter wrapper around existing model/tool calls. It will not fork three Agent implementations.

## 7. Result files

```text
evaluation/results/
├── longtvqa_subset_manifest.json
├── direct.jsonl
├── grounding_reasoning.jsonl
├── full_agent.jsonl
└── summary.json
```

Each row records the question identity, prediction, correctness, intervals, temporal metrics, call counts, elapsed time, status, error details when present, and a configuration hash.

The summary compares the three modes and labels the 20-question run as `mini benchmark / smoke evaluation`.

## 8. Subtitle support

The initial run uses `--use-subtitles=false`.

The loader will support official episode-level subtitles later. When enabled, subtitles are indexed by `episode_name` and supplied as episode context. Clip-level subtitle keys must not be used to reveal `occur_clip` for the current question.

## 9. Error handling

- Missing annotation, episode, clip, or frame paths produce explicit validation errors.
- Malformed frame names, inconsistent FPS assumptions, and impossible `ts` values block temporal scoring.
- Per-question model failures are saved as `failed` rows so resume can continue safely.
- GPU out-of-memory errors are recorded and frame budgets are not silently changed mid-comparison.
- Existing valid results are never overwritten during resume.

## 10. Compatibility

Existing MP4 commands must remain functional, especially:

```bash
python demo_agent.py --video <video.mp4> --question "..."
```

The source abstraction will be introduced with adapters or optional parameters. Existing call signatures remain valid unless a minimal backward-compatible extension is necessary.

## 11. Data acquisition policy

Do not run the official one-shot script unchanged because it downloads LongTVQA, LongTVQA+, and a complete show frame archive.

Acquisition order:

1. Download only `LongTVQA_val.jsonl` (about 7.44 MB).
2. Inspect validation episode distribution and choose 2-5 episodes containing at least 20 varied questions.
3. Query and report the exact frame archive size before downloading it.
4. Prefer a source that permits selected episodes or clips. If the only official source is a monolithic show archive, report the full archive size and obtain approval before downloading.
5. Store all assets under `data/longtvqa/{annotations,frames,cache}` and skip existing verified files.

## 12. Verification sequence

1. Loader test: print and validate three annotation items.
2. Frame-source test: load one episode, verify clip order, frame order, count, and timestamp mapping.
3. Leakage test: assert Agent inputs do not contain answer, `ts`, or `occur_clip`.
4. Single-question manual run: obtain a valid `a0`-`a4` prediction.
5. Five-question run: confirm memory is built once per episode, resume works, and metrics are correct.
6. Twenty-question run in all three modes.
7. Review at least three successes and three failures using saved evidence and traces.

## 13. Acceptance criteria

- MP4 and frame-directory inputs both work.
- Existing project tests and MP4 demos remain compatible.
- The 20-question subset is deterministic and shared by all modes.
- No answer, GT timestamp, or answer-localizing clip identifier reaches Agent inference.
- Results persist after every question and resume without duplicate work.
- Episode Memory is reused across questions.
- Accuracy, IoU, recall, efficiency, and termination metrics are reproducible from saved rows.
- Three modes complete actual runs before any comparative conclusion is reported.
- Downloaded file list and disk usage are documented.

