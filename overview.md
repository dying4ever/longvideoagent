# LongVideoAgent — 项目梳理 (overview.md)

> 最后更新：2026-09-02
> 状态：交互式长视频理解 Agent 完整实现（Agent 闭环 + 时序验证 + 多轮记忆 + OpenClaw + FastAPI + Web UI）✅

---

## 一、项目是什么

**LongVideoAgent**：一个「长视频交互式理解 Agent」。上传/指定一段长视频后，可连续多轮提问，
Agent 自主完成「定位关键画面 → 理解画面 → 校验充分性 → 不充分则继续搜索 → 复用历史事实」的闭环，
给出带时间戳证据的答案。

### 最终目标架构

```
用户问题
   ↓
Planner            （决定看哪些时间片段）
   ↓
Visual Grounding   （层次化检索定位候选片段）
   ↓
Visual Reasoning   （局部观察，重看原始帧）
   ↓
Temporal Verifier  （全局时间语义验证：first/last/repeat/always）
   ↓
Visual Critic      （规则 + 语义校验充分性）
   ↓ 证据不足则 Replan
Final Answer + 时间戳证据
```

### 参考来源

- **LongVideoAgent** — Plan → Grounding → Reasoning → Critic 的 Agent 结构
- **MemDreamer** — 长视频层次化 Memory（global → chapter → segment → event）
- **StreamAgent** — 短期 Working Memory + 持续更新上下文

### 当前阶段定位

**完整交互式 Agent 已实现**，形成：

```
Video → Hierarchical Video Memory → Conversation Session
Q1 → Context Resolver → Temporal Parser → Planner → Grounding → Reasoning
   → Temporal Verifier → Critic → Answer → 更新 Conversation/Working Memory
Q2 → 复用 Q1 实体/时间/事件 → ... → Answer
Q3 → 继续利用历史 → ... → Answer
```

CLI / OpenClaw / Web 三种入口共用同一个 `LongVideoAgentSession`，无三套独立逻辑。

尚未实现：Embedding 检索、Graph Memory、模型训练/LoRA/RL。

---

## 二、已实现功能

| 模块 | 功能 | 状态 |
|---|---|---|
| `tools/video_tool.py` | 视频时长 / 按时间抽帧(带时间戳) / 片段裁剪 | ✅ |
| `tools/vlm_tool.py` | Qwen3-VL 帧理解 + 纯文本生成 + 单例加载 + 调用计数 | ✅ |
| `memory/video_memory.py` | 层次化 Memory（global_summary + chapters + segments）| ✅ |
| `memory/conversation_memory.py` | ConversationMemory（turns/entities/confirmed/tentative events）+ WorkingMemory | ✅ |
| `memory/context_resolver.py` | 指代消解（他/她/它/那个物体，歧义不猜）| ✅ |
| `temporal/parser.py` | 时序意图分类 + target/reference_event 提取 | ✅ |
| `temporal/verifier.py` | FIRST/LAST/REPEAT/ALWAYS/BEFORE/AFTER 时间覆盖验证（纯规则）| ✅ |
| `agents/grounding.py` | 层次化候选检索（chapter → segment 剪枝 + 范围限定）| ✅ |
| `agents/reasoning.py` | 局部 occurrence/violations 检测（不报全局 first/last）| ✅ |
| `agents/planner.py` | 时序类型确定性决策 + NORMAL 走 LLM | ✅ |
| `agents/critic.py` | 规则(verifier) + LLM 语义检查 | ✅ |
| `agent_state.py` / `agent.py` | 统一状态 + 闭环 + occurrence/absence 累积 | ✅ |
| `session.py` | 多轮编排（复用已确认事实）| ✅ |
| `openclaw_adapter.py` + `openclaw/` | OpenClaw skill bundle + 桥接 | ✅（runtime 未验证）|
| `api/` | FastAPI 后端（upload/session/ask/memory/trace/reset）| ✅ |
| `frontend/` | React + Vite Web UI（视频/对话/Inspector 三区）| ✅ |

### 尚未实现

Embedding 检索、Graph Memory、模型训练/LoRA/RL、OpenClaw runtime 实测、Gradio/Streamlit（已用 React 替代）。

---

## 三、项目结构

```
/mnt/sda/zjx_space/agent/
├── model/                      # Qwen3-VL-8B-Instruct（已下载 ~17.5GB）
├── data/
│   ├── videos/                 # test.mp4(小猪佩奇285s) 等
│   └── memory/                 # 生成的 Video Memory JSON
├── tools/
│   ├── video_tool.py           # 视频读取/抽帧/裁剪
│   └── vlm_tool.py             # Qwen3-VL 帧理解 + 文本生成
├── memory/
│   ├── video_memory.py         # 层次化 Video Memory
│   ├── conversation_memory.py  # Conversation + Working Memory
│   └── context_resolver.py     # 指代消解
├── temporal/
│   ├── parser.py               # 时序意图 + target 提取
│   └── verifier.py             # 时间覆盖验证
├── agents/
│   ├── grounding.py            # Visual Grounding（层次化）
│   ├── reasoning.py            # Visual Reasoning（局部 occurrence）
│   ├── planner.py              # Planner
│   └── critic.py               # Visual Critic（规则+LLM）
├── utils/
│   └── intervals.py            # 区间 merge/subtract/covered
├── agent_state.py              # 统一状态 + trace
├── agent.py                    # Agent Loop
├── session.py                  # 多轮 Session
├── openclaw_adapter.py         # OpenClaw 桥接
├── openclaw/skills/long-video-agent/  # OpenClaw skill bundle
├── api/
│   ├── app.py                  # FastAPI 路由
│   ├── schemas.py              # Pydantic 模型
│   └── session_manager.py      # session 持久化
├── frontend/                   # React + Vite UI
├── tests/                      # 8 套测试
├── config.py                   # 统一配置 + 模型路径检测
├── demo_basic.py / demo_long_video.py / demo_agent.py / demo_chat.py
├── run.sh                      # 一键启动前后端
├── log.md / overview.md / README.md
```

---

## 四、核心模块详解

### 4.1 `tools/video_tool.py` — 视频工具（不依赖 VLM）

```python
@dataclass
class VideoFrame:
    timestamp: float      # 秒
    frame_index: int      # 源视频帧号
    image: Image.Image    # RGB PIL 图

get_video_duration(video_path) -> float
sample_frames(video_path, start_time, end_time, interval) -> List[VideoFrame]
cut_clip(video_path, start_time, end_time, output_path) -> str
```

### 4.2 `tools/vlm_tool.py` — VLM 帧理解 + 文本生成

```python
load_model(...) -> _VLM          # 单例，dtype="auto"，local_files_only=True
analyze_frames(frames, prompt, ...) -> Any   # 图像 → JSON
generate_text(prompt, ...) -> str            # 纯文本 → 字符串
_extract_json(text) -> Any                   # 鲁棒 JSON 解析
get_call_counts() / reset_call_counts()      # 调用计数（评测用）
```

### 4.3 `memory/video_memory.py` — 层次化 Video Memory（MemDreamer 风格）

```python
build_video_memory(video_path, window_size, frame_interval) -> dict
build_hierarchy(memory) -> dict   # 加 global_summary + chapters
```

```json
{
  "global_summary": "整集摘要",
  "chapters": [{"chapter_id": 0, "start": 0, "end": 120, "summary": "...", "segment_ids": [0,1]}],
  "segments": [{"segment_id": 0, "start": 0, "end": 60, "summary": "...", "events": [...]}]
}
```

层次：Global Summary → Chapter（连续 segment 聚合）→ Segment → Event。断点续建、模型只加载一次。

### 4.4 `memory/conversation_memory.py` — Conversation + Working Memory

- `turns[]`：每轮 `{question, resolved_question, answer, temporal_type, timestamp, evidence, trace}`。
- `entities{}` / `confirmed_events[]`（去重）/ `tentative_events[]`（区分 verified/tentative）。
- `working_memory{}`：`active_entities / current_subject / reference_event / recent_questions(限4) / recent_intervals(限4)`。
- `save()/load()` 持久化，`reset()` 清空（保留 video memory）。

### 4.5 `memory/context_resolver.py` — 指代消解

规则 + memory：他→男性实体/current_subject、她→女性实体、它/那个物体→最近 object、他们→active entities。
**歧义时返回 `ambiguous` + 候选，不猜**。

### 4.6 `temporal/parser.py` — 时序意图 + target 提取

`parse_temporal_query(question)` → `{type, target, reference_event}`。类型：`FIRST/LAST/REPEAT/BEFORE/AFTER/ALWAYS/NORMAL`。规则关键词优先，LLM 兜底。

`extract_target_reference(question)` → LLM 结构化提取 `{subject, gender, predicate, object}`。

### 4.7 `temporal/verifier.py` — 时间覆盖验证（核心）

`verify_temporal_condition(type, occurrences, searched, verified_absence, duration, violations)` →
`{sufficient, candidate_timestamp, missing_ranges, reason, answer}`。

- **FIRST**：候选=最早 occurrence，验证前缀 `[0,候选)` 无更早（`subtract_intervals`）。
- **LAST**：候选=最晚 occurrence，验证后缀 `(候选,duration]` 无更晚。
- **REPEAT**：`cluster_occurrences` 合并连续帧（阈值 5s），≥2 个 cluster 才成立，返回重复时间戳。
- **ALWAYS**：找反例（violations=subject 存在但违反 predicate，或 absence），反例→false，覆盖不足→insufficient。
- **BEFORE/AFTER**：reference event 定位后限定范围。
- 边界保护：候选距头/尾 < merge 阈值视为平凡满足。

### 4.8 `agents/` — Grounding / Reasoning / Planner / Critic

- **grounding**：层次化检索（先 chapter 后 segment）+ token overlap 粗筛 + 模型判断 + 范围限定。
- **reasoning**：只报**局部** occurrence/violations（禁止全局 first/last），每帧带真实时间戳。
- **planner**：时序类型走**确定性**决策（missing_ranges > 120s 用 ground_video，否则 inspect_interval）；NORMAL 走 LLM。
- **critic**：规则（verifier）优先，规则充分才走 LLM 语义检查；明确「较晚 occurrence 不与更早首次矛盾」。

### 4.9 `session.py` — 多轮 Session

```python
session = LongVideoAgentSession("video.mp4")
session.ask("乔治第一次什么时候出现？")   # FIRST → 6s，写 confirmed event
session.ask("他出现之后做了什么？")       # 指代消解 + 复用 6s reference
session.ask("他后来有没有再次出现？")     # REPEAT 复用 occurrence 种子
```

每轮：Context Resolve → Temporal Parse → 查历史 reference → run_agent → 更新 memory。

---

## 五、数据流（多轮闭环）

```
上传/指定视频
   ↓
build_video_memory() + build_hierarchy()  →  层次化 Memory（一次构建，多轮共享）
   ↓
LongVideoAgentSession.ask(question):
   1. context_resolver.resolve_question()  →  指代消解（他→乔治）
   2. temporal_parser.parse_temporal_query() → type
   3. 查 conversation memory 是否已有 reference event（复用 timestamp）
   4. agent.run_agent()  →  Planner → Grounding → Reasoning → Verifier → Critic → Replan
   5. 更新 conversation + working memory（confirmed/tentative event）
   ↓
Final Answer + Evidence + Trace
```

**解耦点**：`video_tool`↔`vlm_tool` 经 `VideoFrame` 对接；Planner/Critic 纯文本决策不看视频；Verifier 纯 Python；Memory 只定位、答案必须重看原始帧。

---

## 六、环境配置

- **Conda 环境**：`zjx_openvla`（Python 3.10.20，未新建环境）
- **GPU**：2 × RTX 5090（各 32GB），CUDA 12.8
- **核心依赖**：torch `2.10.0+cu128`、transformers `4.57.6`、tokenizers `0.22.2`、qwen-vl-utils `0.0.14`、opencv-python、Pillow、accelerate、ffmpeg
- **Web 依赖**：fastapi、uvicorn、python-multipart、httpx2（新装，不碰核心）
- **前端**：Node 20 + Vite 5 + React 18
- `config.py` 统一设 `PYTORCH_ALLOC_CONF=expandable_segments:True`（防显存碎片 OOM）

> ⚠️ transformers 升级与 `openvla 0.0.3` 的 pin 冲突；该环境原本已漂移。如需用 OpenVLA 请单独核对。

---

## 七、验证结果

### 7.1 自动化测试（8 套，均不加载 17.5GB 模型）

| 测试文件 | 覆盖 | 结果 |
|---|---|---|
| `test_basic_pipeline.py` | 视频 + VLM 纯 Python | ✅ |
| `test_long_video.py` | Memory/Grounding/Reasoning | ✅ |
| `test_agent_loop.py` | Planner/Critic/Loop | ✅ |
| `test_temporal_reasoning.py` | 时序分类/区间/验证规则（20 项）| ✅ |
| `test_conversation_memory.py` | 记忆/去重/指代（6 项）| ✅ |
| `test_hierarchical_memory.py` | 层次化/章节检索 | ✅ |
| `test_multiturn.py` | 多轮复用/reset | ✅ |
| `test_api.py` | FastAPI endpoints | ✅ |

### 7.2 真实端到端（API + 浏览器）

**API E2E**（test.mp4 小猪佩奇 285s）：

```
Q1 乔治第一次什么时候出现？ → 首次出现时间约在 6.0s (finished)
Q2 他出现之后做了什么？     → 乔治出现后…演奏乐器 (ref_ts=6.0 复用)
Q3 他后来有没有再次出现？   → 是（再次出现于 279.0s）(finished)
```

**浏览器 E2E**（Playwright + CDP 连 Chromium，真实点击）：
- ✅ 上传 test.mp4 → session 创建 → 视频时长 04:44 显示
- ✅ Q1 答案 6.0s + evidence 时间戳 chip
- ✅ **点击 evidence → 视频跳转**（`video.currentTime` 跳到 6s 并播放）
- ✅ Trace 面板 13 步完整展示（parser→planner→grounding→reasoning→verifier 78s→6s→前缀验证→critic ✓）
- ✅ Memory 面板三层（Working/Conversation/Video + global summary + 4 chapters）
- ✅ Q2 多轮「他」→「乔治」指代消解

---

## 八、关键设计决策

| 决策 | 理由 |
|---|---|
| 时序语义用**规则覆盖验证**，不用 VLM 声称 first/last | 可验证、确定性，避免局部「首次」污染全局 |
| Reasoning 只报局部 occurrence，Verifier 判全局 | 职责分离 |
| 大范围(>120s)用 ground_video，小范围用 inspect_interval | 避免整段细抽帧 OOM |
| 模型单例 + 一次构建 memory 多轮共享 | 效率 |
| 已确认事实复用（reference event / occurrence 种子）| 第二轮不重新搜索 |
| 指代消解歧义不猜 | 避免错误上下文污染 |
| confirmed vs tentative event 分离 | 防止 Memory 污染 |
| `dtype="auto"`（非 torch_dtype）| Qwen3-VL 专用参数，避免 fp32 OOM |
| CLI/OpenClaw/Web 共用 `LongVideoAgentSession` | 无三套独立逻辑 |
| `local_files_only=True` | 防联网重下模型 |

---

## 九、当前已知问题

1. **语义答案可能幻觉**：BEFORE/AFTER「做了什么」类粗粒度 reasoning 描述，个别细节（如「玩具船」）与真实剧情（礼物盒）不完全吻合。
2. **REPEAT 重复时间戳不稳定**：不同运行返回 168s/177s 或 249s/270s/282s，取决于 grounding 抽到哪些段，可能漏掉中间段的重复 occurrence。
3. **上传进度不实时**：`/sessions` 同步阻塞构建 memory（~50s），前端只显示「Preparing…」，无分段进度。
4. **OpenClaw runtime 未验证**（构建机无 openclaw CLI）。
5. **同视频重复上传不共享缓存**（video_id 变了，memory 路径不同）。
6. **ALWAYS 语义仍偏简**（violations 依赖模型自报，未严格区分「存在但不在状态」的复杂反例）。
7. **FIRST/LAST 已稳定 finished；REPEAT/ALWAYS 覆盖率低于 FIRST/LAST**。

---

## 十、未来规划

1. **Embedding 检索 + Graph Memory**（替代 token overlap 粗筛，提升 grounding 召回）
2. **OpenClaw runtime 实测**
3. **上传进度 streaming**（SSE 报 processed/total segments）
4. **REPEAT/ALWAYS 语义细化**（重复时间戳稳定性、复杂 predicate 反例）
5. **语义答案幻觉缓解**（严格约束 + 事实校验）
6. **Agent Trace 可视化增强**（Memory Hit 标记、时间轴）

---

## 十一、快速运行

```bash
conda activate zjx_openvla
cd /mnt/sda/zjx_space/agent

# 测试（无需模型，8 套）
python tests/test_basic_pipeline.py
python tests/test_temporal_reasoning.py
python tests/test_api.py
# ... 其余 tests/test_*.py

# CLI 多轮对话
python demo_chat.py --video data/videos/test.mp4

# 一键启动 Web（后端 :8123 + 前端 :5173）
./run.sh
# 浏览器打开 http://localhost:5173

# 单独启动
./run.sh backend    # 或 uvicorn api.app:app --port 8123
./run.sh frontend   # 或 cd frontend && npm run dev
```
