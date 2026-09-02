# LongVideoAgent — 项目梳理 (overview.md)

> 最后更新：2026-09-02
> 状态：完整 Agent 闭环已跑通 ✅

---

## 一、项目是什么

**LongVideoAgent**：一个「长视频交互式理解 Agent」。给定一段长视频和一个用户问题，Agent 自主完成
「定位关键画面 → 理解画面内容 → 校验答案是否充分 → 不充分则继续搜索」的闭环，最终给出带时间戳证据的答案。

### 最终目标架构

```
用户问题
   ↓
Planner            （决定看哪些时间片段）
   ↓
Visual Grounding   （定位关键帧/关键片段）
   ↓
Visual Reasoning   （理解画面，回答子问题）
   ↓
Visual Critic      （校验证据是否充分）
   ↓ 证据不足则重新规划和搜索
Final Answer
```

后续叠加：Video Memory、Conversation Memory、多轮交互、OpenClaw Skill、Agent Trace 可视化、Gradio/Streamlit Demo。

### 参考来源

- **LongVideoAgent** — Plan → Grounding → Reasoning 的 Agent 结构
- **MemDreamer** — 长视频层次化 Memory
- **StreamAgent** — 持续视频理解与记忆机制

### 当前阶段定位

**完整闭环已实现**（Memory → Grounding → Reasoning → Planner → Critic → Replan）：

```
长视频
  → 自动建立粗粒度 Video Memory
  → Planner 决策 + Grounding 定位候选时间段
  → Reasoning 细粒度分析（重新观察原始帧）
  → Critic 校验证据充分性
  → 证据不足则重新规划/搜索
  → Final Answer + 时间戳证据
```

尚未实现：Conversation Memory、多轮交互、OpenClaw Skill、Embedding 检索、Graph Memory、Web UI（Gradio/Streamlit）、模型训练/LoRA/RL。

---

## 二、已实现功能

| 模块 | 功能 | 状态 |
|---|---|---|
| `tools/video_tool.py` | 视频时长 / 按时间抽帧(带时间戳) / 片段裁剪 | ✅ |
| `tools/vlm_tool.py` | Qwen3-VL 帧理解 + 纯文本生成 + 单例加载 | ✅ |
| `memory/video_memory.py` | 长视频粗粒度 Memory（分窗摘要、断点续建） | ✅ |
| `agents/grounding.py` | 定位候选区间（粗筛 + 模型判断 + 搜索范围限定） | ✅ |
| `agents/reasoning.py` | 密集观察候选区间 → 证据 + 答案 | ✅ |
| `agents/planner.py` | 决策下一步 action（4 种） | ✅ |
| `agents/critic.py` | 判断证据充分性 + 建议搜索范围 | ✅ |
| `agent_state.py` | 统一状态 + 执行 trace | ✅ |
| `agent.py` | 完整 Agent Loop 闭环 | ✅ |

### 尚未实现

Conversation Memory、多轮交互、OpenClaw Skill、Embedding 检索、Graph Memory、Agent Trace 可视化、Gradio/Streamlit Demo。

---

## 三、项目结构

```
/mnt/sda/zjx_space/agent/
├── model/                      # Qwen3-VL-8B-Instruct（已下载，4 分片 ~17.5GB）
├── data/
│   ├── videos/                 # test.mp4(小猪佩奇285s) / real_bbb.mp4 / sample_30s.mp4
│   ├── frames/                 # 抽帧/裁剪输出（预留）
│   └── memory/                 # 生成的 Video Memory JSON
├── tools/
│   ├── video_tool.py           # 视频读取/抽帧/裁剪（不依赖 VLM）
│   └── vlm_tool.py             # Qwen3-VL 帧理解 + 文本生成（不依赖 video_tool）
├── memory/
│   └── video_memory.py         # 粗粒度 Video Memory
├── agents/
│   ├── grounding.py            # Visual Grounding
│   ├── reasoning.py            # Visual Reasoning
│   ├── planner.py              # Planner
│   └── critic.py               # Visual Critic
├── agent_state.py              # 统一状态 + trace
├── agent.py                    # Agent Loop
├── tests/
│   ├── test_basic_pipeline.py  # 基础（视频+VLM 纯 Python）
│   ├── test_long_video.py      # Memory/Grounding/Reasoning
│   └── test_agent_loop.py      # Planner/Critic/Loop
├── config.py                   # 路径 + 模型路径自动检测 + 推理参数
├── demo_basic.py               # 基础链路 Demo
├── demo_long_video.py          # Memory→Grounding→Reasoning Demo
├── demo_agent.py               # 完整闭环 Demo
├── log.md                      # 开发日志
├── overview.md                 # 本文档
└── README.md                   # 快速使用说明
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

def get_video_duration(video_path) -> float
def sample_frames(video_path, start_time=None, end_time=None, interval=2.0) -> List[VideoFrame]
def cut_clip(video_path, start_time, end_time, output_path) -> str
```

秒为单位、按需逐帧解码、参数校验 + 清晰 `VideoError`。

### 4.2 `tools/vlm_tool.py` — VLM 帧理解 + 文本生成

```python
def load_model(model_path=None, device_map="auto", dtype="auto") -> _VLM  # 单例
def build_frames_prompt(frames, question) -> str   # 显式注入时间戳
def analyze_frames(frames, prompt, model=None, processor=None) -> Any   # 图像→JSON
def generate_text(prompt, model=None, processor=None) -> str            # 纯文本→字符串
def _extract_json(text) -> Any   # 鲁棒 JSON 解析
```

关键：用 `dtype="auto"`（非 `torch_dtype`，后者弃用且会导致 fp32 OOM）；`local_files_only=True` 防联网重下。

### 4.3 `memory/video_memory.py` — 粗粒度 Video Memory

```python
def build_video_memory(video_path, window_size=60, frame_interval=10, output_path=None) -> dict
```

分窗 → 每窗抽帧 → Qwen3-VL 摘要 → JSON。结构：

```json
{
  "video_path": "...", "duration": 285.0, "window_size": 60, "frame_interval": 10,
  "segments": [
    {"segment_id": 0, "start": 0.0, "end": 60.0, "summary": "...", "events": [{"timestamp": 6.0, "description": "..."}]}
  ]
}
```

要点：模型只加载一次、逐段释放帧引用、每段落盘（断点续建）、末段 floor 防越界。

### 4.4 `agents/grounding.py` — Visual Grounding

```python
def ground_video(question, video_memory, top_k=3, search_start=None, search_end=None) -> dict
```

策略：token 重叠粗筛（长视频先缩候选，不把几百段塞给模型）→ 模型判断 → `segment_id` 映射真实时间。支持 `search_start/search_end` 限定搜索范围。

### 4.5 `agents/reasoning.py` — Visual Reasoning

```python
def reason_over_candidates(video_path, question, candidates, fine_interval=2.0) -> dict
```

逐候选密集抽帧（重看原始帧，不信 Memory 摘要）→ 每帧带真实时间戳 → 证据 + 答案。约束：不虚构、只看到子集（不把段内「首次」当全片「首次」）、不确定则明说。

### 4.6 `agents/planner.py` — Planner

```python
def plan_next_action(question, state, video_memory) -> dict
```

4 个 action：`ground_video` / `inspect_interval` / `verify_answer` / `finish`（非法 action 回退 verify_answer）。决策规则：无证据→全范围 ground；有 critic 反馈→优先该范围；证据>0→verify；不重复搜已搜区间。

### 4.7 `agents/critic.py` — Visual Critic

```python
def critique_answer(question, answer, evidence, searched_intervals, video_duration) -> dict
```

只依据 Question+Answer+Evidence+searched_intervals+duration（不看视频），检查时序问题（第一次/最后/前后/是否一直）是否查了必要范围、证据是否支撑、多候选是否冲突、是否越界推断。无法确认时优先 insufficient。输出 `sufficient/reason/missing_evidence/suggested_action/suggested_range`。

### 4.8 `agent_state.py` + `agent.py` — 状态 + 闭环

`AgentState`：`question / iteration / max_iterations / searched_intervals(去重) / grounding_history / reasoning_history / evidence(跨轮累积去重) / current_answer / critic_feedback / status / trace`。

`run_agent`：while 循环 → Planner 决策 → 执行 action → 过滤已搜区间 → 无新区间强制走 Critic → sufficient 则 finish → 最终综合（首次类问题取最早时间戳）。

---

## 五、数据流（完整闭环）

```
用户问题 Question
   │
   ▼
build_video_memory()  →  粗粒度 Memory（segment 摘要）
   │
   ▼
run_agent():
   Planner(纯文本)  →  action + search_range
   │
   ├─ ground_video:  Grounding(限定范围) → candidates
   │                 Reasoning(逐候选, 重看原始帧) → evidence + answer
   ├─ inspect_interval: Reasoning(指定区间) → evidence
   ├─ verify_answer:  Critic → sufficient? → finish : 写 feedback 回 Planner
   └─ finish: 结束
   │
   ▼
最终综合(取最早/最合适时间戳) → Final Answer + Evidence + Trace
```

**解耦点**：`video_tool` 与 `vlm_tool` 通过 `VideoFrame`（`.image`+`.timestamp`）对接；Memory/Grounding/Reasoning 复用它们；Planner/Critic 只做纯文本决策，不看视频。

---

## 六、环境配置

- **Conda 环境**：`zjx_openvla`（Python 3.10.20，未新建环境）
- **GPU**：2 × RTX 5090（各 32GB），CUDA 12.8
- **核心依赖**：torch `2.10.0+cu128`（未动）、transformers `4.57.6`（升级）、tokenizers `0.22.2`、qwen-vl-utils `0.0.14`、opencv-python `4.10.0`、Pillow `12.1.1`、accelerate `1.13.0`、ffmpeg/ffprobe `4.4.2`

> ⚠️ 升级 transformers 与 `openvla 0.0.3` 的 pin（`transformers==4.40.1`）冲突；该环境原本已漂移（torch 已 2.10）。如需继续用 OpenVLA 请单独核对。

---

## 七、验证结果

### 7.1 自动化测试（三套，均不加载 17.5GB 模型）

| 测试文件 | 项数 | 结果 |
|---|---|---|
| `test_basic_pipeline.py` | 5 | ✅ 全过 |
| `test_long_video.py` | 6 | ✅ 全过 |
| `test_agent_loop.py` | 8 | ✅ 全过 |

### 7.2 真实端到端闭环测试

`test.mp4`（小猪佩奇 285s）+「乔治第一次什么时候出现？」：

- 完整走通 Planner → Grounding → Reasoning → Critic → Replan → Answer。
- **真实发生 Critic → Replan 两次**（Critic 从证据中自主发现「6s / 78s / 168s 三处首次矛盾」）。
- **最终答案「6.0s」正确**（乔治 6 秒首次出现在草地）。

关键 Trace 节选：

```
step 1  [planner]   ground_video (全范围)
step 2  [grounding] 候选 60-120s
step 3  [reasoning] 检查 60-120s → 不确定
step 9  [critic]    sufficient=false
step 10 [planner]   inspect_interval 0-78s（按 critic 反馈）
step 13 [critic]    sufficient=false（发现 6s/78s/168s 矛盾）
—— 达 max_iterations，最终综合 → 6.0s
```

---

## 八、关键设计决策

| 决策 | 理由 |
|---|---|
| `video_tool` 与 `vlm_tool` 解耦 | 换模型后端不影响视频处理 |
| 模型路径集中配置 + 自动检测 | 不写死，便于迁移 |
| `dtype="auto"`（非 `torch_dtype`） | Qwen3-VL 专用参数，避免 fp32 OOM |
| 模型单例加载 | 内存/推理共享，进程内只加载一次 |
| 时间戳由程序写入 prompt | 不让模型猜时间 |
| Memory 粗粒度 + Reasoning 重看原始帧 | Memory 只定位，答案必须基于原始帧 |
| Grounding 粗筛 + 模型判断 | 不把几百段塞给模型 |
| Critic 不看视频，只判证据 | 快速校验充分性 |
| 过滤已搜区间 + 强制走 Critic | 避免重复搜索死循环 |
| 最终综合取最早时间戳 | 处理「首次出现」跨段矛盾 |

---

## 九、当前已知问题

1. **循环常以 `max_iterations_reached` 收尾**，而非 Critic 判 sufficient（Critic 过严，反复质疑跨段「首次」矛盾）。
2. **跨段时间推理缺陷**：不同段各自声称「首次」，产生矛盾，靠最终综合兜底纠正（本次兜底正确）。
3. **Planner 偶发重复 ground**（已用「已搜区间过滤」缓解）。
4. **综合的时序规则写死在 prompt**（只覆盖「最早/首次」，未覆盖「最后一次/是否一直」等）。

---

## 十、未来规划

1. **Conversation Memory + 多轮交互**
2. **OpenClaw Skill**
3. **Embedding 检索 + Graph Memory**（替代当前 token 重叠粗筛）
4. **Agent Trace 可视化 + Gradio/Streamlit Demo**
5. **改进 Critic 判 sufficient 的稳定性**（减少 max_iterations 收尾）

---

## 十一、快速运行

```bash
conda activate zjx_openvla
cd /mnt/sda/zjx_space/agent

# 测试（无需模型）
python tests/test_basic_pipeline.py
python tests/test_long_video.py
python tests/test_agent_loop.py

# 完整闭环 QA（需要模型）
python demo_agent.py --video data/videos/test.mp4 \
    --question "乔治第一次什么时候出现？"

# 基础链路
python demo_basic.py --video data/videos/test.mp4 --start 0 --end 10 --interval 2 \
    --question "这段视频里发生了什么？"
```
