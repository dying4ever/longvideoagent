# LongVideoAgent — 项目梳理 (overview.md)

> 最后更新：2026-09-02
> 状态：完整交互式长视频理解 Agent 已实现（P0/P1/P2 全部完成）✅

---

## 一、项目是什么

**LongVideoAgent**：长视频交互式理解 Agent。上传/指定一段长视频后，可连续多轮提问，Agent 自主完成
「自适应事件切分 → 层次化记忆 → 定位关键画面 → 理解画面 → 时序验证 → 校验充分性 → 必要时重规划」的闭环，
给出带时间戳证据的答案，并在多轮间复用已确认的事实。

### 最终架构

```
长视频
  → 自适应事件切分（观察窗口内 VLM 检测真实事件边界）
  → 层次化 Video Memory (Global → Chapter → Event → Evidence)
  → 多轮 Session
     Q → Context Resolver(指代消解) → Temporal Parser → Planner
       → Hierarchical Retrieval → Grounding → Reasoning(局部 occurrence)
       → Temporal Verifier(全局时间覆盖) → Critic(规则+LLM)
       → (证据不足则 Replan) → Answer + Evidence
       → 更新 Conversation/Working Memory
```

### 参考来源

- **LongVideoAgent** — Plan → Grounding → Reasoning → Critic 的 Agent 结构
- **MemDreamer** — 层次化 Memory + 自适应事件切分（非固定窗口）
- **StreamAgent** — 短期 Working Memory + 持续更新上下文

### 借鉴与扩展边界

| 来源 | 继承的思想 | 本工程新增或强化的部分 |
|---|---|---|
| LongVideoAgent | 多 Agent、Grounding、视觉观察、迭代推理 | 独立 Temporal Verifier、Visual Critic、可视化 Round/Replan、多轮 Session |
| StreamAgent | Working Memory、持续上下文 | 指代消解、当前主体、参考事件和已确认时间点复用 |
| MemDreamer | 层次化记忆、事件组织 | Global→Chapter→Event→Evidence 与原始帧回看、内容哈希缓存结合 |
| 本工程 | 系统组合与任务扩展 | 六类显式时序验证、OpenClaw Skill、Web/CLI/API 共用核心 Session |

本工程定位为**面向交互式长视频理解的系统扩展与工程实现**。当前未复现官方 GRPO 训练，不把多 Agent、Working Memory 或层次化记忆本身作为原创，也不声称性能超过官方模型。

---

## 二、已实现功能

| 模块 | 功能 | 状态 |
|---|---|---|
| `tools/video_tool.py` | 视频时长 / 抽帧(带时间戳) / 裁剪 | ✅ |
| `tools/vlm_tool.py` | Qwen3-VL 帧理解 + 文本生成 + 单例 + 调用计数 | ✅ |
| `memory/video_memory.py` | **自适应事件切分** + 层次化 Memory + 内容哈希缓存 | ✅ |
| `memory/conversation_memory.py` | Conversation Memory + Working Memory | ✅ |
| `memory/context_resolver.py` | 指代消解（他/她/它，歧义不猜） | ✅ |
| `temporal/parser.py` | 时序意图 + target/reference_event 提取 | ✅ |
| `temporal/verifier.py` | FIRST/LAST/REPEAT/ALWAYS/BEFORE/AFTER 时间覆盖验证 | ✅ |
| `agents/grounding.py` | 层次化检索（LLM 语义选 chapter） | ✅ |
| `agents/reasoning.py` | 局部 occurrence/violations + 接地约束 | ✅ |
| `agents/planner.py` | 时序确定性决策 + 大范围 ground 防 OOM | ✅ |
| `agents/critic.py` | 规则(verifier) + LLM 语义检查 | ✅ |
| `agent_state.py` / `agent.py` | 状态 + 闭环 + occurrence/absence 累积 | ✅ |
| `session.py` | 多轮编排 + 复用已确认事实 | ✅ |
| `openclaw_adapter.py` + `openclaw/` | OpenClaw skill（**runtime 已验证**） | ✅ |
| `api/` | FastAPI（upload/session/ask/memory/trace/reset/models/status） | ✅ |
| `frontend/` | React Web UI（四层架构 + Liquid Glass） | ✅ |
| `model_registry.py` | 模型注册表 + backend 状态 | ✅ |
| `utils/profiler.py` | 分阶段计时 | ✅ |

### 尚未实现

Embedding 检索、Graph Memory、模型训练/LoRA/RL、OpenClaw 安全/权限深度核查。

---

## 三、核心机制

### 3.1 自适应事件切分（MemDreamer 思想）

观察窗口只限制单次观察长度；VLM 在窗口内检测自适应事件边界，未结束事件 carry 到下一窗合并。
**事件边界非固定窗口**（实测 21 个事件全部非 60s 整数倍）。

### 3.2 时序覆盖验证（核心正确性）

FIRST/LAST/REPEAT/ALWAYS/BEFORE/AFTER 用纯 Python 规则验证「前缀/后缀覆盖 + 聚类 + 反例」，
不依赖 VLM 声称 first/last。这是「多次声称首次」冲突的根本解法。

### 3.3 多轮记忆复用

- Q2「他」→ 乔治（指代消解）+ 复用 Q1 的 6s reference。
- Q3 REPEAT 复用 Q1 的 occurrence 种子。
- 内容哈希缓存：同视频重上传秒开，不重建 Memory。

### 3.4 三入口共用

CLI / OpenClaw / Web API 全部调用同一个 `LongVideoAgentSession`，无三套独立逻辑。

---

## 四、验证结果

### 4.1 自动化测试（8 套，均不加载模型）

| 测试文件 | 覆盖 | 结果 |
|---|---|---|
| `test_basic_pipeline.py` | 视频 + VLM 纯 Python | ✅ |
| `test_long_video.py` | Memory/Grounding/Reasoning | ✅ |
| `test_agent_loop.py` | Planner/Critic/Loop | ✅ |
| `test_temporal_reasoning.py` | 时序分类/验证规则（21 项） | ✅ |
| `test_conversation_memory.py` | 记忆/指代 | ✅ |
| `test_hierarchical_memory.py` | 层次化/章节检索 | ✅ |
| `test_multiturn.py` | 多轮复用/reset | ✅ |
| `test_api.py` | FastAPI endpoints | ✅ |

### 4.2 真实端到端

- **三轮对话**：FIRST→6s、AFTER 复用 6s、REPEAT→224s，均 finished。
- **浏览器 E2E**：上传 → 三轮问答 → 点 evidence 跳转 → trace/memory 可视化 → 模型选择器。
- **OpenClaw**：INSTALLED / DISCOVERED / RUNTIME VERIFIED / MULTITURN VERIFIED 四项全过。
- **性能**：首建 Memory 67s（事件切分主导）→ Q1 21.6s → Q2 9.6s → Q3 14.0s（多轮复用提速）。

---

## 五、已知问题（如实）

1. **事件切分数随 VLM 非确定性波动**（17/21/22 次），未做 Stage-1 shot detection 锚定边界。
2. **段级 Grounding 仍 token overlap**（chapter 级已 LLM 化），长视频语义近义召回需 Benchmark 验证后定 embedding。
3. **语义答案偶发推断**（「似乎正在拿出玩具」），已通过 evidence 派生 + 接地约束缓解但未根除。
4. **OpenClaw 用 DeepSeek 远程 API 作 control model**（本机仅 Qwen3-VL 做视觉），联网/权限未深度核查。
5. 前端未做 headless 浏览器自动化视觉回归（仅 build + 快照验证）。

---

## 六、未来规划

1. Embedding 检索 + Graph Memory（待 Benchmark 证明 recall 不足后加）
2. Stage-1 shot detection 锚定事件边界（提升切分稳定性）
3. 前端上传进度 streaming（SSE）
4. OpenClaw 安全/权限核查
5. 合并 `integration-ui` 回 `main`

---

## 七、快速运行

见 `README.md`「使用方法」章节。
