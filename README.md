# LongVideoAgent

长视频交互式理解 Agent。完整闭环 + 时序验证 + 多轮记忆 + OpenClaw Skill + Web UI 已实现：

```
长视频
  → 自适应事件切分 → 层次化 Video Memory (Global→Chapter→Event→Evidence)
  → 多轮 Session (Conversation/Working Memory + 指代消解 + 时序验证)
  → Planner → Grounding → Reasoning → Temporal Verifier → Critic → (Replan) → Answer
```

## 与参考工作的关系及本项目扩展

本项目是面向**多轮交互式长视频理解**的系统扩展，不将参考论文中的已有方法表述为原创。三项参考工作分别提供 Agent 循环、流式工作记忆和层次化视频记忆思路，本项目的贡献在于把它们组合为一个可运行、可验证、可部署的闭环系统。

| 参考工作 | 借鉴内容 | 本项目的具体扩展 |
|---|---|---|
| [LongVideoAgent](https://github.com/longvideoagent/LongVideoAgent) | Master 调度 Grounding/Vision 的多轮推理范式 | 拆分 Planner、Visual Grounding、Visual Reasoning、Temporal Verifier、Visual Critic；增加多轮 Session 和显式 Replan 轨迹 |
| StreamAgent | 持续更新上下文与短期 Working Memory | 增加人物指代消解、当前主体、参考事件和已确认时间点复用 |
| MemDreamer | 层次化视频记忆和事件组织 | 构建 Global → Chapter → Event → Evidence，并与原始帧回看和内容哈希缓存结合 |
| **本项目系统扩展** | — | FIRST/LAST/REPEAT/BEFORE/AFTER/ALWAYS 显式时序验证；OpenClaw Skill；Web UI 中展示解析结果、Round/Agent/Replan 和时间证据 |

### 贡献边界

- 多 Agent 推理框架、Working Memory 和层次化记忆分别来源于上述参考工作；本项目的重点是面向交互任务的组合、扩展与工程落地。
- 当前没有复现官方 GRPO 强化学习训练，也不声称准确率超过官方模型。
- OpenClaw 负责 Skill 发现、意图路由和适配器调用；内部视觉闭环由同一个 `LongVideoAgentSession` 执行。
- Web UI 直接调用 FastAPI；CLI 与 OpenClaw Skill 通过 `openclaw_adapter.py` 复用相同核心 Session。

---

## 目录结构

```
├── model/                  # Qwen3-VL-8B-Instruct（已下载）
├── data/
│   ├── videos/             # 输入视频
│   └── memory/             # Video Memory + 会话状态（内容哈希命名，自动缓存）
├── tools/
│   ├── video_tool.py       # 视频读取/抽帧(带时间戳)/裁剪
│   └── vlm_tool.py         # Qwen3-VL 帧理解 + 文本生成（单例）
├── memory/
│   ├── video_memory.py     # 自适应事件切分 + 层次化 Memory
│   ├── conversation_memory.py  # Conversation + Working Memory
│   └── context_resolver.py     # 指代消解
├── temporal/
│   ├── parser.py           # 时序意图 + target/reference_event 提取
│   └── verifier.py         # FIRST/LAST/REPEAT/ALWAYS/BEFORE/AFTER 时间覆盖验证
├── agents/
│   ├── grounding.py        # Visual Grounding（层次化检索）
│   ├── reasoning.py        # Visual Reasoning（局部 occurrence）
│   ├── planner.py          # Planner
│   └── critic.py           # Visual Critic（规则+LLM）
├── utils/
│   ├── intervals.py        # 区间工具
│   └── profiler.py         # 分阶段计时
├── agent_state.py          # 统一状态 + trace
├── agent.py                # Agent Loop 闭环
├── session.py              # 多轮 Session
├── openclaw_adapter.py     # OpenClaw 桥接（CLI）
├── openclaw/skills/        # OpenClaw skill bundle
├── api/                    # FastAPI backend
├── frontend/               # React + Vite Web UI
├── model_registry.py       # 模型注册表 + backend 状态
├── tests/                  # 8 套轻量测试
├── config.py               # 路径 + 模型路径检测 + 参数
├── run.sh                  # 一键启动前后端
├── demo_basic.py / demo_long_video.py / demo_agent.py / demo_chat.py / demo_profile.py
├── log.md / overview.md / README.md
```

---

## 环境

- Conda 环境：`zjx_openvla`（Python 3.10, torch 2.10.0+cu128, CUDA 12.8）
- GPU：2 × RTX 5090（各 32GB）
- 前端：Node 22（nvm）+ Vite

```bash
conda activate zjx_openvla
cd /mnt/sda/zjx_space/agent
```

模型路径：自动检测 `<项目根>/model/`，或用 `export LONGVIDEO_MODEL_PATH=/path/to/model` 覆盖。

---

## 使用方法

### 1. Web UI（推荐，交互式）

```bash
./run.sh          # 一键启动后端(:8123) + 前端(:5173)
# 浏览器打开 http://localhost:5173
```

流程：上传视频 → 构建/加载 Memory → 多轮提问（支持「他/她」指代消解、first/last/repeat/always 时序问答）→ 点击 Evidence 时间戳跳转视频 → 查看 Agent Process 管道 / Memory / Manage。

> 后端首次加载模型 + 构建 Memory 约 60s；同视频重上传会自动复用缓存（秒开）。

### 2. CLI 多轮对话

```bash
python demo_chat.py --video data/videos/test.mp4
# 命令: /memory(看记忆) /trace(看上轮trace) /reset /quit
```

### 3. CLI 单次问答（OpenClaw adapter，跨调用复用会话）

```bash
python openclaw_adapter.py --video data/videos/test.mp4 --question "乔治第一次什么时候出现？"
python openclaw_adapter.py --video data/videos/test.mp4 --question "他出现之后做了什么？"
```

### 4. 完整闭环 Demo（单轮）

```bash
python demo_agent.py --video data/videos/test.mp4 --question "乔治第一次什么时候出现？"
```

### 5. 性能 Profiling

```bash
python demo_profile.py --video data/videos/test.mp4
# 输出分阶段耗时（model_load/event_segmentation/grounding/reasoning/critic）+ VLM 调用数 + 帧数
```

### 6. 测试（无需模型，8 套）

```bash
for t in tests/test_*.py; do python "$t"; done
```

### 7. OpenClaw Skill

```bash
source /home/ps/.nvm/nvm.sh
nvm use 22.23.2
cd openclaw-cli
./node_modules/.bin/openclaw skills list          # 确认 long-video-agent 被发现
./node_modules/.bin/openclaw agent --local -m "分析视频 /mnt/sda/zjx_space/agent/data/videos/test.mp4：乔治第一次什么时候出现？"
```

> OpenClaw 通过 skill → `openclaw_adapter.py` CLI → `LongVideoAgentSession` → Qwen3-VL 运行。
> OpenClaw 2026.8.2 需 Node >= 22.22.3；本机已安装 Node 22.23.2。控制模型使用 DeepSeek API（`DEEPSEEK_API_KEY`），视觉理解使用本地 Qwen3-VL。

---

## 支持的能力

| 能力 | 说明 |
|---|---|
| 时序问答 | 第一次/最后一次/再次/之前/之后/一直（`Temporal Verifier` 规则覆盖验证） |
| 多轮交互 | 「他/她/它」指代消解 + 复用已确认事实/时间戳/occurrence |
| 自适应事件切分 | 观察窗口内 VLM 检测自适应事件边界（非固定窗口） |
| 层次化 Memory | Global → Chapter → Event → Evidence |
| 模型选择 | Qwen3-VL-8B Local（available）；Qwen/OpenAI/Gemini API（未配置） |
| Agent Trace | Planner→Grounding→Reasoning→Verifier→Critic 完整决策管道 |
| 缓存 | 内容哈希 Memory，同视频重上传复用 |

---

## 依赖

Python：torch / transformers / accelerate / qwen-vl-utils / opencv-python / Pillow / fastapi / uvicorn。
前端：React 18 + Vite 5（Node 22）。

> 版本说明：Qwen3-VL 需 `transformers >= 4.51`，当前 `zjx_openvla` 已升到 `4.57.6`。
> openvla 0.0.3 原本 pin `transformers==4.40.1`，如需继续用 OpenVLA 请单独核对环境。
