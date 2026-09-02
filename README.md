# LongVideoAgent

长视频交互式理解 Agent。完整闭环已实现：

```
长视频 → Video Memory → Planner → Grounding → Reasoning → Critic → (证据不足则 Replan) → Final Answer
```

## 目录结构

```
├── model/                  # Qwen3-VL-8B-Instruct 模型（已下载）
├── data/
│   ├── videos/             # 输入视频（test.mp4 等）
│   └── memory/             # 生成的 Video Memory JSON
├── tools/
│   ├── video_tool.py       # 视频读取/抽帧/裁剪（不依赖 VLM）
│   └── vlm_tool.py         # Qwen3-VL 帧理解 + 文本生成（不依赖 video_tool）
├── memory/
│   └── video_memory.py     # 粗粒度 Video Memory
├── agents/
│   ├── grounding.py        # Visual Grounding
│   ├── reasoning.py        # Visual Reasoning
│   ├── planner.py          # Planner
│   └── critic.py           # Visual Critic
├── agent_state.py          # 统一状态 + trace
├── agent.py                # Agent Loop 闭环
├── tests/                  # 三套轻量测试
├── config.py               # 路径 + 模型路径自动检测 + 推理参数
├── demo_basic.py           # 基础链路 Demo
├── demo_long_video.py      # Memory→Grounding→Reasoning Demo
├── demo_agent.py           # 完整闭环 Demo
├── log.md                  # 开发日志
├── overview.md             # 项目梳理
└── README.md
```

## 环境

使用现有 Conda 环境 `zjx_openvla`（Python 3.10, torch 2.10.0+cu128, CUDA 12.8）。

```bash
conda activate zjx_openvla
```

## 模型路径

- 自动检测 `<项目根>/model/`。
- 也可用环境变量覆盖：`export LONGVIDEO_MODEL_PATH=/path/to/model`。

## 运行

```bash
conda activate zjx_openvla
cd /mnt/sda/zjx_space/agent

# 测试（无需模型，三套）
python tests/test_basic_pipeline.py
python tests/test_long_video.py
python tests/test_agent_loop.py

# 完整闭环 QA（需要模型）
python demo_agent.py --video data/videos/test.mp4 \
    --question "乔治第一次什么时候出现？"

# Memory→Grounding→Reasoning（无闭环）
python demo_long_video.py --video data/videos/test.mp4 \
    --question "这段视频里发生了什么？"

# 基础链路
python demo_basic.py --video data/videos/test.mp4 --start 0 --end 10 --interval 2 \
    --question "这段视频里发生了什么？"
```

## 依赖

torch / transformers / accelerate / qwen-vl-utils / opencv-python / Pillow。

> 版本说明：Qwen3-VL 需要 `transformers >= 4.51`。当前 `zjx_openvla` 已升级到
> `transformers 4.57.6` + `tokenizers 0.22.2`（torch 2.10.0+cu128 保持不变）。
> 注意 openvla 0.0.3 原本 pin `transformers==4.40.1`，升级后如需继续用 OpenVLA
> 请单独核对其运行环境。
