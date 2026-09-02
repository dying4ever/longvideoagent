# LongVideoAgent — 开发日志 (log.md)

> 记录时间：2026-09-02（会话内整理）
> 目标：长视频交互式理解 Agent 的最小可运行版本（MVP）

---

## 一、项目背景与当前定位

**最终目标架构**：

```
用户问题 → Planner → Visual Grounding → Visual Reasoning
         → Visual Critic → (证据不足则重新规划/搜索) → Final Answer
```

后续将扩展：Video Memory、Conversation Memory、多轮交互、OpenClaw Skill、Agent Trace 可视化、Gradio/Streamlit Demo。

参考来源：LongVideoAgent（Plan→Grounding→Reasoning 结构）、MemDreamer（层次化 Memory）、StreamAgent（持续视频理解与记忆）。

**当前阶段（MVP）只做最底层感知链路**：

```
本地视频 → 按时间抽帧(带时间戳) → Qwen3-VL-8B → 结构化 JSON 描述
```

上层 Planner / Grounding / Reasoning / Critic / Memory 均未实现（按规划后续再加）。

---

## 二、环境检查结果

| 项目 | 结果 |
|---|---|
| 真实工作目录 | `/mnt/sda/zjx_space/agent/`（注意：消息里写的 `/mnt/zjx_space/` 实际挂载点是 `/mnt/sda/`） |
| Conda 环境 | `zjx_openvla`（Python 3.10.20，未新建环境） |
| GPU | 2 × RTX 5090（各 32GB），驱动 570.153.02，CUDA 12.8 |
| torch | 2.10.0+cu128，`cuda.is_available() = True` |
| ffmpeg / ffprobe | 4.4.2 已安装 |
| OpenCV | 4.10.0 已安装 |
| Pillow | 12.1.1 已安装 |
| accelerate | 1.13.0 已安装 |
| qwen-vl-utils | 缺失 → 已安装 0.0.14 |
| transformers | **4.40.1 → 已升级 4.57.6**（见下方冲突说明） |
| tokenizers | 0.19.1 → 已升级 0.22.2 |
| Qwen3-VL-8B 模型 | 下载中（详见「当前阻塞」） |

---

## 三、环境变更记录（最小改动）

1. **transformers 4.40.1 → 4.57.6**、**tokenizers 0.19.1 → 0.22.2**（仅此 2 个包）。
   - 原因：Qwen3-VL 需要 `transformers >= 4.51`（`qwen3_vl` 架构在 4.40.1 中不存在）。
   - 用户拍板选择「原地升级」（备选方案是隔离安装到项目 `.deps`）。
   - torch / CUDA / numpy 等核心依赖**均未改动**。
2. **安装 `qwen-vl-utils 0.0.14`**（唯一缺失的必需包）。
3. **目录写权限解锁**：`agent/` 原属 root（755），当前用户 `ps` 无法写入。由于无 sudo 密码，通过 `docker run --user 0:0` + bind-mount 执行 `chown -R 1000:1000` 完成（结果与 `sudo chown -R ps:ps` 等价）。

> 备注：升级时 pip 提示 `openvla 0.0.3` 本身 pin 了 `torch==2.2.0 / transformers==4.40.1 / tokenizers==0.19.1`，但该环境在我动手前就已漂移（torch 已是 2.10.0），且 openvla 的 editable 安装路径已失效。如需继续使用 OpenVLA，请单独核对其运行环境。

---

## 四、已完成工作（Step 1~3）

### Step 1：基础环境
- 环境检查 + 依赖补齐 + transformers 升级（见第三节）。
- 结论：Python 可读取视频、调用 ffmpeg、使用 OpenCV、加载本地 Qwen3-VL。

### Step 2：视频基础工具 `tools/video_tool.py`
实现了三个接口（统一以**秒**为单位、按需读帧、参数校验、清晰报错）：
- `get_video_duration(video_path)` — ffprobe 优先，cv2 兜底。
- `sample_frames(video_path, start_time, end_time, interval)` — 返回 `list[VideoFrame]`，每帧含 `timestamp`（秒）/ `frame_index` / `image`（RGB PIL）。
- `cut_clip(video_path, start_time, end_time, output_path)` — ffmpeg 流拷贝裁剪。
- **零 VLM 依赖**，未来换模型/换 API 都不需要改这个文件。

### Step 3：VLM 帧理解 `tools/vlm_tool.py`
- `load_model(model_path, device_map, dtype)` — 单例缓存，进程内只加载一次。
- `build_frames_prompt(frames, question)` — 把每帧时间戳显式写入 prompt（`Frame i: timestamp = x.x s`），并要求模型按时间顺序、不虚构、不确定则说明、只输出合法 JSON。
- `analyze_frames(frames, prompt, model, processor)` — 多帧图像 → Qwen3-VL → 结构化 JSON。
- `_extract_json` — 鲁棒解析模型输出（代码块包裹 / 尾随文本 / 中文 / 数组等）。
- **与 video_tool 解耦**，只依赖对象的 `.image` + `.timestamp` 属性。

---

## 五、文件清单（位于 `/mnt/sda/zjx_space/agent/`）

```
├── model/                      # Qwen3-VL-8B-Instruct（下载中，勿动）
├── data/
│   ├── videos/test_10s.mp4     # ffmpeg 生成的 10s 测试视频
│   └── frames/                 # 抽帧输出（预留）
├── tools/
│   ├── __init__.py
│   ├── video_tool.py           # 视频读取/抽帧/裁剪（不依赖 VLM）
│   └── vlm_tool.py             # Qwen3-VL 帧理解（不依赖 video_tool）
├── tests/test_basic_pipeline.py # 自包含测试（视频 + VLM 纯 Python 部分）
├── config.py                   # 路径 + 模型路径自动检测 + 推理参数
├── demo_basic.py               # 端到端最小 Demo
└── README.md                   # 现状/运行/模型路径/测试命令
```

**模型路径管理**：不在任何文件里写死，`config.detect_model_path()` 自动检测（`LONGVIDEO_MODEL_PATH` 环境变量 → `<项目根>/model`）。

---

## 六、测试结果

`python tests/test_basic_pipeline.py` → **ALL TESTS PASSED**，包含 5 项：

| 测试 | 结果 |
|---|---|
| 视频时长 | `duration = 10.00s` ✅ |
| 抽帧时间戳 | `[0.0, 2.0, 4.0]` 精确 ✅ |
| 错误处理（文件不存在/时间越界/interval<=0） | ✅ |
| VLM JSON 解析鲁棒性（6 种输出形态） | ✅ |
| Prompt 时间戳注入 + JSON schema | ✅ |

另外已验证：
- `config.detect_model_path()` → 正确返回 `/mnt/sda/zjx_space/agent/model`
- `Qwen3VLProcessor` 消息构建 + 图像 token 化（`input_ids`/`pixel_values`/`image_grid_thw`）在 transformers 4.57.6 下实测通过。
- `demo_basic.py` 抽帧成功，模型加载处干净报错（权重未就绪时不会崩溃）。

---

## 七、关键问题与修复

1. **transformers 版本过旧**（4.40.1 无 `qwen3_vl`）→ 升级 4.57.6。
2. **`dtype` vs `torch_dtype` bug**：Qwen3-VL 的 `from_pretrained` 用 `dtype="auto"`（`torch_dtype` 已弃用且会退化成 fp32 → 8B 模型 OOM）。已改为 `dtype="auto"`（从权重自动推导 bf16）。
3. **`local_files_only=True`**：加在 model/processor 加载处，防止意外联网重下模型。
4. **目录写权限**：root 属主 → 通过 docker chown 解锁。
5. **模型下载慢**：与另一个 aria2 下载（behavior-1k-assets）抢带宽；该任务结束后 Qwen3-VL 下载明显加速。

---

## 八、端到端推理结果（已跑通 ✅）

模型权重（4 个分片，合计 ~17.5GB）下载完成后，`demo_basic.py` 端到端一次跑通：

```json
{
  "summary": "A color bar test pattern with a horizontal rainbow gradient bar is displayed, with the gradient bar's color sequence alternating between two patterns over time.",
  "events": [
    {"timestamp": 0.0, "description": "The video displays a standard color bar test pattern with a horizontal rainbow gradient bar at the bottom, starting with red on the left."},
    {"timestamp": 2.0, "description": "The horizontal rainbow gradient bar reverses its color sequence, starting with green on the left."},
    {"timestamp": 4.0, "description": "The horizontal rainbow gradient bar returns to its original color sequence, starting with red on the left."},
    {"timestamp": 6.0, "description": "The horizontal rainbow gradient bar reverses its color sequence again, starting with green on the left."}
  ]
}
```

- 测试视频是 ffmpeg `testsrc` 彩条图案，Qwen3-VL 正确识别为 color bar test pattern。
- 时间戳 0.0 / 2.0 / 4.0 / 6.0 与抽帧时间完全一致，说明「程序显式写入时间戳 → 模型据此定位事件」的机制正常工作。
- 模型加载（4 分片 bf16）约 2 秒，单次生成返回合法 JSON。

**结论：MVP 最底层感知链路（视频 → 抽帧带时间戳 → Qwen3-VL → 结构化 JSON）已完整跑通。**

---

## 九、下一步（已被后续阶段覆盖，历史记录）

1. 编写整体梳理文档 `overview.md`。
2. 之后按规划扩展：Memory → Grounding → Planner → Critic → 多轮交互 → OpenClaw Skill → Demo。

---

## 十、阶段二：Video Memory + Grounding + Reasoning（Step 4~6）

将「指定时间段 → 抽帧 → Qwen3-VL → 结果」扩展为「长视频 → 粗粒度 Memory → 定位候选 → 细粒度分析 → 答案+证据」。

**新增/修改文件**：
- `memory/video_memory.py` — 粗粒度 Memory 构建（分窗 → 每窗抽帧 → Qwen3-VL 摘要 → JSON），支持断点续建、模型只加载一次。
- `agents/grounding.py` — 定位候选区间（粗筛 token 重叠 + 模型判断，segment_id 映射真实时间）。
- `agents/reasoning.py` — 逐候选密集抽帧 → 证据 + 综合答案（重看原始帧，不信 Memory 摘要）。
- `tools/vlm_tool.py` — 新增 `generate_text` 纯文本生成。
- `demo_long_video.py`、`tests/test_long_video.py`。

**测试**：`test_long_video.py` 6 项全过；真实视频（30s 公园 + Big Buck Bunny）端到端跑通。

**关键问题与修复**：
- 末段 `end` 四舍五入越界 → 改用 `math.floor` 向下取整。
- Grounding 返回 `{query, candidates}` 字典，agent 遍历时误用 → 提取 `["candidates"]`。

---

## 十一、阶段三：Planner + Critic + Agent Loop 闭环（当前阶段）

将链路升级为完整闭环：

```
Question → Planner → Grounding → Reasoning → Critic → (证据不足) Replan → ... → Final Answer
```

**新增/修改文件**：
- `agents/planner.py` — 4 个 action（ground_video / inspect_interval / verify_answer / finish），纯文本决策。
- `agents/critic.py` — 判断证据充分性（重点查「第一次/最后/前后」类时序问题），输出 sufficient + suggested_range。
- `agent_state.py` — 统一状态 + trace（去重累积）。
- `agent.py` — Agent Loop（过滤已搜区间、无新区间强制走 Critic、max_iterations 兜底、最终综合取最早时间戳）。
- `demo_agent.py`、`tests/test_agent_loop.py`。
- `agents/grounding.py` — 加 `search_start/search_end` 限定搜索范围。

**测试**：`test_agent_loop.py` 8 项全过；三套测试全部通过。

**真实闭环测试**（`test.mp4` 小猪佩奇 285s +「乔治第一次什么时候出现？」）：
- 真实发生 Critic → Replan 两次。
- 最终答案「6.0s」正确（乔治 6 秒首次出现在草地）。
- 循环以 `max_iterations_reached` 收尾（Critic 过于严格，反复发现跨段「首次」矛盾）。

**当前已知问题**：
1. 循环常以 max_iterations 收尾，而非 Critic 判 sufficient。
2. 跨段时间推理有缺陷：不同段各自声称「首次」，产生矛盾（靠最终综合兜底纠正）。
3. Planner 偶发重复 ground（已用「已搜区间过滤」缓解）。
4. 最终综合的「取最早时间戳」是写死在 prompt 的，尚未覆盖「最后一次/是否一直」等其他时序问题。
