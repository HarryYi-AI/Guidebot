# Guidebot

Guidebot 是一个以自适应 agent 为总枢纽的桌面宠物机器人项目。目标能力包括环境温度与房间
状态感知、空调调节、自然对话、触摸互动，以及未来的视觉和移动能力。

当前版本是一个统一 runtime + 自进化 Agent OS 核心原型。语音聊天、场景识别、健康检测、
闹钟计时和移动停止不再作为四个互相孤立的脚本运行，而是先统一产出 Event，再由规则意图分析、
调度器和安全门决定是否执行模块动作：

- 事件总线与统一传感器/动作模型
- EventBus → IntentAnalyzer → Scheduler → SafetyGate → Modules 的统一 runtime
- 语音、场景、健康、闹钟、移动、温控 mock 的轻量模块边界
- 可替换的硬件适配器和内存模拟器
- 可替换的 Agent 接口与基础温控、触摸、空气质量行为
- 位于 agent 和物理设备之间的确定性安全门
- 轨迹记录，以及受限修改、留出验证、人工发布的技能进化模型
- 分级 sigmoid Router、动态 Skill Library、结构化 Reflection
- 带相似度与时间衰减的 Memory Stream
- 基于重复失败聚类的在线技能合成与完整自进化闭环
- 因果失败归因，避免把安全拒绝、设备漏执行和传感器噪声错误学习进技能
- 带 v1/v2 版本与 parent-child lineage 的 SkillCard 生命周期
- 房间动力学仿真，以及安全零违规、held-out 分数严格提升的 Verifier Gate

## 快速开始

要求 Python 3.10+。

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
guidebot demo
guidebot simulate
guidebot evolve --dry-run
guidebot voice-demo
guidebot run
guidebot intent parse "明早七点叫我" --json
guidebot scene scan --label fire --json
guidebot health check --sedentary --json
guidebot alarm set --time "07:00" --json
guidebot schedule demo --scenario fire --json
guidebot climate status --json
pytest
```

树莓派接入阿里云 Qwen3.5 Omni Realtime：

```bash
sudo apt install -y alsa-utils
pip install -e '.[voice-qwen]'
export DASHSCOPE_API_KEY='替换为新生成的密钥'
guidebot voice-qwen --voice Tina
```

密钥只从环境变量读取，不要写入仓库。可用 `--input-device`、`--output-device` 指定 ALSA
设备；默认使用低延迟对话模式，不启用联网搜索。需要实时搜索时加 `--search`，但首句和每轮回复
都会更慢。

实机常驻运行推荐使用 `serve`，而不是手动执行一次性测试命令：

```bash
guidebot serve \
  --voice Tina \
  --input-device plughw:2,0 \
  --output-device default \
  --connect-retries 5 \
  --input-gate-rms 800 \
  --vad-threshold 0.8 \
  --vad-silence-ms 800
```

`serve` 会让语音处于唤醒等待状态，并同时运行传感器/外部脚本轮询。后台场景、健康、超声波或
闹钟事件会自动进入统一 runtime，不需要用户主动询问。

不安装项目也可直接演示：

```bash
PYTHONPATH=src python -m guidebot.cli demo
```

详细设计见 [docs/architecture.md](docs/architecture.md) 和
[docs/self-evolution.md](docs/self-evolution.md)。
语音子系统设计与小车部署说明见 [docs/voice.md](docs/voice.md)。
原厂 `09.AI_Big_Model`、`10.Basic_voice_control` 与控制模块的兼容性审计见
[docs/car-compatibility.md](docs/car-compatibility.md)。
实时 token→TTS、原生 speech-to-speech、情感语音与上车部署方案见
[docs/realtime-voice-deployment.md](docs/realtime-voice-deployment.md)。

## Unified Runtime

Guidebot 内部优先使用 CLI/Python API，而不是让外部 MCP 或 LLM 直接控制物理设备。MCP 后续只作为
外部包装层；内部链路保持可测试、低 token、低复杂度：

```text
Raw Input / Sensor / Timer / Voice
  → EventBus
  → IntentAnalyzer
  → Scheduler
  → Module Executor
  → SafetyGate
  → Device / TTS / Message
  → logs/*.jsonl
```

规则意图分析器不会依赖 LLM 决定物理动作。明火、摔倒、超声波障碍等高优先级意图可以抢占聊天、
TTS、移动和健康提醒；久坐、普通场景播报和温控建议有 cooldown，避免反复打扰。温控当前只做
状态和建议，不真实控制空调，未来可接红外或 Home Assistant adapter。

课程源码和随车示例不进入 Guidebot 仓库；树莓派已有：

```text
/home/pi/project_demo
```

本地开发机也已有：

```text
/workspace/ylj/harry_main/bot/课程程序源码/source_code/project_demo
```

这两个 `project_demo` 目录内容相同，只是树莓派和开发机路径不同。需要复用时通过模块 adapter
调用外部路径，不复制模型、图片、notebook 或厂商源码到 GitHub。

外部脚本接入约定：脚本每次运行输出一个 JSON。Guidebot 负责 while 循环、调度、抢占和安全门。

场景识别脚本输出示例：

```json
{"label": "fire", "summary": "检测到疑似明火", "confidence": 0.95}
```

健康检测脚本输出示例：

```json
{"label": "sedentary", "sedentary": true, "fatigue": false, "confidence": 0.9}
```

超声波流式脚本每行输出一个 JSON：

```json
{"obstacle": true, "distance_mm": 120}
```

Guidebot 仓库提供薄适配器样例，可复制到树莓派：

```bash
mkdir -p /home/pi/project_demo/guidebot_adapters
cp examples/raspberry_pi_adapters/*.py /home/pi/project_demo/guidebot_adapters/
chmod +x /home/pi/project_demo/guidebot_adapters/*.py
```

对应常驻命令示例：

```bash
guidebot serve \
  --voice Tina \
  --input-device plughw:2,0 \
  --output-device default \
  --scene-command 'python3 /home/pi/project_demo/guidebot_adapters/scene_once.py' \
  --scene-interval 10 \
  --health-command 'python3 /home/pi/project_demo/guidebot_adapters/health_once.py' \
  --health-interval 30 \
  --ultrasonic-stream-command 'python3 /home/pi/project_demo/guidebot_adapters/ultrasonic_stream.py'
```

## Voice Module

`src/guidebot/voice/` 提供独立、可整体替换的异步语音流水线：PCM 音频、VAD 轮次检测、STT、
Guidebot 对话适配、流式 TTS、播放器和插话取消均通过轻量 Protocol 解耦。随车的 Yahboom
离线关键词串口模块通过新 adapter 接入；原厂 `/workspace/ylj/harry_main/bot` 代码保持不变。

## Self-Evolving Core

Guidebot 将连续环境观测路由到确定性技能，并记录动作结果。失败先经过因果归因；只有技能错误
或高置信用户偏好变化可以触发候选技能生成。Memory 使用向量相似度和指数时间衰减检索经验，
SkillCard 则记录技能版本、父子谱系、验证分数与已知失败模式。

完整链路：

```text
Observation → Router → Skill → Safety → Device → Feedback
→ Failure Attribution → Reflection → Memory → Candidate → Verifier → Skill Library
```

## Simulation and Verification

轻量房间仿真支持空调延迟、湿度影响、传感器噪声与 IR 漏执行。Verifier 会在 held-out 压力
场景中比较候选技能和父技能：存在任何安全违规立即拒绝，只有评测分数严格提升才能上线。
被拒绝的候选及原因保留在 `rejected_skill_buffer`，用于后续反思而不会污染 active library。

当前回归测试覆盖旧 demo、路由与记忆公式、失败归因、技能谱系、房间仿真、候选拒绝/接纳、
完整自进化闭环、语音 runtime，以及统一 Event/Intent/Scheduler/SafetyGate/CLI；代码同时通过
Ruff 静态检查。

## 近期路线

1. 接入 ESP32-S3（温湿度、触摸）和 MQTT 消息协议。
2. 接入 Home Assistant，以实体白名单控制空调。
3. 加入流式 ASR/TTS 与可插拔 LLM planner。
4. 建立仿真场景、用户反馈评分和技能候选审批界面。
5. 加入摄像头房间检测；若采用移动底盘，再接 ROS 2/Nav2。

## 调研基线

- [Microsoft SkillOpt](https://github.com/microsoft/SkillOpt)：轨迹驱动、受限编辑、验证门控的技能优化。
- [Reachy Mini](https://www.reachymini.dev/)：桌面机器人硬件抽象、SDK 和应用分层参考。
- [Home Assistant](https://developers.home-assistant.io/)：家庭设备集成边界。

本项目不直接复制上述项目代码；第一阶段只借鉴其公开架构思想。
