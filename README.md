# Guidebot

Guidebot 是一个以自适应 agent 为总枢纽的桌面宠物机器人项目。目标能力包括环境温度与房间
状态感知、空调调节、自然对话、触摸互动，以及未来的视觉和移动能力。

当前版本是一个可运行的自进化 Agent OS 核心原型：

- 事件总线与统一传感器/动作模型
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
pytest
```

不安装项目也可直接演示：

```bash
PYTHONPATH=src python -m guidebot.cli demo
```

详细设计见 [docs/architecture.md](docs/architecture.md) 和
[docs/self-evolution.md](docs/self-evolution.md)。

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

当前回归基线为 27 项 pytest 测试，覆盖旧 demo、路由与记忆公式、失败归因、技能谱系、房间
仿真、候选拒绝/接纳和完整自进化闭环；代码同时通过 Ruff 静态检查。

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
