# Guidebot 面试架构讲解

## 一分钟介绍

> Guidebot 是一个面向物理环境的多模态 Agent Runtime。项目最初在树莓派小车上分别验证了 Omni
> 语音、VLM 场景理解、YOLO 人体状态检测、闹钟移动和超声波挡停。我没有重写这些模型和驱动，
> 而是把它们封装为带 JSON Schema 的异步 Tool。新的 AgentLoop 让 Planner 根据 Goal、Observation、
> 最近轨迹和 Memory 选择高层 Tool；Critic 在执行前检查工具存在性、参数和明显风险；所有物理动作
> 还必须经过不可由模型修改的 SafetyGate。ToolResult 会成为下一轮 Observation，直到 finish 或达到
> max_steps。整个新 Runtime 使用 Mock Tool 和 Replay 在 VSCode 中验证，没有夸大为已经整体上车。

## 真实主链

```text
multimodal input
→ RuntimeEvent / Observation
→ EventBus / Perception
→ ContextManager
→ PlannerAgent
→ CriticAgent
→ ToolRegistry / Skill Router
→ SafetyGate
→ real adapter | mock tool | replay
→ ToolResult
→ Observation
→ trajectory + memory
→ next planning step | finish
```

长期运行的传感器服务仍可使用 `GuidebotRuntime`；面试展示的多步任务使用有界 `AgentLoop`。两者共享
SafetyGate、模块边界和结构化结果，但新 AgentLoop 当前只在软件环境验证。

## 为什么 LLM 只做 high-level planner

LLM 擅长从自然语言目标和非结构化观测中做语义推理，例如选择：

```text
scene_inspect | speak | health_check | set_alarm | move_robot | stop_robot | finish
```

它不擅长提供具有硬实时保证的闭环控制。因此 Guidebot 将 LLM 输出限制为严格 JSON：tool name、
arguments 和 reason。Tool 名必须存在于 allow-list，参数必须符合 schema，执行只能经 ToolRegistry。

## 为什么不能 token-by-token 生成底层运动

电机控制通常是高频连续反馈环，而远程 LLM token 具有网络抖动、非确定性和较大的尾延迟。如果让模型
逐 token 输出轮速，会产生四类问题：

1. 无法保证实时 deadline；
2. 同一输入可能产生不同控制输出；
3. 上下文错误可能直接变成物理动作；
4. 难以证明急停、速度和障碍约束始终成立。

因此 `move_robot` 只是高层 Option。导航、motor PID 和超声波闭环保持固定；LLM 无权输出 PWM、PID
参数或绕过停止逻辑。

## Skill 为什么类似 Hierarchical RL 中的 Option

一个 Skill 可以用 Options 抽象表示：

\[
O_k=(I_k,\pi_k,\beta_k)
\]

- `I_k`：前置条件，例如移动前 obstacle 必须为 false；
- `π_k`：固定低层策略，例如导航控制器和 motor PID；
- `β_k`：终止条件，例如到达、超时或检测到障碍。

Planner 相当于 Manager，选择哪个 Option；Tool/驱动相当于 Worker，完成低层执行。当前 Manager 是
Mock/规则或可注入 LLM，并没有通过策略梯度训练，因此准确表述是 **option-style hierarchical
agent architecture**，不是“已经实现完整 Hierarchical RL”。本阶段明确不加入 PPO、GRPO 或 RL。

## SafetyGate 为什么必须与 Planner 解耦

Planner 的优化目标是完成任务，但物理系统必须满足不可违反的约束。若把安全写进 prompt，模型可能
因为上下文截断、提示注入或错误推理忽略它。Guidebot 的执行关系是：

```text
planner proposal != permission
permission = ToolRegistry allow-list ∩ JSON schema ∩ SafetyGate
```

Critic 可以尽早发现问题，但 Critic 仍不是最终安全边界。即使 Critic 错误批准，SafetyGate 仍会拒绝
`obstacle=True`、非法 duration 或未知物理动作；`stop_robot` 不需要 Planner 授权即可执行。

## Planner / Critic 为什么只保留两个 Agent

该项目需要证明角色解耦，而不是 Agent 数量。Planner 提出下一步，Critic 检查：

1. Tool 是否注册；
2. 参数是否符合 schema；
3. 是否有明显安全问题；
4. finish 是否遗漏 required tools。

Critic 最多要求两次 revise，之后终止为 `critic_rejected`。这避免“多个 Agent 无限讨论”并保持轨迹
可解释。每次审查生成 `AgentMessage(sender, receiver, message_type, content, trace_id)`。

## Memory 与 Context 如何控制长度

```text
Context = goal
        + latest observation
        + digest(old steps)
        + recent 6 detailed steps
        + relevant long-term memories
        + available tool schemas
```

WorkingMemory 用固定长度 deque；EpisodicMemory 以 JSONL append 完整 episode；LongTermMemory 保存稳定
事实，并用 `active/superseded_by` 保留变更历史。第一版检索分数为关键词相似度、新近度和 importance
的加权和，不需要向量数据库。旧步骤摘要器可替换，测试使用 deterministic summarizer。

## 最重要的 Demo 怎样讲

用户目标：

> 20分钟后提醒我休息，如果我还没起来，就过来叫我。

Mock trajectory：

```text
set_alarm → triggered
→ speak
→ health_check → still_sitting=True
→ move_robot → ultrasonic obstacle=True
→ stop_robot → stopped=True
→ speak_again
→ health_check → standing=True
→ finish
```

这条轨迹同时证明多步规划、工具结果反馈、条件分支、物理安全和任务完成判断。Mock alarm 立即触发，
所以演示不等待 20 分钟。

第二条 Replay：

```text
possible_fire(0.63)
→ information gathering: scene_inspect again
→ fire(0.91)
→ speak alert
→ finish
```

它证明 Agent 不会为了快速拿到 task reward，在低置信度视觉结果下直接报警。

## 常见追问

### 这是 ReAct 吗？

是最小 ReAct-style 闭环：Planner 产生高层 Action，环境返回 ToolResult/Observation，再进入下一轮。
没有要求模型输出长篇 chain-of-thought，日志只保留简短 reason 和结构化状态。

### Critic 和 SafetyGate 是否重复？

不重复。Critic 提高计划质量并给 Planner 修订反馈；SafetyGate 是执行权限边界，必须独立、确定性且
fail-closed。

### 为什么不用 LangGraph？

当前状态机很小，原生 Python 循环更易读、易测、依赖更少。若未来出现人工审批、长任务恢复和复杂
分支，才值得增加图编排框架。Core Runtime 不需要 LangGraph。

### 新 Runtime 是否已经部署真机？

没有。底层五项能力分别在真实小车验证；Tool Registry、AgentLoop、Memory、Critic 和 Replay 是
软件/Mock 验证。下一步真机集成工作是把已有函数注入对应 Tool wrapper，而不是重写模型。

## 代码阅读顺序

1. `src/guidebot/tools/base.py`
2. `src/guidebot/tools/registry.py`
3. `src/guidebot/planning/planner.py`
4. `src/guidebot/planning/critic.py`
5. `src/guidebot/runtime/context_manager.py`
6. `src/guidebot/runtime/agent_loop.py`
7. `src/guidebot/runtime/demos.py`
8. `src/guidebot/replay.py`

## 演示命令

```bash
guidebot demo break-reminder
guidebot replay data/replays/fire_verify.json
pytest -q
```
