# Guidebot Runtime Architecture

## Scope

Guidebot 的现有 Omni、VLM、YOLO、闹钟、小车运动和超声波代码保持不变，通过 Tool wrapper 接入。
新增 Runtime 只在 Python、Mock Tool 和 Replay 中验证，不代表已经整体部署到真机。

## Main loop

```text
Goal + multimodal Observation
          ↓
ContextManager ← User Memory / TaskStateTree / Token Budget
          ↓
PlannerAgent ── strict JSON ──► CriticAgent
          ▲                         │ approve/revise ≤ 2
          │                         ▼
ToolResult / Observation ◄── ToolRegistry / Skill Router
                                  │
                              SafetyGate
                                  │
                     Real Adapter | Mock | Replay
                                  │
                              Trajectory
```

`AgentLoop` 每轮只执行一个高层 Tool call，并受到 `max_steps` 限制。Tool 失败、安全拒绝和结构化结果都会
变成下一轮 Observation；Planner 返回 `finish` 后结束。Critic 负责计划质量，SafetyGate 负责最终权限，
两者不是同一层。

## Tool boundary

```text
Tool = name + description + JSON schema + async execute(**kwargs)
ToolResult = success + data + error
```

`ToolRegistry` 是唯一执行入口。Planner 不能动态导入代码、构造任意函数名或直接访问设备。真实 Tool
持有已有能力的 adapter；Mock Tool 返回确定性 observation，用于离线演示。

## Physical safety

- `obstacle=True` 禁止 `move_robot`；
- duration 必须位于固定范围；
- 未知物理动作 fail closed；
- `stop_robot` 始终允许；
- SafetyGate 不在 Planner prompt 内，也不属于可进化 Skill。

低层导航、motor PID、YOLO、ASR、VLM 和超声波循环保持固定。LLM 只选择高层 Skill/Option，不能
token-by-token 生成 PWM、轮速或 PID 参数。

## Memory and context

- WorkingMemory：由 ContextManager 持有的最近 6 步 deque，AgentLoop 每步调用 `update()`；
- EpisodicMemory：AgentLoop 默认启用内存存储，也可使用 append-only JSONL 持久化完整 episode；
- LongTermMemory：稳定信息，支持 update/supersede；
- ContextManager：旧步骤生成 deterministic digest，最近 6 步保留详细结构；
- Retrieval：关键词重合 + recency + importance，无向量数据库。

长期层由 `MemoryService` 统一管理 SQLite structured store，将 Episode、Fact、Temporary State、
Preference、Boundary、Relationship State 和 Skill Evidence 分离。Query Planner 先决定需要的记忆类型，
再重建类型化 `MemoryContext`；consolidation 只生成稳定 Preference 和未接受的 SkillCandidate。详细设计见
[memory.md](memory.md)。

## Two compatible entry points

- `guidebot.runtime.AgentLoop`：多步 Planner/Tool/Observation 验证主线；
- `guidebot.runtime.GuidebotRuntime`：保留已有 EventBus/Intent/Scheduler 常驻服务接口。

目录迁移为 Python package 后，旧导入 `from guidebot.runtime import GuidebotRuntime`、
`from guidebot.planning import FixedOptionCompiler` 和 `from guidebot.memory import MemoryStream` 继续有效。

## Replay

Replay 是 JSON observation sequence，不是物理仿真：

```text
possible_fire(0.63) → inspect again → fire(0.91) → speak alert → finish
```

## Offline self-evolution

reflection/evolution/verifier 代码保留为实验性离线路径。生产 Runtime 只使用已批准 Tool/Skill，
候选策略不能自动修改 SafetyGate 或生产代码。本阶段不做 RL、PPO、GRPO、world model、复杂仿真或
自动生产自修改。
