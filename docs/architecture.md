# Guidebot 架构草案

## 核心原则

Guidebot 采用边缘优先、事件驱动、硬件可替换的结构。自进化 agent 是决策总枢纽，但不是
安全边界。所有物理动作都必须通过确定性的 Safety Policy，技能文本不能修改权限、温控硬限制
或运动限速。

```text
传感器/语音/视觉
       │ Reading / Event
       ▼
  Event Bus ─────► Robot State / 轨迹记录
       │
       ▼
 Intent Analyzer
       │ Intent
       ▼
 RuntimeSkillRegistry
       │ Task(module.action)
       ▼
  Safety Policy ──拒绝──► 审计事件
       │ 允许
       ▼
 Module Executor ─────► Device Adapter / TTS / Message
       │
       ▼
 ESP32 / Home Assistant / ROS 2 / 模拟器 / 外部课程源码 adapter
```

在线 runtime 是确定性的：`IntentAnalyzer` 先用规则把事件映射为意图，`RuntimeSkillRegistry`
再把意图解析为可执行模块能力，例如 `scene.fire_alert`、`health.sedentary`、`alarm.remind`、
`climate.comfort`。温控被视为环境场景检测的一类：温湿度、空调状态和人在/离开状态进入
`climate.detected`，异常时只生成建议或告警，真实红外/Home Assistant 控制必须另接 adapter 并继续
经过 SafetyGate。这样后续新增“听音乐、闹钟、健康监测、宠物互动、空调”等功能时，只需要新增
模块或外部 adapter，并在 `src/guidebot/runtime_skills.py` 注册 skill；LLM 不直接决定物理动作。

可选 agent 角色放在 `src/guidebot/agents/`：

- `EmbodiedPlannerAgent`：面向 LLM/云端 planner，只输出结构化动作提案；
- `SkillEvolutionAgent`：面向执行后的学习闭环，只做失败归因、反思和候选技能演化。

旧的桌面宠物自进化 Skill Library 仍负责 Observation → Skill 的学习闭环；RuntimeSkillRegistry
负责常驻服务中的 Intent → Task 调度。两者名字都叫 skill，但层级不同：前者用于策略进化，后者
用于工程调度。

## 自进化闭环

参考 SkillOpt，但将线上执行与离线学习分离：

1. 记录输入、工具调用、动作、用户反馈与任务评分。
2. 在离线批次中区分成功和失败轨迹并反思。
3. 优化器只提出少量 add/delete/replace 技能文本修改。
4. 在留出场景和仿真器中做回归验证；只有严格提升才进入候选版本。
5. 涉及物理行为的候选必须经人工批准后发布；安全策略永不进入可进化域。

## 推荐硬件边界

- ESP32-S3：触摸、温湿度、灯光、舵机等实时 I/O；通过 MQTT/串口上报。
- Raspberry Pi 5 或小型主机：语音、视觉、agent runtime、本地存储。
- Home Assistant：空调和家庭设备接入；使用最小权限实体白名单。
- ROS 2：只有在加入移动底盘、SLAM 或复杂运动控制后再引入，MVP 不强依赖。

## 接下来接口

- `DeviceAdapter`：真实 ESP32/MQTT 和 Home Assistant adapter。
- `Agent`：云端或本地 LLM planner，输出结构化 `Decision`。
- `SkillOptimizer` / `SkillEvaluator`：轨迹反思、仿真评测和技能候选。
- 本地 API：状态查询、事件流、候选技能审批与紧急停止。
- `RuntimeSkill`：新增可调度功能的统一注册点，保持模块和调度器低耦合。
