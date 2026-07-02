# Guidebot 自进化核心引擎

## 1. 闭环与安全边界

引擎把一次交互实现为两个阶段：

```text
Observation → Router → Skill → Safety → Device → Feedback
    → Failure Attribution → Reflection → Memory → Candidate Skill
    → Verifier → Skill Library
```

`SelfEvolvingAgent.decide` 执行闭环前半段，`GuidebotHub` 执行安全检查和设备动作，随后
`SelfEvolvingAgent.observe_outcome` 完成归因、反思、记忆和进化。`feedback_provider` 可以接入
真实设备反馈；未配置时使用动作是否被安全门接受作为即时反馈。

SafetyPolicy 不属于可进化状态。技能增长不能扩大设备白名单、温控范围或运动速度上限。

## 2. Observation 与 Router

Guidebot 的连续观测定义为：

```text
o_t = [T, H, τ, u_t] ∈ R⁴
φ(o_t) = [(T-22)/10, (H-50)/50, clip(τ), clip(u_t)]
S_k(o_t) = sigmoid(w_kᵀ φ(o_t))
π_route(o_t) = argmax_k S_k(o_t)
```

`Observation` 保存连续向量与离散事件上下文；`FeatureMapper` 负责归一化；
`HierarchicalRouter` 先按 level 处理高优先级技能，再在同层选择最高分。每个路由结果都保留
所有候选分数，便于审计和离线训练。

## 3. Skill Library

技能库保存：

```text
L_t = {(name, f_skill, precondition, effects, w_k, level)}
f_skill: (Observation, RobotState) → Decision
```

`SkillLibrary.add` 对应 `L_{t+1}=L_t∪{f_new}`。策略函数和前置条件均为显式可注入接口，
不依赖 Router、Memory 或设备实现。

## 4. Reflection

`ReflectionEngine` 计算：

```text
c_t = R(o_t, a_t, r_t, o_{t+1})
c_t = [causal_factor, failure_mode, severity, suggested_action]
```

实现还输出 confidence 与二元熵 `H(p)=-p log p-(1-p)log(1-p)`。安全拒绝、环境负奖励和
温控恶化会产生结构化失败模式；成功行为产生 `failure_mode=none`。

## 5. Memory

`MemoryStream` 存储完整 `Experience`，并按下式检索：

```text
score(q,m_i) = cosine(φ(q), φ(m_i)) · exp(-λΔt)
```

它支持全部记忆检索、仅失败检索及指定 `failure_mode` 过滤。默认半衰减量级为一天，
部署时可以按家庭互动频率调整 λ，或替换为持久化向量数据库。

## 6. Failure Attribution

在修改技能前，`FailureAttributor` 计算：

```text
z_t = A(τ_t, safety_t, device_t, Δo_t, user_t)
z_t ∈ {skill_error, execution_lapse, sensor_noise, delayed_effect,
       user_preference_shift, safety_rejection, unknown}
evolve(z_t) = 1[z_t=skill_error]
            ∨ 1[z_t=user_preference_shift ∧ confidence≥0.8]
```

设备漏执行、传感器噪声、延迟效应与安全拒绝会保存在 Memory 中供诊断，但不会触发技能重写。
这阻止系统把物理执行问题错误归因为策略问题。

## 7. Policy Evolution 与 Skill Lifecycle

`FailurePatternClusterer` 按结构化 failure mode 聚类，并计算每簇的特征质心。当某一失败簇
数量达到阈值时：

```text
if |cluster({m ∈ M_fail | evolve(attribution(m))})| ≥ threshold:
    f_new = synthesizer(cluster)
    candidate = SkillCard(f_new, parent=current_skill, version=v+1)
```

`SkillCard` 保存版本、parent-child lineage、动作 schema、安全范围、成功/失败计数、验证分数和
已知失败模式。旧 `Skill` 会自动获得 v1 Card，新候选从父 Card 生成 v2、v3 等版本。

## 8. Simulation 与 Verifier Gate

房间仿真采用一阶动力学：

```text
T_{t+1} = T_t + κ_a(T_ambient-T_t)
          + κ_h·h(H)·(T_target-T_t) + ε_t
```

模型显式覆盖 AC 冷热延迟、湿度效率、传感器噪声和 IR 指令漏执行。评测报告包含 comfort
error、recovery steps、unsafe action count、safety rejection count、skill reuse rate 和
evolution accept rate。

候选技能的上线规则为：

```text
ΔJ = J_sim(candidate; held-out) - J_sim(parent; held-out)
accept(candidate) ⇔ safety_violations = 0 ∧ ΔJ > 0
```

Verifier 自动生成压力场景并运行 `SimulationSuite`。不安全候选无条件拒绝；没有严格提升的候选
同样拒绝。拒绝项及原因进入 `rejected_skill_buffer`，只有通过验证的 Card 才能加入 active
Skill Library。

默认 `RecoverySkillSynthesizer` 只生成受现有动作类型约束的恢复技能。后续 LLM synthesizer
必须实现同一接口，且生成动作仍会经过 SafetyPolicy。技能文本的离线优化与留出集门控继续由
已有 `EvolutionEngine` 承担。

## 9. 不可进化安全不变量

SafetyPolicy 不作为 Observation、Memory、SkillCard 或候选补丁的一部分。任何候选只能提出
Action，不能修改动作白名单、HVAC 的 16–30°C 范围或运动速度上限。Verifier 是上线前测试门，
运行时 SafetyPolicy 仍会对每一次动作重新检查；两层检查都不能被技能绕过。

## 10. 扩展接口

- 替换 `FeatureMapper`：学习型 embedding 或多模态特征。
- 替换 `ReflectionEngine`：LLM Reflexion，并保持结构化 `Critique` 输出。
- 替换 `MemoryStream`：SQLite、向量数据库或分层长期记忆。
- 实现 `SkillSynthesizer`：代码生成、技能组合或人机协同审批。
- 替换 `FailureAttributor`：概率因果模型，同时保持 evolution allow-list。
- 扩展 `SimulationSuite`：真实设备数字孪生或 ROS 2/Gazebo 场景。
- 替换 `VerifierAgent`：更大 held-out 集或多目标 Pareto 验证。
- 配置 `GuidebotHub.feedback_provider`：真实 HVAC 状态、用户反馈或仿真奖励。
