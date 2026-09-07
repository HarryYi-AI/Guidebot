# Hierarchical Temporal Memory

Guidebot 的 Memory 是面向物理环境的结构化时间记忆，不是把全部对话写进向量数据库后做 top-k。

## 数据流

```text
RuntimeEvent
→ EpisodeMemory
→ conservative candidate extraction
→ temporal conflict resolution
→ SQLite structured store
→ query plan
→ typed retrieval
→ MemoryContext
→ AgentLoop
→ outcome episode
→ batch consolidation
→ Preference / SkillCandidate
```

`MemoryService` 是应用层唯一入口。`AgentLoop` 和 `ContextManager` 不直接访问 SQLite。

## 数据表

| SQLite table | 类型 | 时间语义 |
|---|---|---|
| `episodes` | 完整事件、观察、动作与结果 | append-only archival memory |
| `facts` | 原子 key/value 事实 | `valid_from` / `valid_to` |
| `states` | tired、anxious、feels_cold 等临时状态 | `expires_at` TTL |
| `preferences` | 经重复证据聚合的稳定偏好 | active/historical 版本链 |
| `boundaries` | 不可被普通 Agent 覆盖的约束 | system-only write |
| `relationship_states` | 稳定关系状态 | active/historical 版本链 |
| `skill_evidence` | Skill 成功/失败证据 | append-only archival memory |
| `skill_candidates` | consolidation 生成的候选流程 | candidate only，不自动生效 |

各表都保存 `memory_id`、`user_id`、`status`、结构化 JSON payload，并为 `status`、`memory_key`、
`timestamp`、`valid_from/valid_to` 建立索引。

## 冲突与权限

`TemporalResolver` 只输出 `ADD / UPDATE / HISTORIZE / IGNORE`。值改变时不会删除旧记录，而会把旧值
标为 `historical`，并令 `old.valid_to = new.valid_from`。单次用户消息只能生成 Episode、TemporaryState
或 Evidence，不能直接写稳定 Preference/Relationship。Preference/Relationship 仅允许 system 或
consolidator 写入；Boundary 仅允许 system 写入，默认 `editable_by_agent=false`。

## 检索

`RuleBasedMemoryQueryPlanner` 先把 task 转为 `MemoryQueryPlan`，明确需要哪些 fact key、当前状态、近期
episode、boundary 和 skill evidence。`MemoryContextBuilder` 再返回类型化 `MemoryContext`，而不是无边界
拼接文本。

归档 Episode 排序为：

```text
score = 0.6 * semantic_relevance + 0.2 * recency + 0.2 * importance
```

默认 `MockEmbeddingBackend` 使用确定性词法相似度，离线可测；`EmbeddingBackend` Protocol 可接入其他
实现，核心代码不绑定模型供应商。

## Consolidation

`MemoryConsolidator` 是显式批处理，不在每条消息后修改 profile。当前支持：

- 重复夜间温控正向反馈聚合为 Preference candidate；
- 同窗口负向反馈形成 contradiction 证据；
- SkillEvidence 聚合为 `SkillCandidate`；
- TemporaryState 永不升级为稳定 profile。

SkillCandidate 始终 `accepted=false`，必须经过现有 verifier/eval gate 才能进入正式 Skill Library。

## API

```python
service.record_event(event, user_id="user-1", action={}, outcome={})
context = service.get_context("user-1", "今晚又有点热")
service.update_memory(memory, actor="system")
report = service.run_consolidation("user-1")
service.list_active_memories(PreferenceMemory, "user-1")
service.list_history(PreferenceMemory, "user-1")
```

离线演示：

```bash
python -m guidebot.memory.demo
```

首版暂不包含真实 embedding API、向量数据库、自动 profile 推断、跨用户共享记忆或自动 Skill 激活。
