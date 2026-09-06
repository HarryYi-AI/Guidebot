from guidebot.memory import Episode, EpisodicMemory, LongTermMemory, MemoryRetriever, WorkingMemory
from guidebot.planning import AgentMessage, CriticReview, DecisionType, PlannerDecision
from guidebot.runtime import AgentLoopStep, ContextManager, Observation
from guidebot.safety import SafetyResult
from guidebot.tools import ToolResult


def test_working_memory_is_a_sliding_window() -> None:
    memory = WorkingMemory(capacity=3)
    memory.extend(range(5))

    assert memory.items() == (2, 3, 4)


def test_episodic_memory_appends_jsonl(tmp_path) -> None:
    memory = EpisodicMemory(tmp_path / "episodes.jsonl")
    memory.append(Episode("task", (), "finished", True))
    memory.append(Episode("other", (), "failed", False))

    records = memory.read_all()
    assert len(records) == 2
    assert records[0]["task"] == "task"
    assert records[1]["success"] is False


def test_long_term_memory_supersedes_without_deleting_history(tmp_path) -> None:
    memory = LongTermMemory(tmp_path / "long_term.json")
    first = memory.add("用户喜欢第一次提醒", importance=0.7)
    second = memory.supersede(first.memory_id, "用户通常需要第二次提醒")

    assert memory.get(first.memory_id).active is False
    assert memory.get(first.memory_id).superseded_by == second.memory_id
    assert memory.list_active() == (second,)
    assert LongTermMemory(tmp_path / "long_term.json").list_active()[0].content == second.content


def test_memory_retrieval_uses_keyword_and_importance() -> None:
    memory = LongTermMemory()
    expected = memory.add("用户久坐后需要再次提醒", importance=0.9)
    memory.add("用户喜欢听音乐", importance=0.4)

    retrieved = MemoryRetriever(memory).retrieve("久坐提醒", limit=1)

    assert retrieved[0].memory.memory_id == expected.memory_id


def _step(index: int) -> AgentLoopStep:
    decision = PlannerDecision(DecisionType.TOOL_CALL, "test", "speak", {"text": str(index)})
    critic = CriticReview(True, (), AgentMessage("critic", "planner", "approve", "ok", "t"))
    return AgentLoopStep(
        index,
        decision,
        critic,
        SafetyResult(True, "allowed"),
        ToolResult(True, {"spoken": str(index)}),
        Observation("tool_result", {"index": index}),
        1.0,
    )


def test_context_manager_digests_old_steps_and_keeps_recent_six() -> None:
    from guidebot.tools import SpeakTool, ToolRegistry

    registry = ToolRegistry()
    registry.register(SpeakTool())
    trajectory = [_step(index) for index in range(8)]

    context = ContextManager(recent_steps=6).build(
        goal="test",
        latest_observation=trajectory[-1].observation,
        trajectory=trajectory,
        registry=registry,
    )

    assert len(context.recent_steps) == 6
    assert context.reasoning_digest == "0:speak:True | 1:speak:True"
