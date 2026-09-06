import pytest

from guidebot.planning import CriticAgent, PlannerAgent, PlannerDecision
from guidebot.runtime import ContextManager, Observation
from guidebot.tools import SpeakTool, ToolRegistry


@pytest.mark.asyncio
async def test_planner_agent_accepts_strict_json() -> None:
    planner = PlannerAgent(
        lambda _: '{"type":"tool_call","reason":"reply","tool_name":"speak",'
        '"arguments":{"text":"hi"},"final_answer":null}'
    )
    registry = ToolRegistry()
    registry.register(SpeakTool())
    context = ContextManager().build(
        goal="say hi",
        latest_observation=Observation("goal", {}),
        trajectory=(),
        registry=registry,
    )

    decision = await planner.decide(context)

    assert decision.tool_name == "speak"


def test_planner_output_rejects_markdown_instead_of_guessing_json() -> None:
    with pytest.raises(ValueError, match="strict JSON"):
        PlannerDecision.parse('```json\n{"type":"finish"}\n```')


def test_critic_rejects_invalid_tool_schema() -> None:
    registry = ToolRegistry()
    registry.register(SpeakTool())
    decision = PlannerDecision.parse(
        {
            "type": "tool_call",
            "reason": "bad arguments",
            "tool_name": "speak",
            "arguments": {"volume": 100},
            "final_answer": None,
        }
    )
    context = ContextManager().build(
        goal="speak",
        latest_observation=Observation("goal", {}),
        trajectory=(),
        registry=registry,
    )

    review = CriticAgent().review(decision, context, registry)

    assert review.approved is False
    assert any("missing required" in item for item in review.feedback)


def test_critic_rejects_finish_when_critical_step_is_missing() -> None:
    registry = ToolRegistry()
    registry.register(SpeakTool())
    context = ContextManager().build(
        goal="remind user",
        latest_observation=Observation("goal", {}),
        trajectory=(),
        registry=registry,
        required_tools=("speak",),
    )
    decision = PlannerDecision.parse(
        {"type": "finish", "reason": "too early", "final_answer": "done"}
    )

    review = CriticAgent().review(decision, context, registry)

    assert review.approved is False
    assert "missing critical tools" in review.feedback[0]
