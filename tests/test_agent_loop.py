import pytest

from guidebot.planning import DecisionType, MockPlanner, PlannerDecision
from guidebot.runtime import AgentLoop, RunStatus
from guidebot.runtime.demos import run_break_reminder_demo
from guidebot.safety import SafetyGate
from guidebot.tools import MockRobotTool, SpeakTool, Tool, ToolRegistry, ToolResult


def _call(tool: str, arguments: dict | None = None) -> PlannerDecision:
    return PlannerDecision(DecisionType.TOOL_CALL, "test", tool, arguments or {})


@pytest.mark.asyncio
async def test_agent_loop_stops_at_max_steps() -> None:
    registry = ToolRegistry()
    registry.register(SpeakTool())
    planner = MockPlanner((_call("speak", {"text": "a"}), _call("speak", {"text": "b"})))

    result = await AgentLoop(planner, registry, max_steps=2).run("keep speaking")

    assert result.status is RunStatus.MAX_STEPS
    assert len(result.trajectory) == 2


@pytest.mark.asyncio
async def test_planner_critic_requests_revision_for_unknown_tool() -> None:
    registry = ToolRegistry()
    registry.register(SpeakTool())
    planner = MockPlanner(
        (
            _call("unknown_tool"),
            _call("speak", {"text": "safe revision"}),
            PlannerDecision(DecisionType.FINISH, "done", final_answer="done"),
        )
    )

    result = await AgentLoop(planner, registry).run("speak safely", required_tools=("speak",))

    assert result.status is RunStatus.FINISHED
    assert planner.contexts[1].latest_observation.type == "critic_feedback"
    assert result.trajectory[0].decision.tool_name == "speak"


@pytest.mark.asyncio
async def test_safety_gate_blocks_obstacle_move_in_agent_loop() -> None:
    registry = ToolRegistry()
    registry.register(MockRobotTool("move_robot"))
    planner = MockPlanner(
        (
            _call(
                "move_robot",
                {"direction": "forward", "duration_s": 1, "speed": 0.25, "obstacle": True},
            ),
            _call(
                "move_robot",
                {"direction": "forward", "duration_s": 1, "speed": 0.25, "obstacle": True},
            ),
            _call(
                "move_robot",
                {"direction": "forward", "duration_s": 1, "speed": 0.25, "obstacle": True},
            ),
        )
    )

    result = await AgentLoop(planner, registry, max_revisions=2).run("move")

    assert result.status is RunStatus.CRITIC_REJECTED
    assert "obstacle" in (result.final_answer or "")


def test_safety_gate_rejects_invalid_move_duration() -> None:
    result = SafetyGate().evaluate_tool_call(
        "move_robot",
        {"duration_s": -1, "obstacle": False},
        known_physical_tool=True,
    )

    assert result.allowed is False


@pytest.mark.asyncio
async def test_break_reminder_workflow() -> None:
    result = await run_break_reminder_demo()

    assert result.status is RunStatus.FINISHED
    assert [step.decision.tool_name or step.decision.type.value for step in result.trajectory] == [
        "set_alarm",
        "speak",
        "health_check",
        "move_robot",
        "stop_robot",
        "speak",
        "health_check",
        "finish",
    ]
    assert result.trajectory[3].tool_result is not None
    assert result.trajectory[3].tool_result.data["obstacle"] is True
    assert result.trajectory[4].tool_result is not None
    assert result.trajectory[4].tool_result.data["stopped"] is True


class _FailingTool(Tool):
    name = "failing"
    description = "test failure"
    schema = {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(False, error="camera unavailable")


@pytest.mark.asyncio
async def test_tool_error_returns_as_next_planner_observation() -> None:
    registry = ToolRegistry()
    registry.register(_FailingTool())
    planner = MockPlanner(
        (
            _call("failing"),
            PlannerDecision(DecisionType.FINISH, "cannot continue", final_answer="failed safely"),
        )
    )

    result = await AgentLoop(planner, registry).run("inspect")

    assert result.status is RunStatus.FINISHED
    assert planner.contexts[1].latest_observation.data["error"] == "camera unavailable"
