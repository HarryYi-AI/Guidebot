from guidebot.safety import RuntimeSafetyState, SafetyGate
from guidebot.scheduler import Task
from guidebot.tooling import ToolContract, ToolPermission


def _task(target: str, action: str, payload: dict | None = None, priority: int = 10) -> Task:
    return Task("t1", target, action, payload or {}, priority, True, False)


def test_obstacle_blocks_move_forward() -> None:
    result = SafetyGate().evaluate_task(
        _task("mobility", "move_forward", {"obstacle": True}),
        RuntimeSafetyState(obstacle=True),
    )

    assert result.allowed is False
    assert "obstacle" in result.reason


def test_safety_alert_blocks_ordinary_task() -> None:
    result = SafetyGate().evaluate_task(
        _task("voice_chat", "chat", priority=10),
        RuntimeSafetyState(active_safety_alert=True),
    )

    assert result.allowed is False


def test_climate_task_is_suggestion_only_without_target() -> None:
    result = SafetyGate().evaluate_task(_task("climate_control", "suggest_comfort"))

    assert result.allowed is True


def test_physical_action_requires_explicit_confirmation() -> None:
    task = Task(
        "t2",
        "mobility",
        "move_forward",
        {"obstacle": False},
        50,
        True,
        True,
        tool_contract=ToolContract("mobility.forward", ToolPermission.PHYSICAL),
    )

    result = SafetyGate().evaluate_task(task, RuntimeSafetyState(obstacle=False))

    assert result.allowed is False
    assert "confirmation" in result.reason
