from guidebot.safety import RuntimeSafetyState, SafetyGate
from guidebot.scheduler import Task


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
