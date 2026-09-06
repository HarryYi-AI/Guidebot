"""Deterministic interview workflows using only Mock Tools."""

from __future__ import annotations

from typing import Any

from ..planning import DecisionType, PlannerDecision
from ..tools import (
    MockAlarmTool,
    MockHealthCheckTool,
    MockRobotTool,
    SpeakTool,
    ToolRegistry,
)
from .agent_loop import AgentLoop
from .state import AgentRunResult


class BreakReminderPlanner:
    """Deterministic planner whose branches depend on the latest ToolResult."""

    def __init__(self) -> None:
        self.phase = "set_alarm"

    async def decide(self, context: Any) -> PlannerDecision:
        observation = context.latest_observation
        tool_name = observation.data.get("tool_name") if observation.type == "tool_result" else None
        data = observation.data.get("data", {}) if observation.type == "tool_result" else {}

        if self.phase == "set_alarm":
            self.phase = "alarm"
            return _call(
                "set_alarm",
                "schedule the requested reminder",
                {"minutes": 20, "message": "该休息了"},
            )
        if self.phase == "alarm" and tool_name == "set_alarm" and data.get("triggered") is True:
            self.phase = "first_speak"
            return _call(
                "speak",
                "the mock alarm has triggered",
                {"text": "已经二十分钟了，起来休息一下吧。"},
            )
        if self.phase == "first_speak" and tool_name == "speak":
            self.phase = "first_health"
            return _call(
                "health_check",
                "check whether the user followed the reminder",
                {"check": "posture"},
            )
        if self.phase == "first_health" and tool_name == "health_check":
            if data.get("standing") is True:
                return _finish("the user already stood up")
            self.phase = "move"
            return _call(
                "move_robot",
                "the user is still sitting; approach with fixed low-level control",
                {"direction": "forward", "duration_s": 1.0, "speed": 0.25, "obstacle": False},
            )
        if self.phase == "move" and tool_name == "move_robot":
            if data.get("obstacle") is True:
                self.phase = "stop"
                return _call("stop_robot", "ultrasonic feedback requires an immediate stop", {})
            self.phase = "second_speak"
            return _call("speak", "approach completed safely", {"text": "请起来活动一下吧。"})
        if self.phase == "stop" and tool_name == "stop_robot" and data.get("stopped") is True:
            self.phase = "second_speak"
            return _call(
                "speak",
                "repeat the reminder from the safe stopped position",
                {"text": "前方有障碍，我已经停下。还是起来活动一下吧。"},
            )
        if self.phase == "second_speak" and tool_name == "speak":
            self.phase = "final_health"
            return _call(
                "health_check",
                "verify the final user state",
                {"check": "posture"},
            )
        if self.phase == "final_health" and tool_name == "health_check":
            if data.get("standing") is True:
                return _finish("the second health observation shows the user standing")
            return _call(
                "speak",
                "the user is still sitting; remind without more movement",
                {"text": "我不会继续移动，但还是建议你起来休息。"},
            )
        return PlannerDecision(
            DecisionType.FINISH,
            "a required tool failed or returned an unexpected observation",
            final_answer="任务安全结束，但未确认用户起身。",
        )


def _call(tool_name: str, reason: str, arguments: dict[str, Any]) -> PlannerDecision:
    return PlannerDecision(DecisionType.TOOL_CALL, reason, tool_name, arguments)


def _finish(reason: str) -> PlannerDecision:
    return PlannerDecision(
        DecisionType.FINISH,
        reason,
        final_answer="提醒完成，用户已经起身。",
    )


async def run_break_reminder_demo() -> AgentRunResult:
    registry = ToolRegistry()
    registry.register(MockAlarmTool())
    registry.register(SpeakTool())
    registry.register(
        MockHealthCheckTool(
            [
                {"still_sitting": True, "standing": False},
                {"still_sitting": False, "standing": True},
            ]
        )
    )
    registry.register(MockRobotTool("move_robot", obstacle_on_move=True))
    registry.register(MockRobotTool("stop_robot"))
    loop = AgentLoop(BreakReminderPlanner(), registry, max_steps=10)
    return await loop.run(
        "20分钟后提醒我休息，如果我还没起来，就过来叫我。",
        required_tools=("set_alarm", "speak", "health_check"),
    )
