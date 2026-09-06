"""Deterministic tools for VSCode demos and unit tests."""

from __future__ import annotations

from collections import deque
from typing import Any

from .base import Tool, ToolResult
from .wrappers import HealthCheckTool, SetAlarmTool


class MockSceneInspectTool(Tool):
    name = "scene_inspect"
    description = "Return the next replayed scene observation"
    schema = {"type": "object", "properties": {"mode": {"type": "string"}}}

    def __init__(self, observations: list[dict[str, Any]]) -> None:
        self.observations = deque(dict(item) for item in observations)

    async def execute(self, **kwargs: Any) -> ToolResult:
        if not self.observations:
            return ToolResult(False, error="scene replay exhausted")
        return ToolResult(True, self.observations.popleft())


class MockHealthCheckTool(HealthCheckTool):
    def __init__(self, observations: list[dict[str, Any]]) -> None:
        super().__init__()
        self.observations = deque(dict(item) for item in observations)

    async def execute(self, **kwargs: Any) -> ToolResult:
        if not self.observations:
            return ToolResult(False, error="health replay exhausted")
        return ToolResult(True, self.observations.popleft())


class MockAlarmTool(SetAlarmTool):
    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(
            True,
            {
                "alarm_id": "mock-alarm-1",
                "minutes": kwargs["minutes"],
                "triggered": True,
            },
        )


class MockRobotTool(Tool):
    description = "Deterministic mock for move_robot or stop_robot"
    physical = True

    def __init__(self, operation: str, *, obstacle_on_move: bool = False) -> None:
        if operation not in {"move_robot", "stop_robot"}:
            raise ValueError("mock robot operation must be move_robot or stop_robot")
        self.name = operation
        self.obstacle_on_move = obstacle_on_move
        self.schema = (
            {
                "type": "object",
                "properties": {
                    "direction": {"type": "string"},
                    "duration_s": {"type": "number", "minimum": 0.01, "maximum": 10.0},
                    "speed": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "obstacle": {"type": "boolean"},
                },
                "required": ["direction", "duration_s", "speed", "obstacle"],
            }
            if operation == "move_robot"
            else {"type": "object", "properties": {}}
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        if self.name == "stop_robot":
            return ToolResult(True, {"stopped": True})
        return ToolResult(
            True,
            {
                "moving": not self.obstacle_on_move,
                "obstacle": self.obstacle_on_move,
                "distance_mm": 120 if self.obstacle_on_move else None,
            },
        )
