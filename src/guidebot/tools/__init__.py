"""Typed tool wrappers for the interview-facing Agent Runtime."""

from .base import Tool, ToolResult
from .mocks import MockAlarmTool, MockHealthCheckTool, MockRobotTool, MockSceneInspectTool
from .registry import ToolRegistry, UnknownToolError
from .wrappers import (
    HealthCheckTool,
    MoveRobotTool,
    SceneInspectTool,
    SetAlarmTool,
    SpeakTool,
    StopRobotTool,
    VoiceChatTool,
)

__all__ = [
    "HealthCheckTool",
    "MockAlarmTool",
    "MockHealthCheckTool",
    "MockRobotTool",
    "MockSceneInspectTool",
    "MoveRobotTool",
    "SceneInspectTool",
    "SetAlarmTool",
    "SpeakTool",
    "StopRobotTool",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "UnknownToolError",
    "VoiceChatTool",
]
