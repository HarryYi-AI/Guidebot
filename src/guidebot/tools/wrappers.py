"""Thin wrappers around existing voice, VLM, YOLO, alarm, and mobility code."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from .base import Tool, ToolResult


Adapter = Callable[..., dict[str, Any] | ToolResult | Awaitable[dict[str, Any] | ToolResult]]


class AdapterTool(Tool):
    name = "adapter"
    description = "Existing capability adapter"
    schema: dict[str, Any] = {"type": "object", "properties": {}}

    def __init__(self, adapter: Adapter | None = None) -> None:
        self.adapter = adapter

    async def execute(self, **kwargs: Any) -> ToolResult:
        if self.adapter is None:
            return ToolResult(True, dict(kwargs))
        if inspect.iscoroutinefunction(self.adapter):
            value = await self.adapter(**kwargs)
        else:
            value = await asyncio.to_thread(self.adapter, **kwargs)
        if inspect.isawaitable(value):
            value = await value
        if isinstance(value, ToolResult):
            return value
        return ToolResult(True, dict(value))


class VoiceChatTool(AdapterTool):
    name = "voice_chat"
    description = "Call the existing Omni voice conversation adapter"
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }


class SceneInspectTool(AdapterTool):
    name = "scene_inspect"
    description = "Call the existing camera and VLM scene-inspection adapter"
    schema = {
        "type": "object",
        "properties": {"mode": {"type": "string"}},
    }


class HealthCheckTool(AdapterTool):
    name = "health_check"
    description = "Call the existing YOLOv8 posture/fatigue adapter"
    schema = {
        "type": "object",
        "properties": {"check": {"type": "string"}},
    }


class SetAlarmTool(AdapterTool):
    name = "set_alarm"
    description = "Schedule a reminder through the existing alarm adapter"
    schema = {
        "type": "object",
        "properties": {
            "minutes": {"type": "number", "minimum": 0},
            "message": {"type": "string"},
        },
        "required": ["minutes"],
    }


class SpeakTool(AdapterTool):
    name = "speak"
    description = "Speak text through the existing TTS/Omni audio output"
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        result = await super().execute(**kwargs)
        if result.success and "spoken" not in result.data:
            return ToolResult(True, {**result.data, "spoken": kwargs["text"]})
        return result


class MoveRobotTool(AdapterTool):
    name = "move_robot"
    description = "Move through the fixed navigation controller and motor PID"
    physical = True
    schema = {
        "type": "object",
        "properties": {
            "direction": {"type": "string"},
            "duration_s": {"type": "number", "minimum": 0.01, "maximum": 10.0},
            "speed": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "obstacle": {"type": "boolean"},
        },
        "required": ["direction", "duration_s", "speed", "obstacle"],
    }


class StopRobotTool(AdapterTool):
    name = "stop_robot"
    description = "Immediately stop the existing robot mobility adapter"
    physical = True
    schema = {"type": "object", "properties": {}}
