"""Climate control mock module.

No real AC/HVAC command is sent here. The module only reports status and emits
comfort suggestions; future infrared or Home Assistant adapters can implement
the same module boundary.
"""

from __future__ import annotations

from typing import Any

from guidebot.events import Event, EventBus
from guidebot.scheduler import Task


class ClimateControlModule:
    name = "climate_control"

    def __init__(self, temperature_c: float = 25.0, humidity: float = 50.0) -> None:
        self.event_bus: EventBus | None = None
        self.running = False
        self.temperature_c = temperature_c
        self.humidity = humidity

    def start(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self.running = True

    def stop(self) -> None:
        self.running = False

    def status(self) -> dict[str, Any]:
        return {
            "temperature_c": self.temperature_c,
            "humidity": self.humidity,
            "adapter": "mock",
            "real_control_enabled": False,
        }

    def comfort_event(self, text: str) -> Event:
        event = Event("user.text", self.name, {"text": text})
        if self.event_bus is not None:
            self.event_bus.publish(event)
        return event

    def handle_task(self, task: Task) -> dict[str, Any]:
        if task.action == "suggest_comfort":
            message = "我先不直接控制空调，建议根据体感把温度调到舒适区间。"
        elif task.action == "ac_left_on_alert":
            message = "检测到空调可能长时间开启，建议确认是否需要关闭。"
        else:
            message = "温控状态已记录。"
        return {
            "module": self.name,
            "action": task.action,
            "message": message,
            "real_control_enabled": False,
        }
