"""Health monitor module wrapper for posture/fatigue signals."""

from __future__ import annotations

from typing import Any

from guidebot.events import Event, EventBus
from guidebot.scheduler import Task


class HealthMonitorModule:
    name = "health_monitor"

    def __init__(self) -> None:
        self.event_bus: EventBus | None = None
        self.running = False

    def start(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self.running = True

    def stop(self) -> None:
        self.running = False

    def check_once(self, *, sedentary: bool = False, fatigue: bool = False) -> Event:
        label = "normal"
        if sedentary:
            label = "sedentary"
        elif fatigue:
            label = "fatigue"
        event = Event(
            "health.detected",
            self.name,
            {"label": label, "sedentary": sedentary, "fatigue": fatigue},
            confidence=0.9,
        )
        if self.event_bus is not None:
            self.event_bus.publish(event)
        return event

    def handle_task(self, task: Task) -> dict[str, Any]:
        if task.action == "sedentary_alert":
            message = "你已经坐了一段时间，起来活动一下吧。"
        elif task.action == "fatigue_alert":
            message = "检测到可能疲劳，建议休息一下。"
        else:
            message = "健康状态已记录。"
        return {"module": self.name, "action": task.action, "message": message}
