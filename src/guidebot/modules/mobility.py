"""Mobility module wrapper.

Physical motion adapters should live behind this module and always pass through
SafetyGate. The default implementation only reports intended actions.
"""

from __future__ import annotations

from typing import Any

from guidebot.events import Event, EventBus
from guidebot.scheduler import Task


class MobilityModule:
    name = "mobility"

    def __init__(self) -> None:
        self.event_bus: EventBus | None = None
        self.running = False
        self.stopped = True

    def start(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self.running = True

    def stop(self) -> None:
        self.running = False
        self.stopped = True

    def obstacle_detected(self, *, distance_mm: int | None = None) -> Event:
        event = Event(
            "ultrasonic.obstacle",
            self.name,
            {"obstacle": True, "distance_mm": distance_mm},
            confidence=1.0,
            priority_hint=100,
        )
        if self.event_bus is not None:
            self.event_bus.publish(event)
        return event

    def handle_task(self, task: Task) -> dict[str, Any]:
        if task.action == "stop":
            self.stopped = True
            message = "已停止小车移动。"
        elif task.action == "move_forward":
            self.stopped = False
            message = "准备前进。"
        else:
            message = "移动任务已记录。"
        return {"module": self.name, "action": task.action, "message": message}
