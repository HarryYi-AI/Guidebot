"""Scene monitor module wrapper.

The heavy camera/VLM implementation stays outside this repository. This module
normalizes its result into Guidebot events and handles scheduled alert tasks.
"""

from __future__ import annotations

from typing import Any

from guidebot.events import Event, EventBus
from guidebot.scheduler import Task


class SceneMonitorModule:
    name = "scene_monitor"

    def __init__(self) -> None:
        self.event_bus: EventBus | None = None
        self.running = False

    def start(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self.running = True

    def stop(self) -> None:
        self.running = False

    def scan_once(
        self,
        *,
        label: str = "normal",
        summary: str = "未发现明显异常。",
        confidence: float = 0.9,
    ) -> Event:
        event = Event(
            "scene.detected",
            self.name,
            {"label": label, "summary": summary},
            confidence=confidence,
        )
        if self.event_bus is not None:
            self.event_bus.publish(event)
        return event

    def handle_task(self, task: Task) -> dict[str, Any]:
        if task.action == "fire_alert":
            message = "检测到疑似明火或烟雾，请立即检查环境。"
        elif task.action == "fall_alert":
            message = "检测到疑似人员摔倒，请及时确认。"
        else:
            message = str(task.payload.get("summary", "场景已记录。"))
        return {"module": self.name, "action": task.action, "message": message}
