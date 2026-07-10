"""Alarm and timer module wrapper."""

from __future__ import annotations

from typing import Any

from guidebot.events import Event, EventBus
from guidebot.scheduler import Task


class AlarmTimerModule:
    name = "alarm_timer"

    def __init__(self) -> None:
        self.event_bus: EventBus | None = None
        self.running = False
        self.alarms: dict[str, str] = {}

    def start(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self.running = True

    def stop(self) -> None:
        self.running = False

    def alarm_triggered(self, alarm_id: str, text: str = "时间到了。") -> Event:
        event = Event("alarm.triggered", self.name, {"alarm_id": alarm_id, "text": text})
        if self.event_bus is not None:
            self.event_bus.publish(event)
        return event

    def handle_task(self, task: Task) -> dict[str, Any]:
        if task.action == "set_alarm":
            alarm_time = str(task.payload.get("time") or task.payload.get("text") or "")
            alarm_id = task.task_id
            self.alarms[alarm_id] = alarm_time
            return {
                "module": self.name,
                "action": task.action,
                "alarm_id": alarm_id,
                "time": alarm_time,
                "message": f"已记录提醒：{alarm_time}",
            }
        if task.action == "remind":
            return {
                "module": self.name,
                "action": task.action,
                "message": str(task.payload.get("text", "时间到了。")),
            }
        return {"module": self.name, "action": task.action, "message": "闹钟任务已处理。"}
