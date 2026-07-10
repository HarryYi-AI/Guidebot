"""Voice chat module wrapper for the unified runtime."""

from __future__ import annotations

from typing import Any

from guidebot.events import Event, EventBus
from guidebot.scheduler import Task


class VoiceChatModule:
    name = "voice_chat"

    def __init__(self) -> None:
        self.event_bus: EventBus | None = None
        self.running = False

    def start(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self.running = True

    def stop(self) -> None:
        self.running = False

    def publish_user_text(self, text: str, *, confidence: float = 1.0) -> Event:
        event = Event("user.text", self.name, {"text": text}, confidence=confidence)
        if self.event_bus is not None:
            self.event_bus.publish(event)
        return event

    def handle_task(self, task: Task) -> dict[str, Any]:
        text = str(task.payload.get("text", ""))
        if task.action == "pet_interaction":
            response = "我在呢。"
        else:
            response = f"我听到了：{text}" if text else "我在。"
        return {"module": self.name, "action": task.action, "message": response}
