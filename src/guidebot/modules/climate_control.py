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

    def observe(
        self,
        *,
        temperature_c: float | None = None,
        humidity: float | None = None,
        ac_on: bool | None = None,
        occupied: bool | None = None,
        confidence: float = 0.9,
    ) -> Event:
        if temperature_c is not None:
            self.temperature_c = float(temperature_c)
        if humidity is not None:
            self.humidity = float(humidity)
        payload: dict[str, Any] = {
            "temperature_c": self.temperature_c,
            "humidity": self.humidity,
        }
        if ac_on is not None:
            payload["ac_on"] = ac_on
        if occupied is not None:
            payload["occupied"] = occupied
        event = Event("climate.detected", self.name, payload, confidence=confidence)
        if self.event_bus is not None:
            self.event_bus.publish(event)
        return event

    def handle_task(self, task: Task) -> dict[str, Any]:
        if task.action == "suggest_comfort":
            message = _comfort_message(task.payload)
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


def _comfort_message(payload: dict[str, Any]) -> str:
    temperature = payload.get("temperature_c", payload.get("temperature"))
    humidity = payload.get("humidity", payload.get("humidity_pct"))
    parts = []
    if isinstance(temperature, (int, float)):
        if float(temperature) > 27:
            parts.append(f"室温 {float(temperature):.1f}°C 偏高")
        elif float(temperature) < 18:
            parts.append(f"室温 {float(temperature):.1f}°C 偏低")
    if isinstance(humidity, (int, float)) and float(humidity) > 70:
        parts.append(f"湿度 {float(humidity):.0f}% 偏高")
    prefix = "，".join(parts) if parts else "当前体感可能不舒适"
    return f"{prefix}。我先不直接控制空调，建议把环境调到舒适区间。"
