"""Decision agent contract and a deterministic MVP implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .models import (
    Action,
    ActionKind,
    Decision,
    DomainEvent,
    Reading,
    RobotState,
    SensorKind,
)


class Agent(Protocol):
    async def decide(self, event: DomainEvent, state: RobotState) -> Decision: ...


class AdaptiveAgent:
    """MVP hub brain.

    The rule policy keeps the first milestone testable. A future LLM planner can
    implement the same Agent protocol and consume the same versioned skill text.
    """

    def __init__(self, skill_path: Path | None = None) -> None:
        default = Path(__file__).parents[2] / "skills" / "core.md"
        self.skill_path = skill_path or default
        self.skill_text = self.skill_path.read_text(encoding="utf-8")

    async def decide(self, event: DomainEvent, state: RobotState) -> Decision:
        if event.topic == "user.message":
            text = str(event.payload).strip()
            return Decision(response=f"我听到了：{text}", rationale="conversation fallback")

        if event.topic != "sensor.reading" or not isinstance(event.payload, Reading):
            return Decision(rationale="no policy for this event")

        reading = event.payload
        if reading.kind is SensorKind.TEMPERATURE and isinstance(reading.value, (int, float)):
            temperature = float(reading.value)
            if temperature > 27:
                return self._set_hvac(25, f"室温 {temperature:.1f}°C 偏高")
            if temperature < 18:
                return self._set_hvac(22, f"室温 {temperature:.1f}°C 偏低")

        if reading.kind is SensorKind.TOUCH and bool(reading.value):
            action = Action(ActionKind.SPEAK, {"text": "嘿，我在呢。"}, "detected friendly touch")
            return Decision((action,), "嘿，我在呢。", "touch interaction")

        if reading.kind is SensorKind.AIR_QUALITY and isinstance(reading.value, (int, float)):
            if float(reading.value) > 100:
                action = Action(
                    ActionKind.NOTIFY,
                    {"level": "warning", "message": "房间空气质量需要关注"},
                    "air quality threshold exceeded",
                )
                return Decision((action,), "空气质量有些差，建议通风。", "room health alert")

        return Decision(rationale="reading recorded; no action required")

    @staticmethod
    def _set_hvac(target: float, reason: str) -> Decision:
        action = Action(ActionKind.SET_HVAC, {"target_c": target}, reason)
        return Decision((action,), f"{reason}，我把空调设为 {target:.0f}°C。", "thermal comfort")
