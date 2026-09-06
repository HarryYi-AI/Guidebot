"""Dependency-free domain models shared by agents and hardware adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SensorKind(str, Enum):
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    AIR_QUALITY = "air_quality"
    TOUCH = "touch"
    MOTION = "motion"
    LIGHT = "light"
    DISTANCE = "distance"


class ActionKind(str, Enum):
    INSPECT_ROOM = "inspect_room"
    SET_HVAC = "set_hvac"
    SET_ALARM = "set_alarm"
    SPEAK = "speak"
    DISPLAY = "display"
    MOVE = "move"
    NOTIFY = "notify"


@dataclass(frozen=True, slots=True)
class Reading:
    kind: SensorKind
    value: float | bool | str
    unit: str = ""
    source: str = "unknown"
    timestamp: datetime = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    parameters: Mapping[str, Any]
    reason: str
    requested_by: str = "agent"
    id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class Decision:
    actions: tuple[Action, ...] = ()
    response: str | None = None
    rationale: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RobotState:
    readings: dict[SensorKind, Reading] = field(default_factory=dict)
    hvac_target_c: float | None = None
    last_interaction: datetime | None = None
    health: str = "starting"

    def update(self, reading: Reading) -> None:
        previous = self.readings.get(reading.kind)
        if previous is None or reading.timestamp >= previous.timestamp:
            self.readings[reading.kind] = reading

    def value(self, kind: SensorKind, default: Any = None) -> Any:
        reading = self.readings.get(kind)
        return reading.value if reading else default


@dataclass(frozen=True, slots=True)
class DomainEvent:
    topic: str
    payload: Any
    id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = field(default_factory=utc_now)


Event = DomainEvent
# Backward-compatible alias for the Hub/self-evolution domain event.


@dataclass(frozen=True, slots=True)
class Trajectory:
    trigger: DomainEvent
    decision: Decision
    accepted_actions: tuple[Action, ...]
    rejected_actions: tuple[Action, ...]
    score: float | None = None
    feedback: str | None = None
    timestamp: datetime = field(default_factory=utc_now)
