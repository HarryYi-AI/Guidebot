"""Intent model shared by analyzers, scheduler, and module executors."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .events import Event


class IntentType(str, Enum):
    CHAT = "chat"
    SET_ALARM = "set_alarm"
    CANCEL_ALARM = "cancel_alarm"
    TIMER_REMINDER = "timer_reminder"
    SAFETY_FIRE_ALERT = "safety_fire_alert"
    SAFETY_FALL_ALERT = "safety_fall_alert"
    HEALTH_SEDENTARY = "health_sedentary"
    HEALTH_FATIGUE = "health_fatigue"
    MOBILITY_STOP = "mobility_stop"
    CLIMATE_COMFORT = "climate_comfort"
    AC_LEFT_ON_ALERT = "ac_left_on_alert"
    PET_INTERACTION = "pet_interaction"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Intent:
    intent_type: IntentType
    source_event: Event
    slots: dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    confidence: float = 1.0
    requires_confirmation: bool = False
