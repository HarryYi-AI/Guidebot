"""Continuous observation representation and feature mapping φ(o_t)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .models import Event, Reading, RobotState, SensorKind


@dataclass(frozen=True, slots=True)
class Observation:
    """Guidebot observation ``o_t = [T, H, τ, u_t] ∈ R⁴``.

    Missing physical values use neutral defaults so the vector dimension remains
    stable. Event metadata is deliberately outside the numeric vector and can be
    used by deterministic skill preconditions.
    """

    temperature_c: float = 22.0
    humidity_pct: float = 50.0
    touch: float = 0.0
    user_signal: float = 0.0
    event_kind: str = "unknown"
    context: Mapping[str, Any] = field(default_factory=dict)

    def vector(self) -> tuple[float, float, float, float]:
        return (self.temperature_c, self.humidity_pct, self.touch, self.user_signal)


class FeatureMapper:
    """Maps raw observations into a bounded router feature vector.

    ``φ(o) = [(T-22)/10, (H-50)/50, clip(τ), clip(u_t)]``.
    Centering temperature and humidity keeps route weights interpretable and
    avoids scale dominance in ``w_kᵀφ(o)``.
    """

    dimension = 4

    def __call__(self, observation: Observation) -> tuple[float, ...]:
        return (
            (observation.temperature_c - 22.0) / 10.0,
            (observation.humidity_pct - 50.0) / 50.0,
            self._clip(observation.touch),
            self._clip(observation.user_signal),
        )

    @staticmethod
    def _clip(value: float) -> float:
        return max(0.0, min(1.0, float(value)))


def observation_from_event(event: Event, state: RobotState) -> Observation:
    """Create a stable continuous observation from an event and current state."""

    temperature = _number(state.value(SensorKind.TEMPERATURE), 22.0)
    humidity = _number(state.value(SensorKind.HUMIDITY), 50.0)
    touch = 0.0
    user_signal = 1.0 if event.topic == "user.message" else 0.0
    event_kind = event.topic
    context: dict[str, Any] = {"event_topic": event.topic}

    if isinstance(event.payload, Reading):
        event_kind = event.payload.kind.value
        context.update(
            reading_value=event.payload.value,
            reading_unit=event.payload.unit,
            reading_source=event.payload.source,
        )
        if event.payload.kind is SensorKind.TEMPERATURE:
            temperature = _number(event.payload.value, temperature)
        elif event.payload.kind is SensorKind.HUMIDITY:
            humidity = _number(event.payload.value, humidity)
        elif event.payload.kind is SensorKind.TOUCH:
            touch = float(bool(event.payload.value))
    elif event.topic == "user.message":
        context["user_message"] = str(event.payload)

    return Observation(temperature, humidity, touch, user_signal, event_kind, context)


def _number(value: Any, default: float) -> float:
    return float(value) if isinstance(value, (int, float)) else default

