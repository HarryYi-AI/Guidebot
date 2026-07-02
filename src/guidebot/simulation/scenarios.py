"""Scenario definitions for reproducible held-out robot evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from guidebot.observation import Observation


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    initial_observation: Observation
    environment_params: Mapping[str, Any] = field(default_factory=dict)
    user_feedback_schedule: Mapping[int, Mapping[str, Any] | str | float] = field(
        default_factory=dict
    )
    expected_behavior: Mapping[str, Any] = field(default_factory=dict)


def default_stress_scenarios() -> tuple[Scenario, ...]:
    """Held-out thermal cases covering delay, humidity, noise, and IR loss."""

    base_expected = {"preferred_temperature_c": 23.0, "comfort_tolerance_c": 1.0}
    return (
        Scenario(
            "hot_room",
            Observation(30.0, 50.0, event_kind="temperature"),
            {"ambient_temperature_c": 31.0},
            expected_behavior=base_expected,
        ),
        Scenario(
            "humid_hot_room",
            Observation(29.0, 82.0, event_kind="temperature"),
            {"ambient_temperature_c": 30.0, "humidity_factor": 0.65},
            expected_behavior=base_expected,
        ),
        Scenario(
            "delayed_ac",
            Observation(30.0, 55.0, event_kind="temperature"),
            {"ambient_temperature_c": 31.0, "ac_delay_steps": 2},
            expected_behavior=base_expected,
        ),
        Scenario(
            "noisy_sensor",
            Observation(29.0, 50.0, event_kind="temperature"),
            {"ambient_temperature_c": 30.0, "sensor_noise": 0.35, "random_seed": 7},
            expected_behavior=base_expected,
        ),
        Scenario(
            "missed_ir_once",
            Observation(30.0, 50.0, event_kind="temperature"),
            {"ambient_temperature_c": 31.0, "ir_missed_action": 0.25, "random_seed": 3},
            expected_behavior=base_expected,
        ),
    )

