"""Small room temperature dynamics model for fast policy stress tests."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping

from guidebot.models import Action, ActionKind
from guidebot.observation import Observation


@dataclass(frozen=True, slots=True)
class RoomStep:
    observation: Observation
    true_temperature_c: float
    action_applied: bool
    delayed: bool


class RoomDynamics:
    """First-order thermal plant with AC delay and imperfect actuation.

    ``T_{t+1}=T_t+κ_a(T_ambient-T_t)+κ_h·h(H)·(T_target-T_t)+ε``.
    AC influence starts after ``ac_delay_steps``; ``ir_missed_action`` is the
    probability that a newly issued IR command never reaches the device.
    """

    def __init__(self, initial: Observation, params: Mapping[str, Any] | None = None) -> None:
        config = dict(params or {})
        self.temperature_c = initial.temperature_c
        self.humidity_pct = initial.humidity_pct
        self.ambient_temperature_c = float(
            config.get("ambient_temperature_c", initial.temperature_c)
        )
        self.ac_delay_steps = int(config.get("ac_delay_steps", 0))
        self.humidity_factor = float(config.get("humidity_factor", 1.0))
        self.sensor_noise = float(config.get("sensor_noise", 0.0))
        self.ir_missed_action = float(config.get("ir_missed_action", 0.0))
        self.ambient_exchange = float(config.get("ambient_exchange", 0.05))
        self.hvac_gain = float(config.get("hvac_gain", 0.22))
        self._rng = random.Random(int(config.get("random_seed", 0)))
        self._target_c: float | None = None
        self._delay_remaining = 0

    def step(self, action: Action | None = None) -> RoomStep:
        action_applied = False
        if action is not None and action.kind is ActionKind.SET_HVAC:
            missed = self._rng.random() < self.ir_missed_action
            if not missed:
                self._target_c = float(action.parameters["target_c"])
                self._delay_remaining = self.ac_delay_steps
                action_applied = True

        ambient_delta = self.ambient_exchange * (
            self.ambient_temperature_c - self.temperature_c
        )
        hvac_delta = 0.0
        delayed = self._target_c is not None and self._delay_remaining > 0
        if self._target_c is not None:
            if self._delay_remaining > 0:
                self._delay_remaining -= 1
            else:
                efficiency = max(0.1, min(1.5, self.humidity_factor))
                raw_delta = self.hvac_gain * efficiency * (
                    self._target_c - self.temperature_c
                )
                hvac_delta = max(-1.5, min(1.5, raw_delta))

        self.temperature_c += ambient_delta + hvac_delta
        self.humidity_pct += 0.02 * (50.0 - self.humidity_pct)
        measured = self.temperature_c + self._rng.gauss(0.0, self.sensor_noise)
        observation = Observation(
            measured,
            self.humidity_pct,
            event_kind="temperature",
            context={"true_temperature_c": self.temperature_c},
        )
        return RoomStep(observation, self.temperature_c, action_applied, delayed)

