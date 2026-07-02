"""Structured trajectory critique c_t = R(o_t, a_t, r_t, o_{t+1})."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from .models import ActionKind, Trajectory
from .observation import Observation


@dataclass(frozen=True, slots=True)
class EnvironmentFeedback:
    reward: float
    success: bool
    next_observation: Observation
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Critique:
    """Structured vector ``[causal_factor, failure_mode, severity, suggested_action]``."""

    causal_factor: str
    failure_mode: str
    severity: float
    suggested_action: str
    confidence: float
    entropy: float


class ReflectionEngine:
    """Deterministic baseline reflector, replaceable by an LLM implementation."""

    def reflect(
        self,
        observation: Observation,
        trajectory: Trajectory,
        feedback: EnvironmentFeedback,
    ) -> Critique:
        if trajectory.rejected_actions:
            return self._critique(
                "proposed action violated a deterministic safety constraint",
                "safety_rejection",
                1.0,
                "revise the skill to stay inside the device safety envelope",
                0.99,
            )

        mode = str(feedback.details.get("failure_mode", ""))
        if not feedback.success or feedback.reward < 0:
            return self._critique(
                feedback.message or "environment reward indicates the selected skill failed",
                mode or "execution_failure",
                min(1.0, max(0.1, abs(feedback.reward))),
                str(feedback.details.get("suggested_action", "select a recovery skill")),
                0.85,
            )

        hvac_actions = [
            action for action in trajectory.accepted_actions if action.kind is ActionKind.SET_HVAC
        ]
        if hvac_actions and self._temperature_worsened(observation, feedback.next_observation):
            return self._critique(
                "temperature moved farther from the comfort band after HVAC control",
                "ineffective_temperature_control",
                0.7,
                "verify HVAC state and use a bounded corrective temperature skill",
                0.9,
            )

        return self._critique(
            "action and observed outcome are consistent",
            "none",
            0.0,
            "preserve the current skill behavior",
            0.95,
        )

    @staticmethod
    def _temperature_worsened(before: Observation, after: Observation) -> bool:
        comfort = 23.0
        return abs(after.temperature_c - comfort) > abs(before.temperature_c - comfort) + 0.25

    @staticmethod
    def _critique(
        cause: str,
        mode: str,
        severity: float,
        suggestion: str,
        confidence: float,
    ) -> Critique:
        # Binary entropy H(p) quantifies reflection uncertainty; lower is sharper.
        p = min(1.0 - 1e-12, max(1e-12, confidence))
        entropy = -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))
        return Critique(cause, mode, severity, suggestion, confidence, entropy)

