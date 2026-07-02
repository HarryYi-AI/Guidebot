"""Causal failure classification before any policy mutation is considered."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from .models import Trajectory
from .observation import Observation
from .reflection import EnvironmentFeedback
from .safety import SafetyResult


class FailureType(str, Enum):
    SKILL_ERROR = "skill_error"
    EXECUTION_LAPSE = "execution_lapse"
    SENSOR_NOISE = "sensor_noise"
    DELAYED_EFFECT = "delayed_effect"
    USER_PREFERENCE_SHIFT = "user_preference_shift"
    SAFETY_REJECTION = "safety_rejection"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FailureAttribution:
    failure_type: FailureType
    confidence: float
    evidence: tuple[str, ...]
    should_evolve_skill: bool


class FailureAttributor:
    """Separates policy errors from sensor, actuator, delay, and safety failures."""

    preference_evolution_threshold = 0.8

    def attribute(
        self,
        trajectory: Trajectory,
        *,
        safety_result: SafetyResult | Sequence[SafetyResult] | None = None,
        device_feedback: EnvironmentFeedback | Mapping[str, Any] | None = None,
        observation_delta: tuple[Observation, Observation] | float | None = None,
        user_feedback: Mapping[str, Any] | str | None = None,
    ) -> FailureAttribution:
        details = self._details(device_feedback)
        if trajectory.rejected_actions or self._safety_rejected(safety_result):
            return self._result(
                FailureType.SAFETY_REJECTION,
                1.0,
                "action was blocked by the immutable safety gate",
            )

        if details.get("ir_missed_action") or details.get("execution_lapse"):
            return self._result(
                FailureType.EXECUTION_LAPSE,
                0.95,
                "device reported that the accepted command was not executed",
            )

        if details.get("sensor_noise") or self._implausible_delta(observation_delta):
            return self._result(
                FailureType.SENSOR_NOISE,
                0.9,
                "observation delta is inconsistent with the physical state estimate",
            )

        if details.get("delayed_effect"):
            return self._result(
                FailureType.DELAYED_EFFECT,
                0.9,
                "actuator effect is inside the configured response delay",
            )

        preference_confidence = self._preference_shift_confidence(user_feedback, details)
        if preference_confidence > 0:
            return self._result(
                FailureType.USER_PREFERENCE_SHIFT,
                preference_confidence,
                "user explicitly changed the desired behavior or comfort target",
            )

        failed = self._feedback_failed(device_feedback)
        if failed and trajectory.accepted_actions:
            return self._result(
                FailureType.SKILL_ERROR,
                0.85,
                "skill executed as proposed but environment or user outcome failed",
            )
        return self._result(FailureType.UNKNOWN, 0.4, "insufficient causal evidence")

    def _result(self, failure_type: FailureType, confidence: float, evidence: str) -> FailureAttribution:
        should_evolve = failure_type is FailureType.SKILL_ERROR or (
            failure_type is FailureType.USER_PREFERENCE_SHIFT
            and confidence >= self.preference_evolution_threshold
        )
        return FailureAttribution(failure_type, confidence, (evidence,), should_evolve)

    @staticmethod
    def _details(
        feedback: EnvironmentFeedback | Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        if isinstance(feedback, EnvironmentFeedback):
            return feedback.details
        return feedback or {}

    @staticmethod
    def _feedback_failed(
        feedback: EnvironmentFeedback | Mapping[str, Any] | None,
    ) -> bool:
        if isinstance(feedback, EnvironmentFeedback):
            return not feedback.success or feedback.reward < 0
        return bool(feedback and feedback.get("success") is False)

    @staticmethod
    def _safety_rejected(result: SafetyResult | Sequence[SafetyResult] | None) -> bool:
        if isinstance(result, SafetyResult):
            return not result.allowed
        return any(not item.allowed for item in result) if result else False

    @staticmethod
    def _implausible_delta(delta: tuple[Observation, Observation] | float | None) -> bool:
        if isinstance(delta, (int, float)):
            return abs(float(delta)) > 8.0
        if isinstance(delta, tuple):
            return abs(delta[1].temperature_c - delta[0].temperature_c) > 8.0
        return False

    @staticmethod
    def _preference_shift_confidence(
        user_feedback: Mapping[str, Any] | str | None,
        details: Mapping[str, Any],
    ) -> float:
        if isinstance(user_feedback, Mapping) and user_feedback.get("preference_shift"):
            return float(user_feedback.get("confidence", 0.9))
        if isinstance(user_feedback, str) and "prefer" in user_feedback.lower():
            return 0.8
        if details.get("user_preference_shift"):
            return float(details.get("preference_confidence", 0.9))
        return 0.0
