"""Multimodal observations converted into explicit, updateable beliefs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .events import RuntimeEvent


class BeliefBand(str, Enum):
    UNLIKELY = "unlikely"
    UNCERTAIN = "uncertain"
    CONFIDENT = "confident"


@dataclass(frozen=True, slots=True)
class BeliefState:
    hypothesis: str
    probability: float
    band: BeliefBand
    evidence_count: int


@dataclass(frozen=True, slots=True)
class BeliefUpdate:
    before: BeliefState
    after: BeliefState
    confidence_delta: float
    evidence: str


class MultimodalPerception:
    """Fuse repeated detector observations without giving them actuator authority."""

    def __init__(self, *, uncertain_low: float = 0.45, confident_high: float = 0.85) -> None:
        if not 0 < uncertain_low < confident_high < 1:
            raise ValueError("belief thresholds must satisfy 0 < low < high < 1")
        self.uncertain_low = uncertain_low
        self.confident_high = confident_high
        self._beliefs: dict[tuple[str, str], BeliefState] = {}

    def observe(self, event: RuntimeEvent) -> BeliefUpdate:
        hypothesis, support = self._evidence(event)
        key = (event.session_id or event.source, hypothesis)
        before = self._beliefs.get(
            key,
            BeliefState(hypothesis, 0.5, BeliefBand.UNCERTAIN, 0),
        )
        probability = support if before.evidence_count == 0 else _bayesian_update(
            before.probability,
            support,
        )
        after = BeliefState(
            hypothesis,
            probability,
            self._band(probability),
            before.evidence_count + 1,
        )
        self._beliefs[key] = after
        return BeliefUpdate(
            before,
            after,
            after.probability - before.probability,
            f"{event.event_type}:{event.payload.get('label', event.source)}",
        )

    def _band(self, probability: float) -> BeliefBand:
        if probability >= self.confident_high:
            return BeliefBand.CONFIDENT
        if probability >= self.uncertain_low:
            return BeliefBand.UNCERTAIN
        return BeliefBand.UNLIKELY

    @staticmethod
    def _evidence(event: RuntimeEvent) -> tuple[str, float]:
        label = str(event.payload.get("label", "")).casefold()
        is_fire = any(token in label for token in ("fire", "smoke", "明火", "烟"))
        explicitly_about_fire = str(event.payload.get("hypothesis", "")).casefold() == "fire"
        if event.event_type.startswith("scene.") and (is_fire or explicitly_about_fire):
            hypothesis = "fire"
            support = event.confidence if is_fire else 1.0 - event.confidence
        else:
            hypothesis = label or event.event_type
            support = event.confidence
        return hypothesis, _clamp_probability(support)


def _bayesian_update(prior: float, support: float) -> float:
    """Update odds assuming conditionally independent detector observations."""
    numerator = prior * support
    denominator = numerator + (1.0 - prior) * (1.0 - support)
    return numerator / denominator if denominator else prior


def _clamp_probability(value: float) -> float:
    return min(1.0 - 1e-6, max(1e-6, float(value)))
