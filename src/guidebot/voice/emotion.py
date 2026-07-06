"""Provider-neutral affect and prosody policy for empathic voice backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import CapturedTurn


@dataclass(frozen=True, slots=True)
class AffectState:
    label: str
    valence: float
    arousal: float
    confidence: float

    def __post_init__(self) -> None:
        if not -1.0 <= self.valence <= 1.0:
            raise ValueError("valence must be within -1..1")
        if not 0.0 <= self.arousal <= 1.0 or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("arousal and confidence must be within 0..1")


@dataclass(frozen=True, slots=True)
class ProsodyStyle:
    name: str
    warmth: float
    energy: float
    speaking_rate: float
    instruction: str


class AffectAnalyzer(Protocol):
    async def analyze(self, turn: CapturedTurn, transcript: str) -> AffectState: ...


class EmpathicStylePolicy:
    """Maps perceived affect to bounded response prosody, without diagnosing users."""

    def choose(self, affect: AffectState) -> ProsodyStyle:
        label = affect.label.lower()
        if affect.confidence < 0.55:
            return ProsodyStyle("neutral-warm", 0.75, 0.45, 1.0, "温和自然，不夸张推断情绪")
        if label in {"sad", "lonely", "anxious", "frustrated"} or affect.valence < -0.35:
            return ProsodyStyle("supportive", 0.95, 0.3, 0.9, "先接住感受，语速稍慢，避免说教")
        if label in {"angry", "stressed"} or affect.arousal > 0.85 and affect.valence < 0:
            return ProsodyStyle("grounding", 0.8, 0.2, 0.85, "保持平稳、简短，先确认诉求")
        if label in {"happy", "excited"} or affect.valence > 0.5:
            return ProsodyStyle("cheerful", 0.8, 0.75, 1.08, "积极回应，但不要抢话或过度兴奋")
        return ProsodyStyle("attentive", 0.8, 0.5, 1.0, "专注、自然、有回应感")

