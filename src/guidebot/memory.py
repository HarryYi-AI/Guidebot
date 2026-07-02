"""Memory stream with similarity and exponential time decay."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Sequence

from .failure_attribution import FailureAttribution

from .models import Decision, utc_now
from .observation import FeatureMapper, Observation
from .reflection import Critique, EnvironmentFeedback


@dataclass(frozen=True, slots=True)
class Experience:
    observation: Observation
    skill_name: str
    decision: Decision
    feedback: EnvironmentFeedback
    critique: Critique
    timestamp: datetime = field(default_factory=utc_now)
    attribution: FailureAttribution | None = None

    @property
    def failed(self) -> bool:
        return not self.feedback.success or self.feedback.reward < 0 or self.critique.severity > 0


@dataclass(frozen=True, slots=True)
class RetrievedMemory:
    experience: Experience
    similarity: float
    decay: float
    score: float


class MemoryStream:
    """Stores ``M_t`` and retrieves by ``cos(φ(q), φ(m_i))·exp(-λΔt)``."""

    def __init__(
        self,
        decay_lambda: float = 1.0 / 86_400.0,
        mapper: FeatureMapper | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        if decay_lambda < 0:
            raise ValueError("decay_lambda must be non-negative")
        self.decay_lambda = decay_lambda
        self.mapper = mapper or FeatureMapper()
        self.clock = clock
        self._items: list[Experience] = []

    def add(self, experience: Experience) -> None:
        self._items.append(experience)

    def failures(self, failure_mode: str | None = None) -> tuple[Experience, ...]:
        return tuple(
            item
            for item in self._items
            if item.failed and (failure_mode is None or item.critique.failure_mode == failure_mode)
        )

    def retrieve(
        self,
        query: Observation,
        limit: int = 5,
        *,
        failures_only: bool = False,
        now: datetime | None = None,
    ) -> tuple[RetrievedMemory, ...]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        current_time = now or self.clock()
        query_vector = self.mapper(query)
        candidates = self.failures() if failures_only else tuple(self._items)
        ranked: list[RetrievedMemory] = []
        for item in candidates:
            similarity = self.cosine(query_vector, self.mapper(item.observation))
            age_seconds = max(0.0, (current_time - item.timestamp).total_seconds())
            decay = math.exp(-self.decay_lambda * age_seconds)
            ranked.append(RetrievedMemory(item, similarity, decay, similarity * decay))
        ranked.sort(key=lambda result: result.score, reverse=True)
        return tuple(ranked[:limit])

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):  # type intentionally inferred for a lightweight read-only iterator
        return iter(self._items)

    @staticmethod
    def cosine(left: Sequence[float], right: Sequence[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 and right_norm == 0:
            return 1.0
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)
