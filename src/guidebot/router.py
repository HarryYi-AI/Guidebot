"""Hierarchical conditional router π_route(o_t)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import RobotState
from .observation import FeatureMapper, Observation
from .skills import Skill, SkillLibrary


@dataclass(frozen=True, slots=True)
class RouteScore:
    skill_name: str
    score: float
    level: int


@dataclass(frozen=True, slots=True)
class RouteDecision:
    skill: Skill
    score: float
    scores: tuple[RouteScore, ...]


class HierarchicalRouter:
    """Computes ``S_k(o_t) = σ(w_kᵀ φ(o_t))`` over applicable skills.

    Lower level numbers are evaluated first. The best skill at the first level
    whose score reaches ``level_threshold`` wins; otherwise the global maximum
    is selected. This makes urgent physical skills precede social fallbacks while
    keeping the scoring rule uniform and inspectable.
    """

    def __init__(self, mapper: FeatureMapper | None = None, level_threshold: float = 0.5) -> None:
        self.mapper = mapper or FeatureMapper()
        self.level_threshold = level_threshold

    def route(
        self,
        observation: Observation,
        state: RobotState,
        library: SkillLibrary,
    ) -> RouteDecision:
        features = self.mapper(observation)
        scored: list[tuple[Skill, float]] = []
        for skill in library:
            if len(skill.route_weights) != len(features):
                raise ValueError(
                    f"skill {skill.name!r} has {len(skill.route_weights)} weights; "
                    f"expected {len(features)}"
                )
            if skill.applicable(observation, state):
                logit = sum(weight * value for weight, value in zip(skill.route_weights, features))
                scored.append((skill, self.sigmoid(logit)))

        if not scored:
            raise LookupError("no applicable skill")

        all_scores = tuple(
            RouteScore(skill.name, score, skill.level)
            for skill, score in sorted(scored, key=lambda item: (item[0].level, -item[1]))
        )
        for level in sorted({skill.level for skill, _ in scored}):
            winner = max(
                ((skill, score) for skill, score in scored if skill.level == level),
                key=lambda item: item[1],
            )
            if winner[1] >= self.level_threshold:
                return RouteDecision(winner[0], winner[1], all_scores)

        skill, score = max(scored, key=lambda item: item[1])
        return RouteDecision(skill, score, all_scores)

    @staticmethod
    def sigmoid(value: float) -> float:
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

