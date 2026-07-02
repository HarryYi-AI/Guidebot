"""Metrics for simulation trajectories and candidate evolution."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Sequence


def comfort_error(temperatures: Sequence[float], preferred_c: float) -> float:
    return mean(abs(value - preferred_c) for value in temperatures) if temperatures else 0.0


def recovery_steps(
    temperatures: Sequence[float], preferred_c: float, tolerance_c: float
) -> int:
    for index, value in enumerate(temperatures, start=1):
        if abs(value - preferred_c) <= tolerance_c:
            return index
    return len(temperatures) + 1


def unsafe_action_count(rejected_as_unsafe: Sequence[bool]) -> int:
    return sum(rejected_as_unsafe)


def safety_rejection_count(rejections: Sequence[bool]) -> int:
    return sum(rejections)


def skill_reuse_rate(skill_names: Sequence[str]) -> float:
    if not skill_names:
        return 0.0
    return max(0.0, (len(skill_names) - len(set(skill_names))) / len(skill_names))


def evolution_accept_rate(accepted: int, proposed: int) -> float:
    return accepted / proposed if proposed else 0.0


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    comfort_error: float
    recovery_steps: float
    unsafe_action_count: int
    safety_rejection_count: int
    skill_reuse_rate: float
    evolution_accept_rate: float


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    scenario_name: str
    score: float
    temperatures: tuple[float, ...]
    selected_skills: tuple[str, ...]
    unsafe_actions: int
    safety_rejections: int
    recovered_in_steps: int
    expected_behavior_met: bool


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    score: float
    metrics: EvaluationMetrics
    scenarios: tuple[ScenarioResult, ...]

    @property
    def failed_scenarios(self) -> tuple[str, ...]:
        return tuple(result.scenario_name for result in self.scenarios if not result.expected_behavior_met)

