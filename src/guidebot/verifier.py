"""Simulation-gated verifier for candidate skill activation."""

from __future__ import annotations

from dataclasses import dataclass

from .models import Decision, RobotState
from .observation import Observation
from .simulation.scenarios import Scenario, default_stress_scenarios
from .simulation.suite import SimulationSuite
from .skill_card import SkillCard
from .skills import Skill


@dataclass(frozen=True, slots=True)
class VerificationResult:
    accepted: bool
    score_delta: float
    failed_scenarios: tuple[str, ...]
    safety_violations: int
    reason: str


class VerifierAgent:
    """Accepts iff safety is clean and held-out score strictly improves."""

    def __init__(self, suite: SimulationSuite | None = None) -> None:
        self.suite = suite

    def generate_stress_scenarios(self, card: SkillCard) -> tuple[Scenario, ...]:
        effects = tuple(card.action_schema.get("effects", ()))
        if "notifies_user" in effects or "notify:user" in effects:
            return (
                Scenario(
                    "notification_recovery",
                    Observation(23, 50, event_kind="user.message"),
                    expected_behavior={
                        "preferred_temperature_c": 23,
                        "required_action": "notify",
                    },
                ),
            )
        return default_stress_scenarios()

    def verify(
        self,
        candidate: Skill | SkillCard,
        *,
        card: SkillCard | None = None,
        baseline: Skill | None = None,
        executable: Skill | None = None,
    ) -> VerificationResult:
        if isinstance(candidate, SkillCard):
            metadata = candidate
            if executable is None:
                raise ValueError("an executable Skill is required when verifying a SkillCard")
            candidate_skill = executable
        else:
            candidate_skill = candidate
            metadata = card or SkillCard.create(candidate.name)

        suite = self.suite or SimulationSuite(self.generate_stress_scenarios(metadata))
        candidate_report = suite.run(candidate_skill)
        baseline_report = suite.run(baseline or self._no_op_skill())
        violations = candidate_report.metrics.unsafe_action_count
        delta = candidate_report.score - baseline_report.score
        if violations > 0:
            return VerificationResult(
                False,
                delta,
                candidate_report.failed_scenarios,
                violations,
                "candidate violated immutable safety constraints",
            )
        if delta <= 0:
            return VerificationResult(
                False,
                delta,
                candidate_report.failed_scenarios,
                0,
                "held-out simulation score did not strictly improve",
            )
        return VerificationResult(
            True,
            delta,
            candidate_report.failed_scenarios,
            0,
            "safe candidate strictly improved held-out simulation score",
        )

    @staticmethod
    def _no_op_skill() -> Skill:
        def precondition(observation: Observation, state: RobotState) -> bool:
            return True

        def policy(observation: Observation, state: RobotState) -> Decision:
            return Decision(rationale="verifier baseline no-op")

        return Skill("verifier_no_op", policy, precondition, (), (0, 0, 0, 0))

