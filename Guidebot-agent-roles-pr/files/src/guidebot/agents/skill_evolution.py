"""Agent role for reflecting on trajectories and evolving reusable skills."""

from __future__ import annotations

from dataclasses import dataclass

from guidebot.failure_attribution import FailureAttribution, FailureAttributor
from guidebot.memory import Experience, MemoryStream
from guidebot.models import RobotState, SensorKind, Trajectory
from guidebot.observation import Observation
from guidebot.policy_evolution import EvolutionOutcome, PolicyEvolution
from guidebot.reflection import Critique, EnvironmentFeedback, ReflectionEngine
from guidebot.skills import SkillLibrary


@dataclass(frozen=True, slots=True)
class SkillEvolutionReport:
    """Audit record for one skill-evolution review step."""

    experience: Experience
    attribution: FailureAttribution
    critique: Critique
    outcome: EvolutionOutcome
    memory_size: int


class SkillEvolutionAgent:
    """Owns the post-action learning loop without owning physical execution."""

    def __init__(
        self,
        library: SkillLibrary,
        *,
        reflection: ReflectionEngine | None = None,
        memory: MemoryStream | None = None,
        evolution: PolicyEvolution | None = None,
        attributor: FailureAttributor | None = None,
    ) -> None:
        self.library = library
        self.reflection = reflection or ReflectionEngine()
        self.memory = memory or MemoryStream()
        self.evolution = evolution or PolicyEvolution()
        self.attributor = attributor or FailureAttributor()
        self.last_report: SkillEvolutionReport | None = None

    def observe(
        self,
        *,
        observation: Observation,
        trajectory: Trajectory,
        state: RobotState,
        skill_name: str,
        feedback: EnvironmentFeedback | None = None,
    ) -> SkillEvolutionReport:
        environment_feedback = feedback or self._infer_feedback(trajectory, observation, state)
        attribution = self.attributor.attribute(
            trajectory,
            device_feedback=environment_feedback,
            observation_delta=(observation, environment_feedback.next_observation),
            user_feedback=environment_feedback.details.get("user_feedback"),
        )
        critique = self.reflection.reflect(observation, trajectory, environment_feedback)
        experience = Experience(
            observation=observation,
            skill_name=skill_name,
            decision=trajectory.decision,
            feedback=environment_feedback,
            critique=critique,
            timestamp=trajectory.timestamp,
            attribution=attribution,
        )
        self.memory.add(experience)

        if self.library.find(skill_name) is not None:
            self.library.record_outcome(
                skill_name,
                success=environment_feedback.success and not trajectory.rejected_actions,
                failure_mode=None if critique.failure_mode == "none" else critique.failure_mode,
            )

        outcome = self.evolution.evolve(self.memory, self.library)
        report = SkillEvolutionReport(
            experience=experience,
            attribution=attribution,
            critique=critique,
            outcome=outcome,
            memory_size=len(self.memory),
        )
        self.last_report = report
        return report

    def review_memory(self) -> EvolutionOutcome:
        """Run evolution over accumulated memory without adding a new trajectory."""

        return self.evolution.evolve(self.memory, self.library)

    @staticmethod
    def _infer_feedback(
        trajectory: Trajectory,
        observation: Observation,
        state: RobotState,
    ) -> EnvironmentFeedback:
        rejected = bool(trajectory.rejected_actions)
        reward = -1.0 if rejected else (1.0 if trajectory.accepted_actions else 0.0)
        next_observation = Observation(
            temperature_c=_numeric(state.value(SensorKind.TEMPERATURE), observation.temperature_c),
            humidity_pct=_numeric(state.value(SensorKind.HUMIDITY), observation.humidity_pct),
            touch=observation.touch,
            user_signal=observation.user_signal,
            event_kind=observation.event_kind,
            context=observation.context,
        )
        return EnvironmentFeedback(
            reward=reward,
            success=not rejected,
            next_observation=next_observation,
            message="action rejected by safety policy" if rejected else "execution accepted",
            details={"failure_mode": "safety_rejection"} if rejected else {},
        )


def _numeric(value: object, default: float) -> float:
    return float(value) if isinstance(value, (int, float)) else default
