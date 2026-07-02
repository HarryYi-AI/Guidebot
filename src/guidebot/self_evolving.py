"""Executable self-evolving agent loop for Guidebot."""

from __future__ import annotations

from dataclasses import dataclass

from .failure_attribution import FailureAttribution, FailureAttributor
from .memory import Experience, MemoryStream
from .models import Action, ActionKind, Decision, Event, RobotState, SensorKind, Trajectory
from .observation import Observation, observation_from_event
from .policy_evolution import EvolutionOutcome, PolicyEvolution
from .reflection import Critique, EnvironmentFeedback, ReflectionEngine
from .router import HierarchicalRouter, RouteDecision
from .skills import Skill, SkillLibrary


@dataclass(frozen=True, slots=True)
class AgentStep:
    observation: Observation
    route: RouteDecision
    decision: Decision


@dataclass(frozen=True, slots=True)
class LearningStep:
    experience: Experience
    attribution: FailureAttribution
    critique: Critique
    evolution: EvolutionOutcome


class SelfEvolvingAgent:
    """Runs the unified closed loop around a deterministic skill library.

    Pre-action: ``o_t → π_route(o_t; L_t) → f_skill(o_t) → a_t``.
    Post-action: ``feedback → R(...) → M_{t+1} → E(M, L_t)``.

    The device-owning :class:`GuidebotHub` sits between these two phases so every
    proposed physical action passes through the non-evolvable safety policy.
    """

    def __init__(
        self,
        library: SkillLibrary | None = None,
        router: HierarchicalRouter | None = None,
        reflection: ReflectionEngine | None = None,
        memory: MemoryStream | None = None,
        evolution: PolicyEvolution | None = None,
        attributor: FailureAttributor | None = None,
    ) -> None:
        self.library = library if library is not None else build_default_library()
        self.router = router if router is not None else HierarchicalRouter()
        self.reflection = reflection if reflection is not None else ReflectionEngine()
        self.memory = memory if memory is not None else MemoryStream()
        self.evolution = evolution if evolution is not None else PolicyEvolution()
        self.attributor = attributor if attributor is not None else FailureAttributor()
        self._pending: dict[str, AgentStep] = {}
        self.last_step: AgentStep | None = None
        self.last_learning: LearningStep | None = None

    async def decide(self, event: Event, state: RobotState) -> Decision:
        observation = observation_from_event(event, state)
        route = self.router.route(observation, state, self.library)
        raw_decision = route.skill.execute(observation, state)
        rationale = raw_decision.rationale or "deterministic skill execution"
        decision = Decision(
            raw_decision.actions,
            raw_decision.response,
            f"route={route.skill.name}; score={route.score:.4f}; {rationale}",
        )
        step = AgentStep(observation, route, decision)
        self._pending[event.id] = step
        self.last_step = step
        return decision

    def observe_outcome(
        self,
        trajectory: Trajectory,
        state: RobotState,
        feedback: EnvironmentFeedback | None = None,
    ) -> LearningStep:
        """Finish feedback, reflection, memory update, and policy evolution."""

        try:
            step = self._pending.pop(trajectory.trigger.id)
        except KeyError as error:
            raise KeyError("trajectory has no pending agent step") from error

        environment_feedback = feedback or self._infer_feedback(trajectory, step.observation, state)
        attribution = self.attributor.attribute(
            trajectory,
            device_feedback=environment_feedback,
            observation_delta=(step.observation, environment_feedback.next_observation),
            user_feedback=environment_feedback.details.get("user_feedback"),
        )
        critique = self.reflection.reflect(step.observation, trajectory, environment_feedback)
        experience = Experience(
            observation=step.observation,
            skill_name=step.route.skill.name,
            decision=trajectory.decision,
            feedback=environment_feedback,
            critique=critique,
            timestamp=trajectory.timestamp,
            attribution=attribution,
        )
        self.memory.add(experience)
        self.library.record_outcome(
            step.route.skill.name,
            success=environment_feedback.success and not trajectory.rejected_actions,
            failure_mode=None if critique.failure_mode == "none" else critique.failure_mode,
        )
        evolution = self.evolution.evolve(self.memory, self.library)
        learning = LearningStep(experience, attribution, critique, evolution)
        self.last_learning = learning
        return learning

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
        )


def build_default_library() -> SkillLibrary:
    """Create the initial ``L_0``; future evolution adds skills without changing it."""

    def cool_precondition(observation: Observation, state: RobotState) -> bool:
        return observation.event_kind == SensorKind.TEMPERATURE.value and observation.temperature_c > 27

    def cool_policy(observation: Observation, state: RobotState) -> Decision:
        reason = f"室温 {observation.temperature_c:.1f}°C 偏高"
        action = Action(ActionKind.SET_HVAC, {"target_c": 25}, reason)
        return Decision((action,), f"{reason}，我把空调设为 25°C。", "cooling skill")

    def warm_precondition(observation: Observation, state: RobotState) -> bool:
        return observation.event_kind == SensorKind.TEMPERATURE.value and observation.temperature_c < 18

    def warm_policy(observation: Observation, state: RobotState) -> Decision:
        reason = f"室温 {observation.temperature_c:.1f}°C 偏低"
        action = Action(ActionKind.SET_HVAC, {"target_c": 22}, reason)
        return Decision((action,), f"{reason}，我把空调设为 22°C。", "warming skill")

    def touch_precondition(observation: Observation, state: RobotState) -> bool:
        return observation.event_kind == SensorKind.TOUCH.value and observation.touch > 0

    def touch_policy(observation: Observation, state: RobotState) -> Decision:
        action = Action(ActionKind.SPEAK, {"text": "嘿，我在呢。"}, "detected friendly touch")
        return Decision((action,), "嘿，我在呢。", "touch interaction")

    def air_precondition(observation: Observation, state: RobotState) -> bool:
        value = observation.context.get("reading_value", 0)
        return (
            observation.event_kind == SensorKind.AIR_QUALITY.value
            and isinstance(value, (int, float))
            and float(value) > 100
        )

    def air_policy(observation: Observation, state: RobotState) -> Decision:
        action = Action(
            ActionKind.NOTIFY,
            {"level": "warning", "message": "房间空气质量需要关注"},
            "air quality threshold exceeded",
        )
        return Decision((action,), "空气质量有些差，建议通风。", "room health alert")

    def conversation_precondition(observation: Observation, state: RobotState) -> bool:
        return observation.user_signal > 0

    def conversation_policy(observation: Observation, state: RobotState) -> Decision:
        text = str(observation.context.get("user_message", "")).strip()
        return Decision(response=f"我听到了：{text}", rationale="conversation fallback")

    def always(observation: Observation, state: RobotState) -> bool:
        return True

    def idle_policy(observation: Observation, state: RobotState) -> Decision:
        return Decision(rationale="reading recorded; no action required")

    return SkillLibrary(
        (
            Skill("cool_room", cool_policy, cool_precondition, ("hvac:25",), (4.0, 0.5, 0, 0), 0),
            Skill("warm_room", warm_policy, warm_precondition, ("hvac:22",), (-4.0, 0, 0, 0), 0),
            Skill("air_warning", air_policy, air_precondition, ("notify:user",), (0, 0, 0, 0), 0),
            Skill("touch_response", touch_policy, touch_precondition, ("speak",), (0, 0, 4.0, 0), 1),
            Skill(
                "conversation",
                conversation_policy,
                conversation_precondition,
                ("respond:user",),
                (0, 0, 0, 4.0),
                2,
            ),
            Skill("idle", idle_policy, always, ("observe",), (0, 0, 0, 0), 99),
        )
    )


def _numeric(value: object, default: float) -> float:
    return float(value) if isinstance(value, (int, float)) else default
