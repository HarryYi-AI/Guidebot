from __future__ import annotations

from guidebot.devices import SimulatedDevice
from guidebot.hub import GuidebotHub
from guidebot.models import Action, ActionKind, Decision, Reading, RobotState, SensorKind
from guidebot.observation import Observation
from guidebot.policy_evolution import PolicyEvolution
from guidebot.reflection import EnvironmentFeedback
from guidebot.self_evolving import SelfEvolvingAgent
from guidebot.skills import Skill, SkillLibrary


def _base_skill() -> Skill:
    def precondition(observation: Observation, state: RobotState) -> bool:
        return observation.event_kind == SensorKind.TEMPERATURE.value

    def policy(observation: Observation, state: RobotState) -> Decision:
        action = Action(ActionKind.SET_HVAC, {"target_c": 28}, "insufficient cooling")
        return Decision((action,))

    return Skill("weak_cooling", policy, precondition, ("hvac:28",), (1, 0, 0, 0))


def _failed_feedback(trajectory, state: RobotState) -> EnvironmentFeedback:
    return EnvironmentFeedback(
        -1,
        False,
        Observation(30, 50, event_kind="temperature"),
        "room remained uncomfortable",
        {"failure_mode": "comfort_failure"},
    )


async def test_repeated_skill_failure_passes_verifier_before_activation() -> None:
    library = SkillLibrary((_base_skill(),))
    evolution = PolicyEvolution(failure_threshold=2)
    agent = SelfEvolvingAgent(library=library, evolution=evolution)
    device = SimulatedDevice()
    hub = GuidebotHub(device, agent=agent, feedback_provider=_failed_feedback)
    await hub.start()

    await hub.ingest(Reading(SensorKind.TEMPERATURE, 30, "°C", "test"))
    await hub.ingest(Reading(SensorKind.TEMPERATURE, 30, "°C", "test"))

    assert "recovery_comfort_failure" in library.names
    assert library.card("recovery_comfort_failure").accepted
    assert agent.last_learning is not None
    assert agent.last_learning.evolution.verification is not None
    assert agent.last_learning.evolution.verification.accepted


async def test_execution_lapse_is_remembered_but_does_not_rewrite_skill() -> None:
    def lapse_feedback(trajectory, state: RobotState) -> EnvironmentFeedback:
        return EnvironmentFeedback(
            -1,
            False,
            Observation(30, 50, event_kind="temperature"),
            details={"execution_lapse": True, "failure_mode": "device_lapse"},
        )

    library = SkillLibrary((_base_skill(),))
    agent = SelfEvolvingAgent(
        library=library,
        evolution=PolicyEvolution(failure_threshold=2),
    )
    hub = GuidebotHub(SimulatedDevice(), agent=agent, feedback_provider=lapse_feedback)
    await hub.start()
    await hub.ingest(Reading(SensorKind.TEMPERATURE, 30, "°C", "test"))
    await hub.ingest(Reading(SensorKind.TEMPERATURE, 30, "°C", "test"))

    assert library.names == ("weak_cooling",)
    assert not agent.evolution.rejected_skill_buffer


class UnsafeSynthesizer:
    def synthesize(self, cluster, library: SkillLibrary) -> Skill:
        def precondition(observation: Observation, state: RobotState) -> bool:
            return True

        def policy(observation: Observation, state: RobotState) -> Decision:
            action = Action(ActionKind.SET_HVAC, {"target_c": 99}, "unsafe candidate")
            return Decision((action,))

        return Skill("unsafe_candidate", policy, precondition, ("hvac:99",), (1, 0, 0, 0))


async def test_rejected_candidate_is_kept_in_rejected_skill_buffer() -> None:
    library = SkillLibrary((_base_skill(),))
    evolution = PolicyEvolution(failure_threshold=1, synthesizer=UnsafeSynthesizer())
    agent = SelfEvolvingAgent(library=library, evolution=evolution)
    hub = GuidebotHub(
        SimulatedDevice(),
        agent=agent,
        feedback_provider=_failed_feedback,
    )
    await hub.start()
    await hub.ingest(Reading(SensorKind.TEMPERATURE, 30, "°C", "test"))

    assert "unsafe_candidate" not in library.names
    assert len(evolution.rejected_skill_buffer) == 1
    assert "safety" in evolution.rejected_skill_buffer[0].reason
