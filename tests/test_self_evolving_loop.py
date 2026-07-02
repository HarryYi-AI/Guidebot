from __future__ import annotations

from guidebot.devices import SimulatedDevice
from guidebot.hub import GuidebotHub
from guidebot.models import Action, ActionKind, Decision, Reading, RobotState, SensorKind
from guidebot.policy_evolution import PolicyEvolution
from guidebot.self_evolving import SelfEvolvingAgent, build_default_library
from guidebot.skills import Skill


async def test_hub_runs_complete_observe_route_act_reflect_remember_loop() -> None:
    device = SimulatedDevice()
    agent = SelfEvolvingAgent()
    hub = GuidebotHub(device, agent=agent)
    await hub.start()

    trajectory = await hub.ingest(Reading(SensorKind.TEMPERATURE, 30, "°C", "test"))

    assert trajectory.accepted_actions[0].kind is ActionKind.SET_HVAC
    assert agent.last_step is not None
    assert agent.last_step.route.skill.name == "cool_room"
    assert agent.last_learning is not None
    assert agent.last_learning.critique.failure_mode == "none"
    assert len(agent.memory) == 1


async def test_repeated_safety_failures_do_not_rewrite_skills() -> None:
    def precondition(observation, state: RobotState) -> bool:
        return observation.event_kind == SensorKind.LIGHT.value

    def unsafe_policy(observation, state: RobotState) -> Decision:
        action = Action(ActionKind.SET_HVAC, {"target_c": 99}, "unsafe generated plan")
        return Decision((action,), rationale="force safety rejection")

    library = build_default_library()
    library.add(Skill("unsafe", unsafe_policy, precondition, (), (0, 0, 0, 0), level=0))
    agent = SelfEvolvingAgent(library=library, evolution=PolicyEvolution(failure_threshold=2))
    device = SimulatedDevice()
    hub = GuidebotHub(device, agent=agent)
    await hub.start()

    await hub.ingest(Reading(SensorKind.LIGHT, 50, "%", "test"))
    await hub.ingest(Reading(SensorKind.LIGHT, 50, "%", "test"))

    assert len(agent.memory.failures("safety_rejection")) == 2
    assert "recovery_safety_rejection" not in agent.library.names
    assert not device.executed_actions
