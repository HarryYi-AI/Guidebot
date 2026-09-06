from __future__ import annotations

from guidebot.agents import EmbodiedPlannerAgent, ScriptedPlannerClient, SkillEvolutionAgent
from guidebot.devices import SimulatedDevice
from guidebot.hub import GuidebotHub
from guidebot.models import (
    Action,
    ActionKind,
    Decision,
    Event,
    Reading,
    RobotState,
    SensorKind,
    Trajectory,
)
from guidebot.observation import Observation
from guidebot.policy_evolution import PolicyEvolution
from guidebot.reflection import EnvironmentFeedback
from guidebot.self_evolving import build_default_library


async def test_embodied_planner_agent_proposes_structured_action() -> None:
    client = ScriptedPlannerClient(
        (
            {
                "response": "我会温和降温到 25 度。",
                "rationale": "user requested gentle cooling",
                "actions": [
                    {
                        "kind": "set_hvac",
                        "parameters": {"target_c": 25},
                        "reason": "temperature comfort request",
                    }
                ],
            },
        )
    )
    device = SimulatedDevice()
    hub = GuidebotHub(device, agent=EmbodiedPlannerAgent(client))

    await hub.start()
    trajectory = await hub.say("有点热，但别太冷")
    await hub.stop()

    assert trajectory.accepted_actions[0].kind is ActionKind.SET_HVAC
    assert trajectory.accepted_actions[0].parameters["target_c"] == 25
    assert not trajectory.rejected_actions
    assert "Guidebot EmbodiedPlannerAgent" in client.prompts[0]


async def test_embodied_planner_agent_still_goes_through_safety_gate() -> None:
    client = ScriptedPlannerClient(
        (
            {
                "response": "我尝试设置一个过低温度。",
                "rationale": "unsafe model proposal",
                "actions": [
                    {
                        "kind": "set_hvac",
                        "parameters": {"target_c": 10},
                        "reason": "unsafe target from planner",
                    }
                ],
            },
        )
    )
    device = SimulatedDevice()
    hub = GuidebotHub(device, agent=EmbodiedPlannerAgent(client))

    await hub.start()
    trajectory = await hub.say("把空调开到最低")
    await hub.stop()

    assert not trajectory.accepted_actions
    assert trajectory.rejected_actions[0].kind is ActionKind.SET_HVAC
    assert not device.executed_actions


async def test_llm_manager_selects_option_and_fixed_compiler_builds_action() -> None:
    client = ScriptedPlannerClient(
        (
            {
                "response": "我靠近一点看看。",
                "rationale": "inspect the nearby area",
                "steps": [
                    {
                        "option": "move_closer",
                        "parameters": {"distance_m": 0.4, "speed": 0.99},
                        "reason": "camera target is too far",
                    }
                ],
            },
        )
    )
    device = SimulatedDevice()
    planner = EmbodiedPlannerAgent(client)
    hub = GuidebotHub(device, agent=planner)

    await hub.start()
    hub.state.update(Reading(SensorKind.DISTANCE, 1.0, "m", "test"))
    trajectory = await hub.say("靠近一点看看")
    await hub.stop()

    assert planner.last_plan is not None
    assert planner.last_plan.steps[0].option.value == "move_closer"
    assert trajectory.accepted_actions[0].kind is ActionKind.MOVE
    assert trajectory.accepted_actions[0].parameters["speed"] == 0.25
    assert trajectory.accepted_actions[0].requested_by == "embodied_planner_manager"
    assert trajectory.decision.metadata["skill_options"] == ["move_closer"]
    assert "Do not output navigation waypoints, wheel speeds" in client.prompts[0]


async def test_high_level_turn_ac_option_cannot_bypass_safety() -> None:
    client = ScriptedPlannerClient(
        (
            {
                "rationale": "unsafe proposal",
                "steps": [{"option": "turn_ac", "parameters": {"target_c": 10}}],
            },
        )
    )
    device = SimulatedDevice()
    hub = GuidebotHub(device, agent=EmbodiedPlannerAgent(client))

    await hub.start()
    trajectory = await hub.say("设为十度")
    await hub.stop()

    assert not trajectory.accepted_actions
    assert trajectory.rejected_actions[0].kind is ActionKind.SET_HVAC
    assert not device.executed_actions


async def test_move_option_fails_closed_without_distance_observation() -> None:
    client = ScriptedPlannerClient(
        (
            {
                "steps": [{"option": "move_closer", "parameters": {"distance_m": 0.3}}],
            },
        )
    )
    device = SimulatedDevice()
    hub = GuidebotHub(device, agent=EmbodiedPlannerAgent(client))

    await hub.start()
    trajectory = await hub.say("靠近")
    await hub.stop()

    assert not trajectory.accepted_actions
    assert trajectory.rejected_actions[0].kind is ActionKind.MOVE
    assert not device.executed_actions


def test_skill_evolution_agent_turns_failure_into_candidate_skill() -> None:
    library = build_default_library()
    agent = SkillEvolutionAgent(library, evolution=PolicyEvolution(failure_threshold=1))
    observation = Observation(22, 50, 0, 1, "user.message")
    action = Action(ActionKind.SPEAK, {"text": "我不确定。"}, "bad conversation response")
    trajectory = Trajectory(
        Event("user.message", "我有点不舒服"),
        Decision((action,), "我不确定。", "conversation fallback"),
        (action,),
        (),
    )
    feedback = EnvironmentFeedback(
        -1.0,
        False,
        observation,
        "misread user comfort intent",
        {"failure_mode": "user_signal_misread"},
    )

    report = agent.observe(
        observation=observation,
        trajectory=trajectory,
        state=RobotState(),
        skill_name="conversation",
        feedback=feedback,
    )

    assert report.memory_size == 1
    assert report.attribution.should_evolve_skill
    assert report.critique.failure_mode == "user_signal_misread"
    assert report.outcome.triggered
    assert report.outcome.added
    assert report.outcome.generated_skill is not None
    assert report.outcome.generated_skill.generated
