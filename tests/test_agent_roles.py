from __future__ import annotations

from guidebot.agents import EmbodiedPlannerAgent, ScriptedPlannerClient, SkillEvolutionAgent
from guidebot.devices import SimulatedDevice
from guidebot.hub import GuidebotHub
from guidebot.models import Action, ActionKind, Decision, Event, RobotState, Trajectory
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
