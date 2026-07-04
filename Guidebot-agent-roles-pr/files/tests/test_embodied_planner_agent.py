from __future__ import annotations

from guidebot.agents import EmbodiedPlannerAgent, ScriptedPlannerClient
from guidebot.devices import SimulatedDevice
from guidebot.hub import GuidebotHub
from guidebot.models import ActionKind


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
