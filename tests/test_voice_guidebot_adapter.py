from __future__ import annotations

from guidebot.devices import SimulatedDevice
from guidebot.hub import GuidebotHub
from guidebot.voice.guidebot_adapter import GuidebotDialogueAdapter


async def test_voice_text_uses_existing_guidebot_user_message_path() -> None:
    hub = GuidebotHub(SimulatedDevice())
    await hub.start()

    response = await GuidebotDialogueAdapter(hub).respond("今天天气怎么样")

    assert response == "我听到了：今天天气怎么样"
    assert hub.trajectories[-1].trigger.topic == "user.message"
    assert len(hub.agent.memory) == 1

