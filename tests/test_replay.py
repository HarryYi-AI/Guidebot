import pytest

from guidebot.replay import load_replay, replay_fire_observations
from guidebot.runtime import RunStatus


@pytest.mark.asyncio
async def test_fire_replay_inspects_twice_then_alerts() -> None:
    observations = load_replay("data/replays/fire_verify.json")

    result = await replay_fire_observations(observations)

    assert result.status is RunStatus.FINISHED
    assert [step.decision.tool_name or step.decision.type.value for step in result.trajectory] == [
        "scene_inspect",
        "scene_inspect",
        "speak",
        "finish",
    ]
    assert result.trajectory[0].tool_result is not None
    assert result.trajectory[0].tool_result.data["confidence"] == 0.63
    assert result.trajectory[1].tool_result is not None
    assert result.trajectory[1].tool_result.data["confidence"] == 0.91
