"""Adapter that turns transcribed speech into the existing Guidebot user-message path."""

from __future__ import annotations

from guidebot.hub import GuidebotHub


class GuidebotDialogueAdapter:
    def __init__(self, hub: GuidebotHub) -> None:
        self.hub = hub

    async def respond(self, text: str) -> str:
        trajectory = await self.hub.say(text)
        return trajectory.decision.response or "我已经收到。"

