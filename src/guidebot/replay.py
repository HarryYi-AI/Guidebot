"""Lightweight JSON observation replay; this is not a physics simulator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .planning import DecisionType, PlannerDecision
from .runtime import AgentLoop, AgentRunResult
from .tools import MockSceneInspectTool, SpeakTool, ToolRegistry


DEFAULT_FIRE_REPLAY = [
    {"type": "scene_result", "label": "possible_fire", "confidence": 0.63},
    {"type": "scene_result", "label": "fire", "confidence": 0.91},
]


def load_replay(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("replay file must contain a JSON array of observation objects")
    return [dict(item) for item in payload]


class FireReplayPlanner:
    """Deterministic policy used to demonstrate active perception from replay events."""

    def __init__(self, confirmation_threshold: float = 0.85) -> None:
        self.confirmation_threshold = confirmation_threshold
        self.alert_spoken = False

    async def decide(self, context: Any) -> PlannerDecision:
        observation = context.latest_observation
        if observation.type == "tool_result":
            tool_name = observation.data.get("tool_name")
            data = observation.data.get("data", {})
            if tool_name == "scene_inspect":
                confidence = float(data.get("confidence", 0.0))
                label = str(data.get("label", ""))
                if label in {"fire", "possible_fire", "smoke"} and confidence >= self.confirmation_threshold:
                    self.alert_spoken = True
                    return PlannerDecision(
                        DecisionType.TOOL_CALL,
                        "second observation confirms a fire risk",
                        "speak",
                        {"text": "检测到高置信度明火，请立即检查并远离危险区域。"},
                    )
                return PlannerDecision(
                    DecisionType.TOOL_CALL,
                    "fire confidence is uncertain; inspect again before alerting",
                    "scene_inspect",
                    {"mode": "fire_verify"},
                )
            if tool_name == "speak" and self.alert_spoken:
                return PlannerDecision(
                    DecisionType.FINISH,
                    "fire verification and alert are complete",
                    final_answer="已完成明火二次确认并播报告警。",
                )
        return PlannerDecision(
            DecisionType.TOOL_CALL,
            "collect the first scene observation",
            "scene_inspect",
            {"mode": "fire_verify"},
        )


async def replay_fire_observations(observations: list[dict[str, Any]]) -> AgentRunResult:
    registry = ToolRegistry()
    registry.register(MockSceneInspectTool(observations))
    registry.register(SpeakTool())
    loop = AgentLoop(FireReplayPlanner(), registry, max_steps=max(4, len(observations) + 3))
    return await loop.run(
        "验证场景中是否存在明火；证据不足时再次观察，确认后播报警告。",
        required_tools=("scene_inspect", "speak"),
    )
