"""Rule-bounded high-level planning with uncertainty-aware information gathering."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .events import RuntimeEvent
from .intent import Intent, IntentType
from .intent_analyzer import IntentAnalyzer
from .perception import BeliefBand, BeliefUpdate


class InformationAction(str, Enum):
    MOVE_CLOSER = "move_closer"
    INSPECT_AGAIN = "inspect_again"
    ASK_USER = "ask_user"
    WAIT = "wait"


@dataclass(frozen=True, slots=True)
class PlannerDecision:
    intent: Intent
    rationale: str
    information_action: InformationAction | None = None


class UncertaintyAwarePlanner:
    """Choose a high-level skill; never emit low-level controller commands."""

    def __init__(self, analyzer: IntentAnalyzer | None = None) -> None:
        self.analyzer = analyzer or IntentAnalyzer()

    def decide(self, event: RuntimeEvent, belief: BeliefUpdate) -> PlannerDecision:
        if belief.after.hypothesis != "fire" or not event.event_type.startswith("scene."):
            return PlannerDecision(self.analyzer.analyze(event), "rule-first intent routing")

        if belief.after.band is BeliefBand.CONFIDENT:
            intent = Intent(
                IntentType.SAFETY_FIRE_ALERT,
                event,
                dict(event.payload),
                priority=100,
                confidence=belief.after.probability,
            )
            return PlannerDecision(intent, "fire belief crossed the confirmation threshold")

        if belief.after.band is BeliefBand.UNCERTAIN:
            action = self._information_action(event)
            intent = self._information_intent(action, event, belief)
            return PlannerDecision(
                intent,
                "high-risk perception is uncertain; gather information before alerting",
                action,
            )

        intent = Intent(
            IntentType.WAIT,
            event,
            {"hypothesis": "fire", "belief": belief.after.probability},
            priority=5,
            confidence=belief.after.probability,
        )
        return PlannerDecision(
            intent,
            "fire belief is below the uncertain interval; wait for new evidence",
            InformationAction.WAIT,
        )

    @staticmethod
    def _information_action(event: RuntimeEvent) -> InformationAction:
        requested = str(event.payload.get("information_action", "")).strip()
        if requested:
            try:
                return InformationAction(requested)
            except ValueError:
                pass
        if event.payload.get("can_reinspect", True) is True:
            return InformationAction.INSPECT_AGAIN
        if event.payload.get("target_too_far") is True:
            return InformationAction.MOVE_CLOSER
        if event.payload.get("user_available", True) is True:
            return InformationAction.ASK_USER
        return InformationAction.WAIT

    @staticmethod
    def _information_intent(
        action: InformationAction,
        event: RuntimeEvent,
        belief: BeliefUpdate,
    ) -> Intent:
        common = {
            **event.payload,
            "hypothesis": "fire",
            "belief_before": belief.before.probability,
            "belief_after": belief.after.probability,
        }
        mapping = {
            InformationAction.INSPECT_AGAIN: (IntentType.SCENE_INSPECT_AGAIN, 90),
            InformationAction.MOVE_CLOSER: (IntentType.SCENE_MOVE_CLOSER, 70),
            InformationAction.ASK_USER: (IntentType.SCENE_ASK_USER, 80),
            InformationAction.WAIT: (IntentType.WAIT, 5),
        }
        intent_type, priority = mapping[action]
        if action is InformationAction.MOVE_CLOSER:
            common.setdefault("distance_m", 0.3)
            common.setdefault("speed", 0.25)
            common.setdefault("obstacle", event.payload.get("obstacle", False))
            common.setdefault("confirmed", event.payload.get("confirmed", False))
        elif action is InformationAction.ASK_USER:
            common.setdefault("text", "我不确定是否有明火，请你确认一下当前环境。")
        return Intent(
            intent_type,
            event,
            common,
            priority=priority,
            confidence=belief.after.probability,
            requires_confirmation=action is InformationAction.MOVE_CLOSER,
        )
