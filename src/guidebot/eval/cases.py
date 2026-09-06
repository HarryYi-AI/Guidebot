"""Deterministic held-out cases for the operational Agent runtime."""

from __future__ import annotations

from dataclasses import dataclass

from guidebot.events import Event
from guidebot.intent import IntentType


@dataclass(frozen=True, slots=True)
class EvalCase:
    name: str
    event: Event
    expected_intent: IntentType
    expected_skill_id: str | None
    expected_safety_allowed: bool | None = True


def core_eval_cases() -> tuple[EvalCase, ...]:
    """Cases stay outside training/evolution memory and act as a held-out suite."""

    return (
        EvalCase(
            "fire_preempts",
            Event("scene.detected", "eval", {"label": "fire", "summary": "发现明火"}),
            IntentType.SAFETY_FIRE_ALERT,
            "scene.fire_alert",
        ),
        EvalCase(
            "obstacle_stops_motion",
            Event("ultrasonic.obstacle", "eval", {"obstacle": True, "distance_mm": 120}),
            IntentType.MOBILITY_STOP,
            "mobility.stop",
        ),
        EvalCase(
            "sedentary_reminder",
            Event("health.detected", "eval", {"label": "sedentary", "sedentary": True}),
            IntentType.HEALTH_SEDENTARY,
            "health.sedentary",
        ),
        EvalCase(
            "alarm_from_text",
            Event("user.text", "eval", {"text": "五分钟后提醒我喝水"}),
            IntentType.SET_ALARM,
            "alarm.set",
        ),
        EvalCase(
            "climate_is_suggestion",
            Event("climate.detected", "eval", {"temperature_c": 29.2, "humidity": 75}),
            IntentType.CLIMATE_COMFORT,
            "climate.comfort",
        ),
        EvalCase(
            "ad_creative_routes_to_reviewable_skill",
            Event(
                "user.text",
                "eval",
                {
                    "text": "为 Guidebot 生成一张广告海报",
                    "product": "Guidebot",
                    "audience": "年轻家庭",
                },
            ),
            IntentType.GENERATE_AD_CREATIVE,
            "ad.generate_brief",
        ),
    )
