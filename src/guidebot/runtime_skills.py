"""Runtime skill registry for event-driven Guidebot tools.

The self-evolving ``SkillLibrary`` is still used for Observation -> Decision
policies. This registry is the operational layer used by the resident runtime:
Intent -> RuntimeSkill -> Task(module.action).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .intent import IntentType


@dataclass(frozen=True, slots=True)
class RuntimeSkill:
    skill_id: str
    intent_type: IntentType
    target_module: str
    action: str
    description: str
    interruptible: bool | None = None


class RuntimeSkillRegistry:
    """Registry that keeps dispatch extensible without LLM-controlled actions."""

    def __init__(self, skills: Iterable[RuntimeSkill] = ()) -> None:
        self._by_id: dict[str, RuntimeSkill] = {}
        self._by_intent: dict[IntentType, RuntimeSkill] = {}
        for skill in skills:
            self.register(skill)

    def register(self, skill: RuntimeSkill) -> None:
        if skill.skill_id in self._by_id:
            raise ValueError(f"duplicate runtime skill id: {skill.skill_id}")
        self._by_id[skill.skill_id] = skill
        self._by_intent[skill.intent_type] = skill

    def resolve(self, intent_type: IntentType) -> RuntimeSkill:
        try:
            return self._by_intent[intent_type]
        except KeyError as exc:
            raise KeyError(f"no runtime skill registered for intent {intent_type.value}") from exc

    def get(self, skill_id: str) -> RuntimeSkill:
        return self._by_id[skill_id]

    def all(self) -> tuple[RuntimeSkill, ...]:
        return tuple(self._by_id.values())


def build_default_runtime_skills() -> RuntimeSkillRegistry:
    return RuntimeSkillRegistry(
        (
            RuntimeSkill("voice.chat", IntentType.CHAT, "voice_chat", "chat", "General chat"),
            RuntimeSkill("alarm.set", IntentType.SET_ALARM, "alarm_timer", "set_alarm", "Set alarm"),
            RuntimeSkill(
                "alarm.cancel",
                IntentType.CANCEL_ALARM,
                "alarm_timer",
                "cancel_alarm",
                "Cancel alarm",
            ),
            RuntimeSkill(
                "alarm.remind",
                IntentType.TIMER_REMINDER,
                "alarm_timer",
                "remind",
                "Alarm reminder, optionally backed by a hardware alarm command",
                interruptible=False,
            ),
            RuntimeSkill(
                "scene.fire_alert",
                IntentType.SAFETY_FIRE_ALERT,
                "scene_monitor",
                "fire_alert",
                "Fire or smoke alert",
                interruptible=False,
            ),
            RuntimeSkill(
                "scene.fall_alert",
                IntentType.SAFETY_FALL_ALERT,
                "scene_monitor",
                "fall_alert",
                "Fall alert",
                interruptible=False,
            ),
            RuntimeSkill(
                "scene.abnormal_alert",
                IntentType.SAFETY_SCENE_ALERT,
                "scene_monitor",
                "scene_alert",
                "Generic scene anomaly alert",
                interruptible=False,
            ),
            RuntimeSkill(
                "health.sedentary",
                IntentType.HEALTH_SEDENTARY,
                "health_monitor",
                "sedentary_alert",
                "Sedentary reminder",
            ),
            RuntimeSkill(
                "health.fatigue",
                IntentType.HEALTH_FATIGUE,
                "health_monitor",
                "fatigue_alert",
                "Fatigue reminder",
            ),
            RuntimeSkill(
                "mobility.stop",
                IntentType.MOBILITY_STOP,
                "mobility",
                "stop",
                "Emergency stop",
                interruptible=False,
            ),
            RuntimeSkill(
                "climate.comfort",
                IntentType.CLIMATE_COMFORT,
                "climate_control",
                "suggest_comfort",
                "Climate comfort suggestion",
            ),
            RuntimeSkill(
                "climate.ac_left_on",
                IntentType.AC_LEFT_ON_ALERT,
                "climate_control",
                "ac_left_on_alert",
                "Air conditioner left-on alert",
            ),
            RuntimeSkill(
                "pet.interaction",
                IntentType.PET_INTERACTION,
                "voice_chat",
                "pet_interaction",
                "Pet-like social interaction",
            ),
            RuntimeSkill("voice.unknown", IntentType.UNKNOWN, "voice_chat", "chat", "Fallback chat"),
        )
    )
