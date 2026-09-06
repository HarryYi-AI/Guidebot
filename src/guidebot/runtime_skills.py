"""Runtime skill registry for event-driven Guidebot tools.

The self-evolving ``SkillLibrary`` is still used for Observation -> Decision
policies. This registry is the operational layer used by the resident runtime:
Intent -> RuntimeSkill -> Task(module.action).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from .intent import IntentType
from .tooling import ErrorSemantics, ToolContract, ToolPermission


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


def _base_result_schema() -> Mapping[str, tuple[str, ...]]:
    return {"module": ("string",), "action": ("string",)}


@dataclass(frozen=True, slots=True)
class RuntimeSkill:
    skill_id: str
    intent_type: IntentType
    target_module: str
    action: str
    description: str
    interruptible: bool | None = None
    contract: ToolContract | None = None
    name: str | None = None
    precondition: str = "matching intent and a valid ToolContract payload"
    result_schema: Mapping[str, tuple[str, ...]] = field(default_factory=_base_result_schema)
    risk_level: RiskLevel = RiskLevel.LOW

    def __post_init__(self) -> None:
        if self.name is None:
            object.__setattr__(self, "name", self.skill_id)

    @property
    def input_schema(self) -> Mapping[str, tuple[str, ...]]:
        return self.contract.input_schema if self.contract is not None else {}

    def execute(self, modules: Mapping[str, Any], task: Any) -> dict[str, Any]:
        """Execute through the registered module, never through model-generated code."""
        module = modules[self.target_module]
        result = module.handle_task(task)
        if not isinstance(result, dict):
            raise TypeError("skill result must be a dictionary observation")
        result_errors = ToolContract(
            f"{self.skill_id}.result",
            ToolPermission.READ_ONLY,
            input_schema=self.result_schema,
        ).validate(result)
        if result_errors:
            raise ValueError("; ".join(result_errors))
        return result


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
        if skill.contract is not None and skill.contract.name != skill.skill_id:
            raise ValueError("tool contract name must match runtime skill id")
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


class SkillRouter(RuntimeSkillRegistry):
    """Explicit high-level intent-to-skill routing boundary."""


def build_default_runtime_skills() -> RuntimeSkillRegistry:
    return SkillRouter(
        (
            _skill("voice.chat", IntentType.CHAT, "voice_chat", "chat", "General chat"),
            _skill(
                "ad.generate_brief",
                IntentType.GENERATE_AD_CREATIVE,
                "ad_creative",
                "generate_brief",
                "Generate a reviewable advertising creative brief",
                permission=ToolPermission.READ_ONLY,
                schema={
                    "text": ("string",),
                    "product": ("string",),
                    "audience": ("string",),
                    "goal": ("string",),
                },
                timeout_s=30.0,
            ),
            _skill(
                "alarm.set",
                IntentType.SET_ALARM,
                "alarm_timer",
                "set_alarm",
                "Set alarm",
                permission=ToolPermission.WRITE_STATE,
                schema={"time": ("string", "null"), "text": ("string",)},
            ),
            _skill(
                "alarm.cancel",
                IntentType.CANCEL_ALARM,
                "alarm_timer",
                "cancel_alarm",
                "Cancel alarm",
                permission=ToolPermission.WRITE_STATE,
            ),
            _skill(
                "alarm.remind",
                IntentType.TIMER_REMINDER,
                "alarm_timer",
                "remind",
                "Alarm reminder, optionally backed by a hardware alarm command",
                interruptible=False,
                permission=ToolPermission.EXTERNAL_SIDE_EFFECT,
                schema={"text": ("string",)},
            ),
            _skill(
                "scene.fire_alert",
                IntentType.SAFETY_FIRE_ALERT,
                "scene_monitor",
                "fire_alert",
                "Fire or smoke alert",
                interruptible=False,
                permission=ToolPermission.SAFETY_CRITICAL,
                precondition="fire belief is at or above the confirmation threshold",
                risk_level=RiskLevel.CRITICAL,
            ),
            _skill(
                "scene.fall_alert",
                IntentType.SAFETY_FALL_ALERT,
                "scene_monitor",
                "fall_alert",
                "Fall alert",
                interruptible=False,
                permission=ToolPermission.SAFETY_CRITICAL,
                risk_level=RiskLevel.CRITICAL,
            ),
            _skill(
                "scene.abnormal_alert",
                IntentType.SAFETY_SCENE_ALERT,
                "scene_monitor",
                "scene_alert",
                "Generic scene anomaly alert",
                interruptible=False,
                permission=ToolPermission.SAFETY_CRITICAL,
                risk_level=RiskLevel.CRITICAL,
            ),
            _skill(
                "scene.inspect_again",
                IntentType.SCENE_INSPECT_AGAIN,
                "scene_monitor",
                "inspect_again",
                "Capture another scene observation to reduce fire uncertainty",
                interruptible=True,
                permission=ToolPermission.READ_ONLY,
                precondition="fire belief is inside the uncertain interval and camera is available",
                result_schema={
                    "module": ("string",),
                    "action": ("string",),
                    "rescan_requested": ("boolean",),
                },
            ),
            _skill(
                "scene.ask_user",
                IntentType.SCENE_ASK_USER,
                "voice_chat",
                "ask_user",
                "Ask the user for evidence when visual fire belief is uncertain",
                interruptible=True,
                schema={"text": ("string",)},
                precondition="visual evidence is uncertain and a user is reachable",
            ),
            _skill(
                "mobility.move_closer",
                IntentType.SCENE_MOVE_CLOSER,
                "mobility",
                "move_forward",
                "Gather a closer observation using the fixed navigation controller",
                permission=ToolPermission.PHYSICAL,
                schema={
                    "distance_m": ("number",),
                    "speed": ("number",),
                    "obstacle": ("boolean",),
                    "confirmed": ("boolean",),
                },
                required_fields=("distance_m", "speed", "obstacle", "confirmed"),
                precondition="explicit confirmation and a fresh obstacle-free distance reading",
                risk_level=RiskLevel.HIGH,
            ),
            _skill(
                "planner.wait",
                IntentType.WAIT,
                "scene_monitor",
                "wait",
                "Wait for a safer or more informative observation",
                permission=ToolPermission.READ_ONLY,
                precondition="available evidence does not justify a side effect",
            ),
            _skill(
                "health.sedentary",
                IntentType.HEALTH_SEDENTARY,
                "health_monitor",
                "sedentary_alert",
                "Sedentary reminder",
                precondition="sedentary detector emitted a positive observation",
            ),
            _skill(
                "health.fatigue",
                IntentType.HEALTH_FATIGUE,
                "health_monitor",
                "fatigue_alert",
                "Fatigue reminder",
            ),
            _skill(
                "mobility.stop",
                IntentType.MOBILITY_STOP,
                "mobility",
                "stop",
                "Emergency stop",
                interruptible=False,
                permission=ToolPermission.SAFETY_CRITICAL,
                precondition="obstacle, stop command, or emergency preemption was observed",
                risk_level=RiskLevel.CRITICAL,
            ),
            _skill(
                "mobility.forward",
                IntentType.MOBILITY_MOVE,
                "mobility",
                "move_forward",
                "Move using fixed navigation and motor PID",
                permission=ToolPermission.PHYSICAL,
                schema={
                    "action": ("string",),
                    "speed": ("number",),
                    "obstacle": ("boolean",),
                    "confirmed": ("boolean",),
                },
                required_fields=("action", "speed", "obstacle", "confirmed"),
                precondition="explicit confirmation and obstacle=False",
                risk_level=RiskLevel.HIGH,
            ),
            _skill(
                "climate.comfort",
                IntentType.CLIMATE_COMFORT,
                "climate_control",
                "suggest_comfort",
                "Climate comfort suggestion",
            ),
            _skill(
                "climate.ac_left_on",
                IntentType.AC_LEFT_ON_ALERT,
                "climate_control",
                "ac_left_on_alert",
                "Air conditioner left-on alert",
            ),
            _skill(
                "pet.interaction",
                IntentType.PET_INTERACTION,
                "voice_chat",
                "pet_interaction",
                "Pet-like social interaction",
            ),
            _skill("voice.unknown", IntentType.UNKNOWN, "voice_chat", "chat", "Fallback chat"),
        )
    )


def _skill(
    skill_id: str,
    intent_type: IntentType,
    target_module: str,
    action: str,
    description: str,
    *,
    interruptible: bool | None = None,
    permission: ToolPermission = ToolPermission.COMMUNICATE,
    schema: dict[str, tuple[str, ...]] | None = None,
    required_fields: tuple[str, ...] = (),
    timeout_s: float = 5.0,
    precondition: str = "matching intent and a valid ToolContract payload",
    result_schema: Mapping[str, tuple[str, ...]] | None = None,
    risk_level: RiskLevel = RiskLevel.LOW,
) -> RuntimeSkill:
    contract = ToolContract(
        skill_id,
        permission,
        input_schema=schema or {},
        required_fields=required_fields,
        timeout_s=timeout_s,
        error_semantics=ErrorSemantics.FAIL_CLOSED,
        max_attempts=1,
        idempotent=permission is ToolPermission.READ_ONLY,
    )
    return RuntimeSkill(
        skill_id,
        intent_type,
        target_module,
        action,
        description,
        interruptible,
        contract,
        skill_id,
        precondition,
        result_schema or _base_result_schema(),
        risk_level,
    )
