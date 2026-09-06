"""Deterministic policy gate between probabilistic agents and the physical world."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from .models import Action, ActionKind, RobotState, SensorKind, utc_now
from .scheduler import Task
from .tooling import ToolPermission


@dataclass(frozen=True, slots=True)
class SafetyResult:
    allowed: bool
    reason: str


class SafetyPolicy:
    """Hard limits are code/config, never an evolvable skill."""

    def __init__(self, min_hvac_c: float = 16.0, max_hvac_c: float = 30.0) -> None:
        self.min_hvac_c = min_hvac_c
        self.max_hvac_c = max_hvac_c
        self.allowed_actions = frozenset(ActionKind)

    def evaluate(self, action: Action, state: RobotState) -> SafetyResult:
        if action.kind not in self.allowed_actions:
            return SafetyResult(False, "action type is not allow-listed")

        if action.kind is ActionKind.SET_HVAC:
            target = action.parameters.get("target_c")
            if not isinstance(target, (int, float)):
                return SafetyResult(False, "HVAC target must be numeric")
            if not self.min_hvac_c <= float(target) <= self.max_hvac_c:
                return SafetyResult(False, "HVAC target is outside the hard safety range")

        if action.kind is ActionKind.MOVE:
            speed = action.parameters.get("speed", 0)
            if not isinstance(speed, (int, float)) or not 0 <= float(speed) <= 1:
                return SafetyResult(False, "motion speed must be within 0..1")
            if action.parameters.get("direction") == "forward":
                distance = state.readings.get(SensorKind.DISTANCE)
                if distance is None or (utc_now() - distance.timestamp).total_seconds() > 2:
                    return SafetyResult(False, "forward motion requires fresh distance observation")
                if not isinstance(distance.value, (int, float)):
                    return SafetyResult(False, "distance observation must be numeric")
                distance_m = float(distance.value)
                if distance.unit.casefold() in {"mm", "millimeter", "millimeters"}:
                    distance_m /= 1_000
                elif distance.unit.casefold() in {"cm", "centimeter", "centimeters"}:
                    distance_m /= 100
                if distance_m <= 0.2:
                    return SafetyResult(False, "obstacle is too close for forward motion")

        if action.kind is ActionKind.SET_ALARM:
            alarm_time = action.parameters.get("time")
            if not isinstance(alarm_time, str) or not alarm_time.strip():
                return SafetyResult(False, "alarm time must be a non-empty string")

        return SafetyResult(True, "allowed")


@dataclass(slots=True)
class RuntimeSafetyState:
    obstacle: bool = False
    active_safety_alert: bool = False
    last_climate_action_at: datetime | None = None


class SafetyGate:
    """Safety gate for runtime tasks before module execution."""

    def __init__(
        self,
        *,
        min_climate_c: float = 16.0,
        max_climate_c: float = 30.0,
        climate_min_interval: timedelta = timedelta(minutes=10),
    ) -> None:
        self.min_climate_c = min_climate_c
        self.max_climate_c = max_climate_c
        self.climate_min_interval = climate_min_interval

    def evaluate_task(
        self,
        task: Task,
        state: RuntimeSafetyState | None = None,
        *,
        now: datetime | None = None,
    ) -> SafetyResult:
        state = state or RuntimeSafetyState()
        now = now or datetime.now().astimezone()

        if state.active_safety_alert and task.priority < 100:
            return SafetyResult(False, "ordinary task cannot override active safety alert")

        contract = task.tool_contract
        if (
            contract is not None
            and contract.permission is ToolPermission.PHYSICAL
            and task.action != "stop"
            and task.payload.get("confirmed") is not True
        ):
            return SafetyResult(False, "physical action requires explicit confirmation")

        if task.target_module == "mobility":
            if task.action == "move_forward" and state.obstacle:
                return SafetyResult(False, "obstacle detected; move_forward is blocked")
            if task.action == "move_forward" and "obstacle" not in task.payload:
                return SafetyResult(False, "move_forward requires obstacle state")

        if task.target_module == "climate_control":
            target = task.payload.get("target_c")
            if target is not None:
                if not isinstance(target, (int, float)):
                    return SafetyResult(False, "climate target must be numeric")
                if not self.min_climate_c <= float(target) <= self.max_climate_c:
                    return SafetyResult(False, "climate target outside safe range")
            if state.last_climate_action_at is not None:
                if now - state.last_climate_action_at < self.climate_min_interval:
                    return SafetyResult(False, "climate action frequency is limited")

        return SafetyResult(True, "allowed")

    def evaluate_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        known_physical_tool: bool = False,
    ) -> SafetyResult:
        """Fail-closed checks for the interview AgentLoop tool boundary."""
        if tool_name == "stop_robot":
            return SafetyResult(True, "safety stop always allowed")
        if known_physical_tool and tool_name != "move_robot":
            return SafetyResult(False, "unknown physical action is blocked")
        if tool_name != "move_robot":
            return SafetyResult(True, "non-physical tool allowed")
        duration = arguments.get("duration_s")
        if not isinstance(duration, (int, float)) or isinstance(duration, bool):
            return SafetyResult(False, "move duration must be numeric")
        if not 0 < float(duration) <= 10:
            return SafetyResult(False, "move duration must be within 0..10 seconds")
        if arguments.get("obstacle") is not False:
            return SafetyResult(False, "obstacle detected; move_robot is blocked")
        return SafetyResult(True, "fixed mobility controller may execute")

    @staticmethod
    def nighttime_tts_volume(hour: int) -> int:
        return 35 if hour >= 22 or hour < 7 else 70
