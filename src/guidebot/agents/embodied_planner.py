"""Cloud-planner agent that turns language and state into safe action proposals."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from guidebot.agent import AdaptiveAgent, Agent
from guidebot.models import Action, ActionKind, Decision, DomainEvent, Reading, RobotState
from guidebot.planning import FixedOptionCompiler, HighLevelPlan, OptionStep, SkillOption


class PlannerClient(Protocol):
    """Minimal interface for an OpenAI-compatible or locally mocked planner."""

    async def complete(self, prompt: str) -> str: ...


class PlannerParseError(ValueError):
    """Raised when the planner response cannot be converted into a Decision."""


@dataclass(slots=True)
class ScriptedPlannerClient:
    """Deterministic planner client for demos and tests."""

    responses: Sequence[str | Mapping[str, Any]]
    prompts: list[str] = field(default_factory=list)
    _index: int = 0

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self._index >= len(self.responses):
            raise RuntimeError("scripted planner has no response left")
        response = self.responses[self._index]
        self._index += 1
        if isinstance(response, str):
            return response
        return json.dumps(response, ensure_ascii=False)


class EmbodiedPlannerAgent:
    """LLM manager that selects options while Guidebot owns control and safety.

    The planner proposes a high-level option such as ``turn_ac`` or
    ``move_closer``; it cannot generate or execute low-level controller commands.
    ``FixedOptionCompiler`` maps the option to a typed action and ``GuidebotHub``
    evaluates it with deterministic ``SafetyPolicy`` before any device adapter
    receives it.
    """

    def __init__(
        self,
        client: PlannerClient | None = None,
        *,
        fallback: Agent | None = None,
        compiler: FixedOptionCompiler | None = None,
    ) -> None:
        self.client = client
        self.fallback = fallback or AdaptiveAgent()
        self.compiler = compiler or FixedOptionCompiler()
        self.last_prompt: str | None = None
        self.last_raw_response: str | None = None
        self.last_plan: HighLevelPlan | None = None

    async def decide(self, event: DomainEvent, state: RobotState) -> Decision:
        if self.client is None:
            return await self.fallback.decide(event, state)

        prompt = self.build_prompt(event, state)
        self.last_prompt = prompt
        try:
            raw_response = await self.client.complete(prompt)
            self.last_raw_response = raw_response
            data = self._load_json_object(raw_response)
            if "steps" in data:
                plan = self.parse_plan(data)
                self.last_plan = plan
                return self.compiler.compile(plan, state)
            return self.parse_decision(data)
        except Exception as error:
            fallback_decision = await self.fallback.decide(event, state)
            rationale = (
                f"embodied planner unavailable: {type(error).__name__}: {error}; "
                f"fallback={fallback_decision.rationale}"
            )
            return Decision(fallback_decision.actions, fallback_decision.response, rationale)

    def build_prompt(self, event: DomainEvent, state: RobotState) -> str:
        payload = {
            "role": "Guidebot EmbodiedPlannerAgent",
            "objective": (
                "Select exactly one next high-level skill option. Do not generate low-level "
                "motor, PID, perception, ASR, or IR controller commands."
            ),
            "output_schema": {
                "response": "short Chinese user-facing response or null",
                "rationale": "brief reason for routing and planning",
                "steps": [
                    {
                        "option": "inspect_room|move_closer|ask_user|turn_ac|alarm|wait",
                        "parameters": "JSON object matching the option schema",
                        "reason": "why this high-level option is useful",
                    }
                ],
            },
            "hard_rules": [
                "Return JSON only.",
                "Do not claim that an action has executed.",
                "Do not modify permissions, safety limits, or device allow-lists.",
                "Do not output navigation waypoints, wheel speeds, PID, YOLO, ASR, or IR commands.",
                "Return at most one option step; observe feedback before planning the next step.",
                "Prefer no action when intent or state is ambiguous.",
            ],
            "option_schema": self._option_schema(),
            "event": self._event_snapshot(event),
            "robot_state": self._state_snapshot(state),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @classmethod
    def parse_plan(cls, payload: str | Mapping[str, Any]) -> HighLevelPlan:
        data = cls._load_json_object(payload)
        step_items = data.get("steps", ()) or ()
        if not isinstance(step_items, Sequence) or isinstance(step_items, (str, bytes)):
            raise PlannerParseError("steps must be a JSON array")
        steps = []
        for item in step_items:
            if not isinstance(item, Mapping):
                raise PlannerParseError("each step must be a JSON object")
            try:
                option = SkillOption(str(item["option"]))
            except (KeyError, ValueError) as error:
                raise PlannerParseError("unknown or missing skill option") from error
            parameters = item.get("parameters", {})
            if not isinstance(parameters, Mapping):
                raise PlannerParseError("option parameters must be a JSON object")
            steps.append(OptionStep(option, dict(parameters), str(item.get("reason", ""))))
        response = data.get("response")
        return HighLevelPlan(
            tuple(steps),
            None if response is None else str(response),
            str(data.get("rationale", "embodied high-level plan")),
        )

    @classmethod
    def parse_decision(cls, payload: str | Mapping[str, Any]) -> Decision:
        data = cls._load_json_object(payload)
        response = data.get("response")
        rationale = str(data.get("rationale", "embodied planner"))
        action_items = data.get("actions", ()) or ()
        if not isinstance(action_items, Sequence) or isinstance(action_items, (str, bytes)):
            raise PlannerParseError("actions must be a JSON array")

        actions = []
        for item in action_items:
            if not isinstance(item, Mapping):
                raise PlannerParseError("each action must be a JSON object")
            actions.append(cls._parse_action(item, rationale))

        return Decision(tuple(actions), None if response is None else str(response), rationale)

    @staticmethod
    def _parse_action(item: Mapping[str, Any], fallback_reason: str) -> Action:
        try:
            kind = ActionKind(str(item["kind"]))
        except (KeyError, ValueError) as error:
            raise PlannerParseError("unknown or missing action kind") from error

        parameters = item.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise PlannerParseError("action parameters must be a JSON object")
        reason = str(item.get("reason", fallback_reason))
        return Action(kind, dict(parameters), reason, requested_by="embodied_planner_agent")

    @staticmethod
    def _load_json_object(payload: str | Mapping[str, Any]) -> Mapping[str, Any]:
        if isinstance(payload, Mapping):
            return payload
        text = payload.strip()
        if not text:
            raise PlannerParseError("empty planner response")
        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise PlannerParseError("planner response does not contain a JSON object")
            text = text[start : end + 1]
        data = json.loads(text)
        if not isinstance(data, Mapping):
            raise PlannerParseError("planner response must be a JSON object")
        return data

    @staticmethod
    def _option_schema() -> Mapping[str, Any]:
        return {
            SkillOption.INSPECT_ROOM.value: {"mode": "safety|general"},
            SkillOption.MOVE_CLOSER.value: {"distance_m": "positive number"},
            SkillOption.ASK_USER.value: {"question": "string"},
            SkillOption.TURN_AC.value: {"target_c": "number proposed to local SafetyPolicy"},
            SkillOption.ALARM.value: {"time": "string", "message": "string"},
            SkillOption.WAIT.value: {"duration_s": "optional number; local scheduler owns timing"},
        }

    @staticmethod
    def _event_snapshot(event: DomainEvent) -> Mapping[str, Any]:
        payload: Any = event.payload
        if isinstance(payload, Reading):
            payload = {
                "kind": payload.kind.value,
                "value": payload.value,
                "unit": payload.unit,
                "source": payload.source,
                "timestamp": payload.timestamp.isoformat(),
            }
        return {
            "id": event.id,
            "topic": event.topic,
            "payload": payload,
            "timestamp": event.timestamp.isoformat(),
        }

    @staticmethod
    def _state_snapshot(state: RobotState) -> Mapping[str, Any]:
        return {
            "health": state.health,
            "hvac_target_c": state.hvac_target_c,
            "last_interaction": (
                state.last_interaction.isoformat() if state.last_interaction else None
            ),
            "readings": {
                kind.value: {
                    "value": reading.value,
                    "unit": reading.unit,
                    "source": reading.source,
                    "timestamp": reading.timestamp.isoformat(),
                }
                for kind, reading in state.readings.items()
            },
        }
