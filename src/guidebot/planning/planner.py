"""Strict JSON planner boundary for the bounded AgentLoop."""

from __future__ import annotations

import json
import inspect
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from ..logbook import to_jsonable


class DecisionType(str, Enum):
    TOOL_CALL = "tool_call"
    FINISH = "finish"


@dataclass(frozen=True, slots=True)
class PlannerDecision:
    type: DecisionType
    reason: str
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    final_answer: str | None = None

    @classmethod
    def parse(cls, raw: str | Mapping[str, Any]) -> PlannerDecision:
        try:
            decoded = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError as exc:
            raise ValueError("planner output must be strict JSON") from exc
        if not isinstance(decoded, Mapping):
            raise ValueError("planner output must be a JSON object")
        payload = dict(decoded)
        allowed = {"type", "reason", "tool_name", "arguments", "final_answer"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"planner output contains unknown fields: {sorted(unknown)}")
        try:
            decision_type = DecisionType(payload["type"])
        except (KeyError, ValueError) as exc:
            raise ValueError("planner type must be tool_call or finish") from exc
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("planner reason must be a non-empty string")
        tool_name = payload.get("tool_name")
        arguments = payload.get("arguments", {})
        final_answer = payload.get("final_answer")
        if not isinstance(arguments, dict):
            raise ValueError("planner arguments must be an object")
        if decision_type is DecisionType.TOOL_CALL and not isinstance(tool_name, str):
            raise ValueError("tool_call requires tool_name")
        if decision_type is DecisionType.FINISH and not isinstance(final_answer, str):
            raise ValueError("finish requires final_answer")
        if final_answer is not None and not isinstance(final_answer, str):
            raise ValueError("final_answer must be a string or null")
        return cls(decision_type, reason, tool_name, arguments, final_answer)


class Planner(Protocol):
    async def decide(self, context: Any) -> PlannerDecision: ...


PlannerBackend = Callable[[str], str | Awaitable[str]]


class PlannerAgent:
    """LLM-facing planner; the backend can only return a parsed high-level decision."""

    def __init__(self, backend: PlannerBackend) -> None:
        self.backend = backend

    async def decide(self, context: Any) -> PlannerDecision:
        prompt = json.dumps(_context_payload(context), ensure_ascii=False, sort_keys=True)
        raw = self.backend(prompt)
        if inspect.isawaitable(raw):
            raw = await raw
        return PlannerDecision.parse(raw)


class MockPlanner:
    """Deterministic planner with no model/API dependency."""

    def __init__(self, decisions: Sequence[PlannerDecision | Mapping[str, Any]]) -> None:
        self._decisions = deque(
            item if isinstance(item, PlannerDecision) else PlannerDecision.parse(item)
            for item in decisions
        )
        self.contexts: list[Any] = []

    async def decide(self, context: Any) -> PlannerDecision:
        self.contexts.append(context)
        if not self._decisions:
            raise RuntimeError("mock planner decision sequence exhausted")
        return self._decisions.popleft()


def _context_payload(context: Any) -> dict[str, Any]:
    payload = getattr(context, "planner_payload", None)
    if callable(payload):
        return payload()
    return to_jsonable(
        {
            "goal": context.goal,
            "latest_observation": context.latest_observation,
            "memory": context.budgeted_context,
            "available_tools": context.available_tools,
            "required_tools": context.required_tools,
            "completed_tools": context.completed_tools,
        }
    )
