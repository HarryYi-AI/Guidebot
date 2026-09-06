"""Minimal independent critic for tool existence, schema, and obvious safety."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from ..safety import SafetyGate
from ..tools import ToolRegistry, UnknownToolError
from .planner import DecisionType, PlannerDecision


@dataclass(frozen=True, slots=True)
class AgentMessage:
    sender: str
    receiver: str
    message_type: str
    content: str
    trace_id: str


@dataclass(frozen=True, slots=True)
class CriticReview:
    approved: bool
    feedback: tuple[str, ...]
    message: AgentMessage


class CriticAgent:
    def __init__(self, safety_gate: SafetyGate | None = None) -> None:
        self.safety_gate = safety_gate or SafetyGate()

    def review(
        self,
        decision: PlannerDecision,
        context: Any,
        registry: ToolRegistry,
        *,
        trace_id: str | None = None,
    ) -> CriticReview:
        problems: list[str] = []
        if decision.type is DecisionType.TOOL_CALL:
            try:
                tool = registry.get(decision.tool_name or "")
            except UnknownToolError:
                problems.append(f"unknown tool: {decision.tool_name}")
            else:
                problems.extend(registry.validate(tool.name, decision.arguments))
                safety = self.safety_gate.evaluate_tool_call(
                    tool.name,
                    decision.arguments,
                    known_physical_tool=tool.physical,
                )
                if not safety.allowed:
                    problems.append(safety.reason)
        else:
            missing = set(getattr(context, "required_tools", ())) - set(
                getattr(context, "completed_tools", ())
            )
            if missing:
                problems.append(f"cannot finish; missing critical tools: {sorted(missing)}")

        content = "approved" if not problems else "; ".join(problems)
        message = AgentMessage(
            "critic",
            "planner",
            "approve" if not problems else "revise",
            content,
            trace_id or uuid4().hex,
        )
        return CriticReview(not problems, tuple(problems), message)
