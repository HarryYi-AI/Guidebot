"""Serializable state models for the bounded ReAct-style AgentLoop."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

from ..models import utc_now
from ..planning.critic import AgentMessage, CriticReview
from ..planning.planner import PlannerDecision
from ..safety import SafetyResult
from ..tools import ToolResult


class RunStatus(str, Enum):
    FINISHED = "finished"
    MAX_STEPS = "max_steps"
    CRITIC_REJECTED = "critic_rejected"
    PLANNER_ERROR = "planner_error"


@dataclass(frozen=True, slots=True)
class Observation:
    type: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentLoopStep:
    index: int
    decision: PlannerDecision
    critic: CriticReview
    safety: SafetyResult | None
    tool_result: ToolResult | None
    observation: Observation
    latency_ms: float


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    goal: str
    status: RunStatus
    trajectory: tuple[AgentLoopStep, ...]
    final_answer: str | None
    messages: tuple[AgentMessage, ...] = ()
    trace_id: str = field(default_factory=lambda: uuid4().hex)
    finished_at: str = field(default_factory=lambda: utc_now().isoformat())
