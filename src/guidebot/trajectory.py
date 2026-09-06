"""One auditable record spanning perception, planning, execution, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .agent_planner import PlannerDecision
from .events import RuntimeEvent
from .perception import BeliefUpdate
from .reward import RewardBreakdown
from .safety import SafetyResult


@dataclass(frozen=True, slots=True)
class AgentTrajectory:
    observation: RuntimeEvent
    belief: BeliefUpdate
    planner_decision: PlannerDecision
    selected_skill: str | None
    arguments: dict[str, Any]
    safety_decision: SafetyResult | None
    result: dict[str, Any] | None
    reward: RewardBreakdown
    success: bool
    latency_ms: float
