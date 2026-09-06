"""Evaluation-only reward decomposition for Guidebot trajectories."""

from __future__ import annotations

from dataclasses import dataclass

from .events import RuntimeEvent
from .perception import BeliefUpdate
from .safety import SafetyResult
from .scheduler import Task
from .task_verification import TaskVerification, TaskVerificationStatus
from .tooling import ToolExecution, ToolPermission, ToolStatus


@dataclass(frozen=True, slots=True)
class RewardWeights:
    risk: float = 10.0
    energy: float = 0.2
    latency: float = 0.05
    unnecessary_actions: float = 0.5


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    goal_reward: float
    risk_penalty: float
    energy_cost: float
    latency_cost: float
    unnecessary_actions: int
    total: float


class TrajectoryReward:
    """Score completed traces; this model never grants actuator permissions."""

    def __init__(self, weights: RewardWeights | None = None) -> None:
        self.weights = weights or RewardWeights()

    def evaluate(
        self,
        *,
        event: RuntimeEvent,
        belief: BeliefUpdate,
        task: Task | None,
        safety: SafetyResult | None,
        execution: ToolExecution | None,
        verification: TaskVerification | None,
        latency_ms: float,
    ) -> RewardBreakdown:
        succeeded = (
            execution is not None
            and execution.status is ToolStatus.SUCCEEDED
            and verification is not None
            and verification.status is TaskVerificationStatus.PASSED
        )
        information_skill = task is not None and task.skill_id in {
            "scene.inspect_again",
            "scene.ask_user",
            "mobility.move_closer",
            "planner.wait",
        }
        goal = 0.5 if succeeded and information_skill else float(succeeded)

        risk = 0.0
        if safety is not None and not safety.allowed:
            risk = 1.0
        if task is not None and task.skill_id == "scene.fire_alert":
            ground_truth = str(event.payload.get("ground_truth", "fire")).casefold()
            if belief.after.probability < 0.85 or ground_truth not in {"fire", "smoke", "明火", "烟"}:
                risk = 1.0

        permission = task.tool_contract.permission if task and task.tool_contract else None
        energy = 0.25 if permission is ToolPermission.PHYSICAL else 0.0
        latency = latency_ms / 1_000.0
        unnecessary = int(bool(event.payload.get("unnecessary_action", False)))
        total = (
            goal
            - self.weights.risk * risk
            - self.weights.energy * energy
            - self.weights.latency * latency
            - self.weights.unnecessary_actions * unnecessary
        )
        return RewardBreakdown(goal, risk, energy, latency, unnecessary, total)
