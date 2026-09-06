"""Post-execution verification independent of the module that produced output."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .safety import SafetyResult
from .scheduler import Task
from .tooling import ToolExecution, ToolStatus


class TaskVerificationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"


@dataclass(frozen=True, slots=True)
class TaskVerification:
    status: TaskVerificationStatus
    checks: tuple[str, ...]
    reason: str


class TaskVerifier:
    """Verify observable postconditions; it never trusts a success-shaped string alone."""

    def verify(
        self,
        task: Task,
        safety: SafetyResult,
        execution: ToolExecution | None,
    ) -> TaskVerification:
        if not safety.allowed:
            return TaskVerification(
                TaskVerificationStatus.NOT_RUN,
                ("safety_allowed=false",),
                "tool was not executed because safety rejected the task",
            )
        if execution is None or execution.status is not ToolStatus.SUCCEEDED:
            status = execution.status.value if execution else "missing"
            return TaskVerification(
                TaskVerificationStatus.FAILED,
                (f"tool_status={status}",),
                "tool did not complete successfully",
            )
        output = execution.output or {}
        checks = [
            f"module_match={output.get('module') == task.target_module}",
            f"action_match={output.get('action') == task.action}",
        ]
        passed = output.get("module") == task.target_module and output.get("action") == task.action

        if task.target_module == "mobility" and task.action == "stop":
            stopped = output.get("stopped") is True
            checks.append(f"stopped={stopped}")
            passed = passed and stopped
        if task.target_module == "mobility" and task.action == "move_forward":
            moving = output.get("stopped") is False
            checks.append(f"moving={moving}")
            passed = passed and moving
        if task.target_module == "scene_monitor" and task.action == "inspect_again":
            rescan_requested = output.get("rescan_requested") is True
            checks.append(f"rescan_requested={rescan_requested}")
            passed = passed and rescan_requested
        if task.target_module == "ad_creative":
            reviewable = output.get("requires_human_review") is True
            unpublished = output.get("published") is False
            checks.extend((f"human_review={reviewable}", f"unpublished={unpublished}"))
            passed = passed and reviewable and unpublished
        if task.target_module == "climate_control" and task.action == "suggest_comfort":
            proposal_only = output.get("real_control_enabled") is False
            checks.append(f"proposal_only={proposal_only}")
            passed = passed and proposal_only

        return TaskVerification(
            TaskVerificationStatus.PASSED if passed else TaskVerificationStatus.FAILED,
            tuple(checks),
            "postconditions satisfied" if passed else "one or more postconditions failed",
        )
