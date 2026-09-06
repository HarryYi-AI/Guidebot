"""Existing event-driven Guidebot orchestration, retained with its public API."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from ..agent_planner import UncertaintyAwarePlanner
from ..events import Event, EventBus
from ..intent import Intent
from ..intent_analyzer import IntentAnalyzer
from ..logbook import RuntimeLogger
from ..model_serving import DeterministicBackend, ModelBackend
from ..modules import (
    AdCreativeModule,
    AlarmTimerModule,
    ClimateControlModule,
    HealthMonitorModule,
    MobilityModule,
    SceneMonitorModule,
    VoiceChatModule,
)
from ..outcomes import OutcomeType
from ..perception import MultimodalPerception
from ..reward import TrajectoryReward
from ..runtime_skills import RuntimeSkillRegistry, build_default_runtime_skills
from ..safety import RuntimeSafetyState, SafetyGate, SafetyResult
from ..scheduler import ScheduleDisposition, ScheduleResult, Scheduler, Task
from ..task_verification import TaskVerification, TaskVerificationStatus, TaskVerifier
from ..tooling import ToolContract, ToolDispatcher, ToolExecution, ToolPermission, ToolStatus
from ..trajectory import AgentTrajectory


@dataclass(frozen=True, slots=True)
class RuntimeTrace:
    event: Event
    intent: Intent
    task: Task | None
    safety: SafetyResult | None
    action: dict[str, Any] | None
    trace_id: str = ""
    session_id: str | None = None
    execution: ToolExecution | None = None
    verification: TaskVerification | None = None
    final_status: str = "unknown"
    trajectory: AgentTrajectory | None = None
    outcome_type: OutcomeType = OutcomeType.FAILED
    reason: str = "unknown"

    def human_summary(self) -> str:
        task_name = self.trajectory.selected_skill if self.trajectory else None
        task_name = task_name or self.intent.intent_type.value
        parts = [f"[{self.outcome_type.value}]", f"[{task_name}]"]
        if self.outcome_type is OutcomeType.SUPPRESSED:
            parts.append(f"[{self.reason}]")
        duration = self.trajectory.latency_ms if self.trajectory else 0.0
        parts.append(f"[{duration:.2f}ms]")
        return " ".join(parts)


class GuidebotRuntime:
    """Observation -> EventBus -> perception/planner -> skill -> safety -> module."""

    def __init__(
        self,
        *,
        logger: RuntimeLogger | None = None,
        safety_state: RuntimeSafetyState | None = None,
        skill_registry: RuntimeSkillRegistry | None = None,
        model_backend: ModelBackend | None = None,
        tool_dispatcher: ToolDispatcher | None = None,
        task_verifier: TaskVerifier | None = None,
        perception: MultimodalPerception | None = None,
        planner: UncertaintyAwarePlanner | None = None,
        reward_model: TrajectoryReward | None = None,
    ) -> None:
        self.event_bus = EventBus()
        self.analyzer = IntentAnalyzer()
        self.perception = perception or MultimodalPerception()
        self.planner = planner or UncertaintyAwarePlanner(self.analyzer)
        self.reward_model = reward_model or TrajectoryReward()
        self.skill_registry = skill_registry or build_default_runtime_skills()
        self.scheduler = Scheduler(skill_registry=self.skill_registry)
        self.safety = SafetyGate()
        self.tool_dispatcher = tool_dispatcher or ToolDispatcher()
        self.task_verifier = task_verifier or TaskVerifier()
        self.safety_state = safety_state or RuntimeSafetyState()
        self.logger = logger
        self.modules = {
            "ad_creative": AdCreativeModule(model_backend or DeterministicBackend()),
            "voice_chat": VoiceChatModule(),
            "scene_monitor": SceneMonitorModule(),
            "health_monitor": HealthMonitorModule(),
            "alarm_timer": AlarmTimerModule(),
            "mobility": MobilityModule(),
            "climate_control": ClimateControlModule(),
        }
        for module in self.modules.values():
            module.start(self.event_bus)

    def stop(self) -> None:
        for module in self.modules.values():
            module.stop()

    def ingest(self, event: Event) -> RuntimeTrace:
        started_at = perf_counter()
        self.event_bus.publish(event)
        observation = self.event_bus.poll(event.event_id)
        if observation is None:
            raise RuntimeError("published observation was not available on EventBus")
        belief = self.perception.observe(observation)
        planner_decision = self.planner.decide(observation, belief)

        if observation.event_type == "ultrasonic.obstacle":
            obstacle = observation.payload.get("obstacle")
            if isinstance(obstacle, bool):
                self.safety_state.obstacle = obstacle

        intent = planner_decision.intent
        schedule_result = self.scheduler.schedule_with_outcome(intent)
        task = schedule_result.task
        safety = None
        action = None
        execution = None
        verification = None
        if task is not None:
            safety = self.safety.evaluate_task(task, self.safety_state)
            if safety.allowed:
                contract = task.tool_contract or ToolContract(
                    task.skill_id or f"{task.target_module}.{task.action}",
                    ToolPermission.READ_ONLY,
                )
                execution = self.tool_dispatcher.execute(
                    contract,
                    task.payload,
                    lambda: self.execute(task),
                )
                action = execution.output
                if execution.status is not ToolStatus.SUCCEEDED:
                    action = {
                        "module": task.target_module,
                        "action": task.action,
                        "blocked": True,
                        "reason": execution.error_message,
                        "error_code": execution.error_code,
                    }
                verification = self.task_verifier.verify(task, safety, execution)
                if execution.status is ToolStatus.SUCCEEDED:
                    self._update_safety_state_after(task)
                if (
                    execution.status is ToolStatus.SUCCEEDED
                    and task.priority >= 100
                    and not task.interruptible
                ):
                    self.safety_state.active_safety_alert = True
            else:
                action = {
                    "module": task.target_module,
                    "action": task.action,
                    "blocked": True,
                    "reason": safety.reason,
                }
                verification = self.task_verifier.verify(task, safety, None)

        final_status = _final_status(schedule_result, safety, execution, verification)
        outcome_type, outcome_reason = _outcome(
            schedule_result,
            safety,
            execution,
            verification,
        )
        latency_ms = (perf_counter() - started_at) * 1_000
        reward = self.reward_model.evaluate(
            event=observation,
            belief=belief,
            task=task,
            safety=safety,
            execution=execution,
            verification=verification,
            latency_ms=latency_ms,
            outcome_type=outcome_type,
        )
        trajectory = AgentTrajectory(
            observation=observation,
            belief=belief,
            planner_decision=planner_decision,
            selected_skill=schedule_result.skill_id,
            arguments=dict(task.payload) if task is not None else {},
            safety_decision=safety,
            result=action,
            reward=reward,
            success=outcome_type is not OutcomeType.FAILED,
            latency_ms=latency_ms,
            outcome_type=outcome_type,
            reason=outcome_reason,
        )
        trace = RuntimeTrace(
            event=observation,
            intent=intent,
            task=task,
            safety=safety,
            action=action,
            trace_id=observation.event_id,
            session_id=observation.session_id or observation.source,
            execution=execution,
            verification=verification,
            final_status=final_status,
            trajectory=trajectory,
            outcome_type=outcome_type,
            reason=outcome_reason,
        )
        self._log(trace)
        return trace

    def execute(self, task: Task) -> dict[str, Any]:
        if task.skill_id is None:
            module = self.modules[task.target_module]
            return module.handle_task(task)
        skill = self.skill_registry.get(task.skill_id)
        return skill.execute(self.modules, task)

    def _update_safety_state_after(self, task: Task) -> None:
        if task.target_module == "climate_control" and task.payload.get("target_c") is not None:
            self.safety_state.last_climate_action_at = task.created_at

    def _log(self, trace: RuntimeTrace) -> None:
        if self.logger is None:
            return
        self.logger.event(trace.event)
        self.logger.intent(trace.intent)
        if trace.task is not None:
            self.logger.task(
                {
                    "trace_id": trace.trace_id,
                    "task": trace.task,
                    "safety": trace.safety,
                    "execution": trace.execution,
                    "verification": trace.verification,
                    "action": trace.action,
                    "final_status": trace.final_status,
                }
            )
        self.logger.trace(trace)


def _final_status(
    schedule: ScheduleResult,
    safety: SafetyResult | None,
    execution: ToolExecution | None,
    verification: TaskVerification | None,
) -> str:
    if schedule.disposition is ScheduleDisposition.SUPPRESSED:
        return "suppressed"
    if schedule.disposition is ScheduleDisposition.NO_ACTION_REQUIRED:
        return "no_action_required"
    if safety is not None and not safety.allowed:
        return "safety_rejected"
    if execution is None or execution.status is not ToolStatus.SUCCEEDED:
        return "tool_failed"
    if verification is None or verification.status is not TaskVerificationStatus.PASSED:
        return "verification_failed"
    return "succeeded"


def _outcome(
    schedule: ScheduleResult,
    safety: SafetyResult | None,
    execution: ToolExecution | None,
    verification: TaskVerification | None,
) -> tuple[OutcomeType, str]:
    if schedule.disposition is ScheduleDisposition.SUPPRESSED:
        return OutcomeType.SUPPRESSED, schedule.reason
    if schedule.disposition is ScheduleDisposition.NO_ACTION_REQUIRED:
        return OutcomeType.NO_ACTION_REQUIRED, schedule.reason
    if safety is not None and not safety.allowed:
        return OutcomeType.FAILED, safety.reason
    if execution is None or execution.status is not ToolStatus.SUCCEEDED:
        reason = execution.error_message if execution else "tool execution missing"
        return OutcomeType.FAILED, reason or "tool execution failed"
    if verification is None or verification.status is not TaskVerificationStatus.PASSED:
        reason = verification.reason if verification else "verification missing"
        return OutcomeType.FAILED, reason
    return OutcomeType.EXECUTED, "completed"
