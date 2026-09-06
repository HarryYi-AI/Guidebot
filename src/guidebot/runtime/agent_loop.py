"""Bounded Planner/Critic/Tool loop for mock and replay validation."""

from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from ..memory import Episode, EpisodicMemory
from ..planning import AgentMessage, CriticAgent, DecisionType, Planner, PlannerDecision
from ..safety import SafetyGate, SafetyResult
from ..tools import ToolRegistry, ToolResult, UnknownToolError
from .context_manager import ContextManager
from .state import AgentLoopStep, AgentRunResult, Observation, RunStatus


class AgentLoop:
    def __init__(
        self,
        planner: Planner,
        registry: ToolRegistry,
        *,
        critic: CriticAgent | None = None,
        safety_gate: SafetyGate | None = None,
        context_manager: ContextManager | None = None,
        episodic_memory: EpisodicMemory | None = None,
        max_steps: int = 12,
        max_revisions: int = 2,
    ) -> None:
        if max_steps < 1 or max_revisions < 0:
            raise ValueError("max_steps must be positive and max_revisions non-negative")
        self.planner = planner
        self.registry = registry
        self.safety_gate = safety_gate or SafetyGate()
        self.critic = critic or CriticAgent(self.safety_gate)
        self.context_manager = context_manager or ContextManager()
        self.episodic_memory = episodic_memory or EpisodicMemory()
        self.max_steps = max_steps
        self.max_revisions = max_revisions
        self.working_memory = self.context_manager.working_memory

    async def run(
        self,
        goal: str,
        *,
        initial_observation: Observation | None = None,
        required_tools: tuple[str, ...] = (),
    ) -> AgentRunResult:
        self.context_manager.reset()
        trace_id = uuid4().hex
        trajectory: list[AgentLoopStep] = []
        messages: list[AgentMessage] = []
        observation = initial_observation or Observation("goal", {"text": goal})

        for index in range(1, self.max_steps + 1):
            context = self.context_manager.build(
                goal=goal,
                latest_observation=observation,
                trajectory=trajectory,
                registry=self.registry,
                required_tools=required_tools,
            )
            reviewed = None
            decision = None
            for revision in range(self.max_revisions + 1):
                try:
                    decision = await self.planner.decide(context)
                except Exception as exc:  # noqa: BLE001 - planner failures become observations.
                    return self._finish(
                        goal,
                        RunStatus.PLANNER_ERROR,
                        trajectory,
                        f"planner error: {type(exc).__name__}: {exc}",
                        messages,
                        trace_id,
                    )
                reviewed = self.critic.review(
                    decision,
                    context,
                    self.registry,
                    trace_id=trace_id,
                )
                messages.append(reviewed.message)
                if reviewed.approved:
                    break
                observation = Observation(
                    "critic_feedback",
                    {"feedback": list(reviewed.feedback), "revision": revision + 1},
                )
                if revision < self.max_revisions:
                    context = self.context_manager.build(
                        goal=goal,
                        latest_observation=observation,
                        trajectory=trajectory,
                        registry=self.registry,
                        required_tools=required_tools,
                    )
            assert decision is not None and reviewed is not None
            if not reviewed.approved:
                return self._finish(
                    goal,
                    RunStatus.CRITIC_REJECTED,
                    trajectory,
                    "; ".join(reviewed.feedback),
                    messages,
                    trace_id,
                )

            started = perf_counter()
            if decision.type is DecisionType.FINISH:
                final_observation = Observation("finished", {"answer": decision.final_answer})
                step = AgentLoopStep(
                    index,
                    decision,
                    reviewed,
                    None,
                    None,
                    final_observation,
                    (perf_counter() - started) * 1_000,
                )
                trajectory.append(step)
                self.context_manager.update(step)
                return self._finish(
                    goal,
                    RunStatus.FINISHED,
                    trajectory,
                    decision.final_answer,
                    messages,
                    trace_id,
                )

            safety, tool_result = await self._execute(decision)
            observation = Observation(
                "tool_result",
                {
                    "tool_name": decision.tool_name,
                    "success": tool_result.success,
                    "data": tool_result.data,
                    "error": tool_result.error,
                },
            )
            step = AgentLoopStep(
                index,
                decision,
                reviewed,
                safety,
                tool_result,
                observation,
                (perf_counter() - started) * 1_000,
            )
            trajectory.append(step)
            self.context_manager.update(step)

        return self._finish(
            goal,
            RunStatus.MAX_STEPS,
            trajectory,
            "maximum steps reached",
            messages,
            trace_id,
        )

    async def _execute(self, decision: PlannerDecision) -> tuple[SafetyResult, ToolResult]:
        try:
            tool = self.registry.get(decision.tool_name or "")
        except UnknownToolError:
            safety = SafetyResult(False, "unknown tool rejected")
            return safety, ToolResult(False, error=safety.reason)
        safety = self.safety_gate.evaluate_tool_call(
            tool.name,
            decision.arguments,
            known_physical_tool=tool.physical,
        )
        if not safety.allowed:
            return safety, ToolResult(False, error=safety.reason)
        return safety, await self.registry.execute(tool.name, **decision.arguments)

    def _finish(
        self,
        goal: str,
        status: RunStatus,
        trajectory: list[AgentLoopStep],
        answer: str | None,
        messages: list[AgentMessage],
        trace_id: str,
    ) -> AgentRunResult:
        result = AgentRunResult(goal, status, tuple(trajectory), answer, tuple(messages), trace_id)
        self.episodic_memory.append(
            Episode(
                task=goal,
                trajectory=result.trajectory,
                outcome=status.value,
                success=status is RunStatus.FINISHED,
                episode_id=result.trace_id,
            )
        )
        return result
