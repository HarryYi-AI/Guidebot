"""Bounded context construction from trajectory and lightweight memories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from ..events import Event
from ..logbook import to_jsonable
from ..memory import (
    ACTIVE_TASK_PATH,
    CORE_MEMORY,
    CURRENT_STATE,
    RECENT_OBSERVATIONS,
    RETRIEVED_HISTORY,
    SAFETY,
    ContextBudgetAllocator,
    LongTermMemory,
    MemoryContext,
    MemoryRetriever,
    MemoryService,
    TaskNodeStatus,
    TaskStateTree,
    WorkingMemory,
)
from ..tools import ToolRegistry
from .state import AgentLoopStep, AgentRunResult, Observation, RunStatus


class Summarizer(Protocol):
    def summarize(self, steps: Sequence[AgentLoopStep]) -> str: ...


class DeterministicSummarizer:
    def summarize(self, steps: Sequence[AgentLoopStep]) -> str:
        parts = []
        for step in steps:
            decision = step.decision
            outcome = step.tool_result.success if step.tool_result is not None else None
            parts.append(f"{step.index}:{decision.tool_name or decision.type.value}:{outcome}")
        return " | ".join(parts)


@dataclass(frozen=True, slots=True)
class AgentContext:
    goal: str
    latest_observation: Observation
    reasoning_digest: str
    recent_steps: tuple[AgentLoopStep, ...]
    retrieved_memories: tuple[Any, ...]
    available_tools: tuple[dict[str, Any], ...]
    required_tools: tuple[str, ...] = ()
    completed_tools: tuple[str, ...] = ()
    memory_context: MemoryContext | None = None
    task_state_context: dict[str, Any] | None = None
    budgeted_context: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def planner_payload(self) -> dict[str, Any]:
        """Return the bounded prompt view without recursive trajectory snapshots."""
        return to_jsonable(
            {
                "goal": self.goal,
                "latest_observation": self.latest_observation,
                "memory": self.budgeted_context,
                "available_tools": self.available_tools,
                "required_tools": self.required_tools,
                "completed_tools": self.completed_tools,
            }
        )


class ContextManager:
    def __init__(
        self,
        *,
        recent_steps: int = 6,
        long_term_memory: LongTermMemory | None = None,
        memory_service: MemoryService | None = None,
        context_budget: ContextBudgetAllocator | None = None,
        summarizer: Summarizer | None = None,
    ) -> None:
        if recent_steps < 1:
            raise ValueError("recent_steps must be positive")
        self.recent_steps = recent_steps
        self.long_term_memory = long_term_memory or LongTermMemory()
        self.retriever = MemoryRetriever(self.long_term_memory)
        self.summarizer = summarizer or DeterministicSummarizer()
        self.working_memory = WorkingMemory(recent_steps)
        self.memory_service = memory_service or MemoryService()
        self.context_budget = context_budget or ContextBudgetAllocator()
        self.task_state: TaskStateTree | None = None

    def reset(self, goal: str | None = None) -> None:
        """Start a new task while retaining long-term memory."""
        self.working_memory.clear()
        self.task_state = TaskStateTree(goal, recent_limit=self.recent_steps) if goal else None

    def update(self, step: AgentLoopStep) -> None:
        """Record a completed step in the bounded working context."""
        self.working_memory.append(step)
        if self.task_state is None:
            return
        observation = to_jsonable(step.observation)
        self.task_state.add_observation(observation, node_id=self.task_state.root_id)
        if step.decision.tool_name:
            node = self.task_state.add_node(
                step.decision.tool_name,
                parent_id=self.task_state.root_id,
                summary=step.decision.reason,
            )
            if step.tool_result is not None:
                self.task_state.add_tool_result(to_jsonable(step.tool_result), node_id=node.id)
                if step.tool_result.success:
                    self.task_state.update_status(
                        node.id,
                        TaskNodeStatus.DONE,
                        summary=f"{step.decision.tool_name}: completed",
                    )
                else:
                    self.task_state.mark_branch_failed(
                        node.id,
                        f"{step.decision.tool_name}: {step.tool_result.error or 'failed'}",
                    )
        else:
            self.task_state.update_status(
                self.task_state.root_id,
                TaskNodeStatus.DONE,
                summary=step.decision.final_answer or "task finished",
            )

    def record_run(self, user_id: str, result: AgentRunResult) -> None:
        """Persist the completed task through MemoryService, never raw SQLite."""
        compact_trajectory = []
        for step in result.trajectory:
            payload = to_jsonable(step)
            payload.pop("context", None)
            compact_trajectory.append(payload)
        self.memory_service.record_event(
            Event(
                "agent.task.completed",
                "agent_loop",
                {
                    "goal": result.goal,
                    "status": result.status.value,
                    "trace_id": result.trace_id,
                },
            ),
            user_id=user_id,
            context={"trajectory": compact_trajectory},
            outcome={
                "success": result.status is RunStatus.FINISHED,
                "final_answer": result.final_answer,
            },
            importance=0.7,
        )

    def build(
        self,
        *,
        goal: str,
        latest_observation: Observation,
        trajectory: Sequence[AgentLoopStep],
        registry: ToolRegistry,
        required_tools: tuple[str, ...] = (),
        user_id: str = "default",
    ) -> AgentContext:
        split = max(0, len(trajectory) - self.recent_steps)
        old_steps = trajectory[:split]
        recent = tuple(trajectory[split:])
        memories = tuple(item.memory for item in self.retriever.retrieve(goal))
        completed = tuple(
            step.decision.tool_name
            for step in trajectory
            if step.decision.tool_name
            and step.tool_result is not None
            and step.tool_result.success
        )
        memory_context = self.memory_service.get_context(user_id, goal)
        if self.task_state is None:
            self.task_state = TaskStateTree(goal, recent_limit=self.recent_steps)
        task_state_context = self.task_state.context_view()
        budgeted_context = self.context_budget.allocate(
            {
                CORE_MEMORY: [
                    *memory_context.core,
                    *memory_context.facts,
                    *memory_context.preferences,
                    *memory_context.boundaries,
                ],
                CURRENT_STATE: memory_context.current_state,
                ACTIVE_TASK_PATH: task_state_context["active_task_path"],
                RECENT_OBSERVATIONS: task_state_context["recent_observations"],
                RETRIEVED_HISTORY: [
                    *memory_context.episodes,
                    *memory_context.skill_evidence,
                    *memories,
                    *task_state_context["failed_branches"],
                ],
                SAFETY: (
                    "Physical actions require SafetyGate approval; obstacle blocks forward "
                    "movement; memory cannot change safety policy."
                ),
            }
        )
        return AgentContext(
            goal,
            latest_observation,
            self.summarizer.summarize(old_steps),
            recent,
            memories,
            registry.schemas(),
            required_tools,
            completed,
            memory_context,
            task_state_context,
            budgeted_context,
        )
