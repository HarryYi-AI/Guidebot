"""Bounded context construction from trajectory and lightweight memories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from ..events import Event
from ..logbook import to_jsonable
from ..memory import LongTermMemory, MemoryContext, MemoryRetriever, MemoryService, WorkingMemory
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

    def as_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


class ContextManager:
    def __init__(
        self,
        *,
        recent_steps: int = 6,
        long_term_memory: LongTermMemory | None = None,
        memory_service: MemoryService | None = None,
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

    def reset(self) -> None:
        """Start a new task while retaining long-term memory."""
        self.working_memory.clear()

    def update(self, step: AgentLoopStep) -> None:
        """Record a completed step in the bounded working context."""
        self.working_memory.append(step)

    def record_run(self, user_id: str, result: AgentRunResult) -> None:
        """Persist the completed task through MemoryService, never raw SQLite."""
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
            context={"trajectory": to_jsonable(result.trajectory)},
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
        return AgentContext(
            goal,
            latest_observation,
            self.summarizer.summarize(old_steps),
            recent,
            memories,
            registry.schemas(),
            required_tools,
            completed,
            self.memory_service.get_context(user_id, goal),
        )
