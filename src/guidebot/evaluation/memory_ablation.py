"""Leave-one-memory-out evaluation for recorded trajectories and mock replays."""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..logbook import to_jsonable


@dataclass(frozen=True, slots=True)
class ReplayDecision:
    """Comparable offline outcome; a larger safety_score means safer behavior."""

    action: Any
    safety_score: float
    verifier_passed: bool


@dataclass(frozen=True, slots=True)
class AblatedMemoryResult:
    memory_id: str
    action_changed: bool
    baseline_action: Any
    ablated_action: Any
    safety_score_delta: float
    verifier_changed: bool
    verifier_passed: bool
    harmful: bool
    stale: bool


@dataclass(frozen=True, slots=True)
class MemoryAblationReport:
    baseline: ReplayDecision
    results: tuple[AblatedMemoryResult, ...]

    @property
    def harmful_memory_ids(self) -> tuple[str, ...]:
        return tuple(result.memory_id for result in self.results if result.harmful)

    @property
    def stale_memory_ids(self) -> tuple[str, ...]:
        return tuple(result.memory_id for result in self.results if result.stale)


ReplayRunner = Callable[
    [Any, tuple[Any, ...]],
    ReplayDecision | Awaitable[ReplayDecision],
]


class MemoryAblationEvaluator:
    """Replays N+1 decisions offline; it is intentionally not imported by AgentLoop."""

    def __init__(self, *, safety_epsilon: float = 1e-9) -> None:
        self.safety_epsilon = safety_epsilon

    async def evaluate(
        self,
        trajectory: Any,
        replay: ReplayRunner,
        *,
        memory_entries: Sequence[Any] | None = None,
    ) -> MemoryAblationReport:
        entries = tuple(memory_entries) if memory_entries is not None else _extract_memories(trajectory)
        baseline = await _run(replay, trajectory, entries)
        results = []
        for index, memory in enumerate(entries):
            remaining = entries[:index] + entries[index + 1 :]
            ablated = await _run(replay, trajectory, remaining)
            ablated_action = ablated.action
            action_changed = _stable(ablated_action) != _stable(baseline.action)
            safety_delta = ablated.safety_score - baseline.safety_score
            verifier_changed = ablated.verifier_passed != baseline.verifier_passed
            harmful = safety_delta > self.safety_epsilon or (
                not baseline.verifier_passed and ablated.verifier_passed
            )
            no_effect = (
                not action_changed
                and abs(safety_delta) <= self.safety_epsilon
                and not verifier_changed
            )
            results.append(
                AblatedMemoryResult(
                    _memory_id(memory, index),
                    action_changed,
                    baseline.action,
                    ablated_action,
                    safety_delta,
                    verifier_changed,
                    ablated.verifier_passed,
                    harmful,
                    _historical(memory) or no_effect,
                )
            )
        return MemoryAblationReport(baseline, tuple(results))


async def _run(replay: ReplayRunner, trajectory: Any, memories: tuple[Any, ...]) -> ReplayDecision:
    outcome = replay(trajectory, memories)
    if inspect.isawaitable(outcome):
        outcome = await outcome
    if not isinstance(outcome, ReplayDecision):
        raise TypeError("memory ablation replay must return ReplayDecision")
    return outcome


def _extract_memories(trajectory: Any) -> tuple[Any, ...]:
    payload = to_jsonable(trajectory)
    if not isinstance(payload, Mapping):
        raise ValueError("trajectory must contain a structured context")
    context = payload.get("context", payload)
    trajectory_steps = payload.get("trajectory")
    if isinstance(trajectory_steps, Sequence) and not isinstance(trajectory_steps, (str, bytes)):
        contexts = [
            step.get("context")
            for step in trajectory_steps
            if isinstance(step, Mapping) and isinstance(step.get("context"), Mapping)
        ]
        if contexts:
            context = contexts[-1]
    if not isinstance(context, Mapping):
        raise ValueError("trajectory context must be an object")
    direct = context.get("memories")
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes)):
        return tuple(direct)
    memory_context = context.get("memory_context", context.get("memory"))
    if not isinstance(memory_context, Mapping):
        raise ValueError("trajectory does not contain memory entries")
    entries = []
    for section in (
        "core",
        "core_memory",
        "current_state",
        "active_task_path",
        "recent_observations",
        "retrieved_history",
        "facts",
        "preferences",
        "episodes",
        "boundaries",
        "skill_evidence",
    ):
        values = memory_context.get(section, ())
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            entries.extend(values)
    return tuple(entries)


def _memory_id(memory: Any, index: int) -> str:
    if isinstance(memory, Mapping):
        return str(memory.get("memory_id") or memory.get("id") or f"memory-{index}")
    return str(getattr(memory, "memory_id", getattr(memory, "id", f"memory-{index}")))


def _historical(memory: Any) -> bool:
    status = memory.get("status") if isinstance(memory, Mapping) else getattr(memory, "status", None)
    if isinstance(status, Enum):
        status = status.value
    return status in {"historical", "expired", "abandoned"}


def _stable(value: Any) -> str:
    return json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True)
