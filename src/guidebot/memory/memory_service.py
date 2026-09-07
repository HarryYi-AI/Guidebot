"""Single application-facing API for structured temporal memory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, cast

from ..events import RuntimeEvent
from .consolidator import ConsolidationReport, MemoryConsolidator
from .context_builder import MemoryContextBuilder
from .extractor import BaseMemoryExtractor, RuleBasedMemoryExtractor
from .query_planner import MemoryQueryPlanner, RuleBasedMemoryQueryPlanner
from .retriever import StructuredMemoryRetriever
from .schemas import (
    BoundaryMemory,
    EpisodeMemory,
    MemoryContext,
    MemoryObject,
    MemoryStatus,
    PreferenceMemory,
    RelationshipMemory,
    ResolutionAction,
    SkillCandidate,
    SkillEvidenceMemory,
)
from .store import StructuredMemoryStore
from .temporal_resolver import TemporalResolver


@dataclass(frozen=True, slots=True)
class MemoryUpdateResult:
    action: ResolutionAction
    memory: MemoryObject | None
    reason: str


class MemoryService:
    def __init__(
        self,
        store: StructuredMemoryStore | None = None,
        *,
        extractor: BaseMemoryExtractor | None = None,
        query_planner: MemoryQueryPlanner | None = None,
    ) -> None:
        self.store = store or StructuredMemoryStore()
        self.extractor = extractor or RuleBasedMemoryExtractor()
        self.query_planner = query_planner or RuleBasedMemoryQueryPlanner()
        self.resolver = TemporalResolver()
        self.retriever = StructuredMemoryRetriever(self.store)
        self.context_builder = MemoryContextBuilder(self.retriever)
        self.consolidator = MemoryConsolidator(self.store)

    def close(self) -> None:
        self.store.close()

    def record_event(
        self,
        event: RuntimeEvent,
        *,
        user_id: str,
        context: dict[str, Any] | None = None,
        action: dict[str, Any] | None = None,
        outcome: dict[str, Any] | None = None,
        importance: float = 0.5,
    ) -> EpisodeMemory:
        episode = self.store.add(
            EpisodeMemory(
                user_id,
                event.event_type,
                context or {},
                dict(event.payload),
                action,
                outcome or {},
                event.source,
                event.confidence,
                importance,
                event.timestamp,
            )
        )
        for candidate in self.extractor.extract_memory_candidates(
            event,
            user_id=user_id,
            source_episode_id=episode.memory_id,
        ):
            self.update_memory(candidate)
        return episode

    def get_context(
        self,
        user_id: str,
        query: str,
        task_state: dict[str, Any] | None = None,
    ) -> MemoryContext:
        plan = self.query_planner.plan_memory_query(query, task_state)
        return self.context_builder.build(user_id, plan)

    def update_memory(
        self,
        memory: MemoryObject,
        *,
        actor: str = "agent",
    ) -> MemoryUpdateResult:
        if isinstance(memory, SkillEvidenceMemory):
            return MemoryUpdateResult(
                ResolutionAction.ADD,
                self.store.add(memory),
                "skill evidence is append-only",
            )
        memory_type = type(memory)
        key = cast(str | None, getattr(memory, "key", None))
        existing = self.store.list(
            memory_type,
            memory.user_id,
            status=MemoryStatus.ACTIVE,
            key=key,
        )
        if isinstance(memory, BoundaryMemory) and actor != "system":
            raise PermissionError("boundary memory is writable only by the system")
        if isinstance(memory, (PreferenceMemory, RelationshipMemory)) and actor not in {
            "system",
            "consolidator",
        }:
            raise PermissionError("stable profile memory requires consolidation")
        resolution = self.resolver.resolve(memory, existing)
        if resolution.action is ResolutionAction.IGNORE:
            return MemoryUpdateResult(resolution.action, existing[0], resolution.reason)
        if resolution.action in {ResolutionAction.HISTORIZE, ResolutionAction.UPDATE}:
            transition_time = getattr(memory, "valid_from", None)
            for memory_id in resolution.conflicting_ids:
                self.store.set_status(
                    memory_type,
                    memory_id,
                    MemoryStatus.HISTORICAL,
                    valid_to=transition_time,
                )
        stored = self.store.add(memory)
        return MemoryUpdateResult(resolution.action, stored, resolution.reason)

    def run_consolidation(
        self,
        user_id: str,
        time_window: timedelta = timedelta(days=30),
    ) -> ConsolidationReport:
        report = self.consolidator.consolidate(user_id, time_window)
        for preference in report.preference_candidates:
            self.update_memory(preference, actor="consolidator")
        for candidate in report.skill_candidates:
            self.store.add(candidate)
        return report

    def list_active_memories(
        self,
        memory_type: type[MemoryObject],
        user_id: str,
    ) -> tuple[MemoryObject, ...]:
        if memory_type.__name__ == "TemporaryStateMemory":
            self.store.expire_states(user_id)
        return self.store.list(memory_type, user_id, status=MemoryStatus.ACTIVE)

    def list_history(
        self,
        memory_type: type[MemoryObject],
        user_id: str,
    ) -> tuple[MemoryObject, ...]:
        return self.store.list(memory_type, user_id, status=MemoryStatus.HISTORICAL)

    def list_skill_candidates(self, user_id: str) -> tuple[SkillCandidate, ...]:
        return self.store.list(SkillCandidate, user_id)
