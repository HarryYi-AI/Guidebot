"""Query-conditioned structured retrieval with scored archival episodes."""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Protocol

from ..models import utc_now
from .schemas import (
    BoundaryMemory,
    EpisodeMemory,
    FactMemory,
    MemoryQueryPlan,
    MemoryStatus,
    PreferenceMemory,
    RelationshipMemory,
    SkillEvidenceMemory,
    TemporaryStateMemory,
)
from .store import StructuredMemoryStore


class EmbeddingBackend(Protocol):
    def similarity(self, query: str, document: str) -> float: ...


class MockEmbeddingBackend:
    """Deterministic lexical similarity; replaceable by a real embedding backend."""

    def similarity(self, query: str, document: str) -> float:
        left, right = _tokens(query), _tokens(document)
        union = left | right
        return len(left & right) / len(union) if union else 0.0


class StructuredMemoryRetriever:
    def __init__(
        self,
        store: StructuredMemoryStore,
        embedding: EmbeddingBackend | None = None,
    ) -> None:
        self.store = store
        self.embedding = embedding or MockEmbeddingBackend()

    def retrieve_facts(self, user_id: str, keys: tuple[str, ...]) -> tuple[FactMemory, ...]:
        return tuple(
            item
            for key in keys
            for item in self.store.list(FactMemory, user_id, status=MemoryStatus.ACTIVE, key=key)
        )

    def retrieve_active_states(
        self,
        user_id: str,
        *,
        now: datetime | None = None,
    ) -> tuple[TemporaryStateMemory, ...]:
        self.store.expire_states(user_id, now=now)
        return self.store.list(TemporaryStateMemory, user_id, status=MemoryStatus.ACTIVE)

    def retrieve_preferences(
        self,
        user_id: str,
        keys: tuple[str, ...] = (),
    ) -> tuple[PreferenceMemory, ...]:
        active = self.store.list(PreferenceMemory, user_id, status=MemoryStatus.ACTIVE)
        return tuple(item for item in active if not keys or item.key in keys)

    def retrieve_recent_episodes(
        self,
        user_id: str,
        query: str,
        *,
        limit: int = 5,
        now: datetime | None = None,
    ) -> tuple[EpisodeMemory, ...]:
        current = now or utc_now()
        episodes = self.store.list(EpisodeMemory, user_id)

        def score(item: EpisodeMemory) -> float:
            document = f"{item.event_type} {item.context} {item.observation} {item.outcome}"
            semantic = self.embedding.similarity(query, document)
            age_days = max(0.0, (current - item.timestamp).total_seconds() / 86_400)
            recency = math.exp(-age_days / 30.0)
            return 0.6 * semantic + 0.2 * recency + 0.2 * item.importance

        return tuple(sorted(episodes, key=score, reverse=True)[: max(0, limit)])

    def retrieve_skill_evidence(
        self,
        user_id: str,
        query: str,
        *,
        limit: int = 5,
    ) -> tuple[SkillEvidenceMemory, ...]:
        evidence = self.store.list(SkillEvidenceMemory, user_id, status=MemoryStatus.ACTIVE)
        return tuple(
            sorted(
                evidence,
                key=lambda item: self.embedding.similarity(
                    query,
                    f"{item.skill_name} {item.applicability} {item.outcome}",
                ),
                reverse=True,
            )[: max(0, limit)]
        )

    def retrieve_core_memory(
        self,
        user_id: str,
    ) -> tuple[FactMemory | PreferenceMemory | RelationshipMemory, ...]:
        facts = tuple(
            item
            for item in self.store.list(FactMemory, user_id, status=MemoryStatus.ACTIVE)
            if item.key.startswith("persona.")
            or item.key in {"user_name", "home_location", "accessibility_need"}
        )
        preferences = self.store.list(PreferenceMemory, user_id, status=MemoryStatus.ACTIVE)
        relationships = self.store.list(
            RelationshipMemory,
            user_id,
            status=MemoryStatus.ACTIVE,
        )
        return (*facts, *preferences, *relationships)

    def retrieve_boundaries(
        self,
        user_id: str,
        plan: MemoryQueryPlan,
    ) -> tuple[BoundaryMemory, ...]:
        boundaries = self.store.list(BoundaryMemory, user_id, status=MemoryStatus.ACTIVE)
        if not plan.boundary_keys:
            return boundaries
        return tuple(item for item in boundaries if item.key in plan.boundary_keys)


def _tokens(text: str) -> set[str]:
    normalized = text.casefold()
    words = set(re.findall(r"[a-z0-9_]+", normalized))
    return words | {char for char in normalized if "\u4e00" <= char <= "\u9fff"}
