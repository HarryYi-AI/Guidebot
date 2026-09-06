"""Dependency-free keyword, recency, and importance retrieval."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime

from ..models import utc_now
from .long_term import LongTermMemory, MemoryRecord


@dataclass(frozen=True, slots=True)
class RetrievedLongTermMemory:
    memory: MemoryRecord
    score: float


class MemoryRetriever:
    def __init__(self, memory: LongTermMemory) -> None:
        self.memory = memory

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 3,
        now: datetime | None = None,
    ) -> tuple[RetrievedLongTermMemory, ...]:
        current = now or utc_now()
        query_tokens = _tokens(query)
        ranked = []
        for record in self.memory.list_active():
            memory_tokens = _tokens(record.content)
            union = query_tokens | memory_tokens
            similarity = len(query_tokens & memory_tokens) / len(union) if union else 0.0
            age_days = max(0.0, (current - record.updated_at).total_seconds() / 86_400)
            recency = math.exp(-age_days / 30.0)
            score = 0.6 * similarity + 0.2 * recency + 0.2 * record.importance
            ranked.append(RetrievedLongTermMemory(record, score))
        ranked.sort(key=lambda item: item.score, reverse=True)
        return tuple(ranked[: max(0, limit)])


def _tokens(text: str) -> set[str]:
    normalized = text.casefold()
    words = set(re.findall(r"[a-z0-9_]+", normalized))
    chinese = {char for char in normalized if "\u4e00" <= char <= "\u9fff"}
    return words | chinese
