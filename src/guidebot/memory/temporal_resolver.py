"""Deterministic temporal conflict resolution for versioned memories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .schemas import MemoryObject, MemoryStatus, ResolutionAction, TemporaryStateMemory


@dataclass(frozen=True, slots=True)
class Resolution:
    action: ResolutionAction
    conflicting_ids: tuple[str, ...] = ()
    reason: str = ""


class TemporalResolver:
    def resolve(self, new: MemoryObject, existing: Sequence[MemoryObject]) -> Resolution:
        active = tuple(item for item in existing if item.status is MemoryStatus.ACTIVE)
        if not active:
            return Resolution(ResolutionAction.ADD, reason="no active memory")
        if all(_identity(item) != _identity(new) for item in active):
            return Resolution(ResolutionAction.ADD, reason="different semantic key")

        same_key = tuple(item for item in active if _identity(item) == _identity(new))
        if any(_value(item) == _value(new) for item in same_key):
            if isinstance(new, TemporaryStateMemory) and any(
                new.valid_from > item.valid_from
                for item in same_key
                if isinstance(item, TemporaryStateMemory)
            ):
                return Resolution(
                    ResolutionAction.UPDATE,
                    tuple(item.memory_id for item in same_key),
                    "new observation refreshes temporary-state TTL",
                )
            best = max(float(getattr(item, "confidence", 0.0)) for item in same_key)
            confidence = float(getattr(new, "confidence", 0.0))
            action = ResolutionAction.UPDATE if confidence > best else ResolutionAction.IGNORE
            return Resolution(action, tuple(item.memory_id for item in same_key), "same value")
        return Resolution(
            ResolutionAction.HISTORIZE,
            tuple(item.memory_id for item in same_key),
            "new value supersedes active value without deleting history",
        )


def _identity(memory: MemoryObject) -> tuple[type[Any], str]:
    key = getattr(memory, "key", None) or getattr(memory, "state", None)
    return type(memory), str(key)


def _value(memory: MemoryObject) -> Any:
    if hasattr(memory, "value"):
        return getattr(memory, "value")
    return getattr(memory, "rule", getattr(memory, "state", None))
