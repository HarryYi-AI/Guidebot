"""Typed schemas for hierarchical temporal embodied-agent memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Union
from uuid import uuid4

from ..models import utc_now


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    HISTORICAL = "historical"
    EXPIRED = "expired"
    CANDIDATE = "candidate"


class ResolutionAction(str, Enum):
    ADD = "add"
    UPDATE = "update"
    HISTORIZE = "historize"
    IGNORE = "ignore"


def _id() -> str:
    return uuid4().hex


@dataclass(frozen=True, slots=True)
class EpisodeMemory:
    user_id: str
    event_type: str
    context: dict[str, Any]
    observation: dict[str, Any]
    action: dict[str, Any] | None
    outcome: dict[str, Any]
    source: str
    confidence: float = 1.0
    importance: float = 0.5
    timestamp: datetime = field(default_factory=utc_now)
    memory_id: str = field(default_factory=_id)
    status: MemoryStatus = MemoryStatus.ACTIVE


@dataclass(frozen=True, slots=True)
class FactMemory:
    user_id: str
    key: str
    value: Any
    confidence: float
    valid_from: datetime = field(default_factory=utc_now)
    valid_to: datetime | None = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    source_episode_id: str | None = None
    memory_id: str = field(default_factory=_id)


@dataclass(frozen=True, slots=True)
class TemporaryStateMemory:
    user_id: str
    state: str
    target: str
    cause: str | None
    confidence: float
    valid_from: datetime
    expires_at: datetime
    status: MemoryStatus = MemoryStatus.ACTIVE
    source_episode_id: str | None = None
    memory_id: str = field(default_factory=_id)


@dataclass(frozen=True, slots=True)
class PreferenceMemory:
    user_id: str
    key: str
    value: Any
    confidence: float
    evidence_episode_ids: tuple[str, ...]
    valid_from: datetime = field(default_factory=utc_now)
    valid_to: datetime | None = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    memory_id: str = field(default_factory=_id)


@dataclass(frozen=True, slots=True)
class BoundaryMemory:
    user_id: str
    key: str
    rule: str
    priority: int = 100
    editable_by_agent: bool = False
    status: MemoryStatus = MemoryStatus.ACTIVE
    created_at: datetime = field(default_factory=utc_now)
    memory_id: str = field(default_factory=_id)


@dataclass(frozen=True, slots=True)
class RelationshipMemory:
    user_id: str
    state: str
    confidence: float
    evidence_episode_ids: tuple[str, ...] = ()
    valid_from: datetime = field(default_factory=utc_now)
    valid_to: datetime | None = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    memory_id: str = field(default_factory=_id)


@dataclass(frozen=True, slots=True)
class SkillEvidenceMemory:
    user_id: str
    skill_name: str
    applicability: str
    procedure: tuple[str, ...]
    success: bool
    outcome: str
    source_episode_id: str
    confidence: float = 1.0
    importance: float = 0.6
    timestamp: datetime = field(default_factory=utc_now)
    status: MemoryStatus = MemoryStatus.ACTIVE
    memory_id: str = field(default_factory=_id)


@dataclass(frozen=True, slots=True)
class SkillCandidate:
    user_id: str
    name: str
    applicability: str
    evidence_episode_ids: tuple[str, ...]
    success_count: int
    failure_count: int
    proposed_procedure: tuple[str, ...]
    confidence: float
    accepted: bool = False
    status: MemoryStatus = MemoryStatus.CANDIDATE
    created_at: datetime = field(default_factory=utc_now)
    memory_id: str = field(default_factory=_id)


@dataclass(frozen=True, slots=True)
class MemoryQueryPlan:
    query: str
    need_core: bool = True
    fact_keys: tuple[str, ...] = ()
    preference_keys: tuple[str, ...] = ()
    need_recent_episodes: bool = False
    need_current_state: bool = False
    need_relationship_state: bool = False
    need_skill_evidence: bool = False
    boundary_keys: tuple[str, ...] = ()
    episode_limit: int = 5


@dataclass(frozen=True, slots=True)
class MemoryContext:
    core: tuple[FactMemory | PreferenceMemory | RelationshipMemory, ...] = ()
    current_state: tuple[TemporaryStateMemory, ...] = ()
    facts: tuple[FactMemory, ...] = ()
    preferences: tuple[PreferenceMemory, ...] = ()
    episodes: tuple[EpisodeMemory, ...] = ()
    boundaries: tuple[BoundaryMemory, ...] = ()
    skill_evidence: tuple[SkillEvidenceMemory, ...] = ()


MemoryObject = Union[
    EpisodeMemory,
    FactMemory,
    TemporaryStateMemory,
    PreferenceMemory,
    BoundaryMemory,
    RelationshipMemory,
    SkillEvidenceMemory,
    SkillCandidate,
]
