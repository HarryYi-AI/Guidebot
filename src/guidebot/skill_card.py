"""Versioned metadata for the lifecycle of executable skills."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Mapping
from uuid import uuid4

from .models import utc_now


@dataclass(frozen=True, slots=True)
class SkillCard:
    skill_id: str
    name: str
    version: str
    parent_id: str | None
    description: str
    preconditions: tuple[str, ...]
    action_schema: Mapping[str, Any]
    safety_scope: tuple[str, ...]
    success_count: int = 0
    failure_count: int = 0
    validation_score: float = 0.0
    known_failure_modes: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    accepted: bool = False

    @classmethod
    def create(
        cls,
        name: str,
        *,
        version: int = 1,
        parent_id: str | None = None,
        description: str = "",
        preconditions: tuple[str, ...] = (),
        action_schema: Mapping[str, Any] | None = None,
        safety_scope: tuple[str, ...] = (),
        accepted: bool = False,
    ) -> SkillCard:
        return cls(
            skill_id=uuid4().hex,
            name=name,
            version=f"v{version}",
            parent_id=parent_id,
            description=description,
            preconditions=preconditions,
            action_schema=action_schema or {},
            safety_scope=safety_scope,
            accepted=accepted,
        )

    @property
    def version_number(self) -> int:
        if not self.version.startswith("v") or not self.version[1:].isdigit():
            raise ValueError(f"invalid skill version: {self.version}")
        return int(self.version[1:])

    def child(self, *, name: str | None = None, description: str | None = None) -> SkillCard:
        return SkillCard.create(
            name or self.name,
            version=self.version_number + 1,
            parent_id=self.skill_id,
            description=self.description if description is None else description,
            preconditions=self.preconditions,
            action_schema=self.action_schema,
            safety_scope=self.safety_scope,
        )

    def with_validation(self, score: float, *, accepted: bool) -> SkillCard:
        return replace(self, validation_score=score, accepted=accepted, updated_at=utc_now())

    def record_outcome(self, *, success: bool, failure_mode: str | None = None) -> SkillCard:
        modes = self.known_failure_modes
        if failure_mode and failure_mode not in modes:
            modes = (*modes, failure_mode)
        return replace(
            self,
            success_count=self.success_count + int(success),
            failure_count=self.failure_count + int(not success),
            known_failure_modes=modes,
            updated_at=utc_now(),
        )

