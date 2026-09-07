"""Batch consolidation from episodes/evidence into candidates, never active skills."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import mean

from ..models import utc_now
from .schemas import (
    EpisodeMemory,
    MemoryStatus,
    PreferenceMemory,
    SkillCandidate,
    SkillEvidenceMemory,
)
from .store import StructuredMemoryStore


@dataclass(frozen=True, slots=True)
class ConsolidationReport:
    preference_candidates: tuple[PreferenceMemory, ...]
    skill_candidates: tuple[SkillCandidate, ...]
    contradictions: tuple[str, ...]


class MemoryConsolidator:
    def __init__(self, store: StructuredMemoryStore, *, minimum_evidence: int = 2) -> None:
        self.store = store
        self.minimum_evidence = minimum_evidence

    def consolidate(
        self,
        user_id: str,
        time_window: timedelta = timedelta(days=30),
        *,
        now: datetime | None = None,
    ) -> ConsolidationReport:
        current = now or utc_now()
        cutoff = current - time_window
        episodes = tuple(
            item
            for item in self.store.list(EpisodeMemory, user_id)
            if item.timestamp >= cutoff
        )
        preferences, contradictions = self._temperature_preferences(user_id, episodes, current)
        skills = self._skill_candidates(user_id, cutoff, current)
        return ConsolidationReport(preferences, skills, contradictions)

    def _temperature_preferences(
        self,
        user_id: str,
        episodes: tuple[EpisodeMemory, ...],
        now: datetime,
    ) -> tuple[tuple[PreferenceMemory, ...], tuple[str, ...]]:
        successes: list[tuple[float, str]] = []
        failures: list[float] = []
        for episode in episodes:
            payload = {**episode.observation, **episode.outcome}
            if str(payload.get("period", "")).casefold() not in {"night", "夜间", "夜晚"}:
                continue
            try:
                temperature = float(payload["target_temperature"])
            except (KeyError, TypeError, ValueError):
                continue
            if payload.get("comfortable") is True:
                successes.append((temperature, episode.memory_id))
            elif payload.get("comfortable") is False:
                failures.append(temperature)
        if len(successes) < self.minimum_evidence:
            return (), ()
        preferred = round(mean(item[0] for item in successes), 1)
        confidence = min(0.95, 0.55 + 0.1 * len(successes))
        preference = PreferenceMemory(
            user_id,
            "night_temperature_preference",
            {"preferred_c": preferred, "range_c": [preferred - 0.5, preferred]},
            confidence,
            tuple(item[1] for item in successes),
            valid_from=now,
            status=MemoryStatus.ACTIVE,
        )
        contradictions = (
            (f"negative outcomes observed at {sorted(set(failures))}C",) if failures else ()
        )
        return (preference,), contradictions

    def _skill_candidates(
        self,
        user_id: str,
        cutoff: datetime,
        now: datetime,
    ) -> tuple[SkillCandidate, ...]:
        evidence = tuple(
            item
            for item in self.store.list(SkillEvidenceMemory, user_id)
            if item.timestamp >= cutoff
        )
        groups: dict[tuple[str, str, tuple[str, ...]], list[SkillEvidenceMemory]] = {}
        for item in evidence:
            groups.setdefault((item.skill_name, item.applicability, item.procedure), []).append(item)
        candidates = []
        for (name, applicability, procedure), items in groups.items():
            if len(items) < self.minimum_evidence:
                continue
            successes = sum(item.success for item in items)
            failures = len(items) - successes
            candidates.append(
                SkillCandidate(
                    user_id,
                    name,
                    applicability,
                    tuple(item.source_episode_id for item in items),
                    successes,
                    failures,
                    procedure,
                    min(0.95, successes / len(items)),
                    created_at=now,
                )
            )
        return tuple(candidates)
