"""Priority scheduler for Guidebot intents."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from .intent import Intent, IntentType
from .models import utc_now
from .runtime_skills import RuntimeSkillRegistry, build_default_runtime_skills


@dataclass(frozen=True, slots=True)
class Task:
    task_id: str
    target_module: str
    action: str
    payload: dict[str, Any]
    priority: int
    interruptible: bool
    requires_confirmation: bool
    created_at: datetime = field(default_factory=utc_now)
    source_intent: Intent | None = None
    skill_id: str | None = None


class Scheduler:
    """Converts intents into executable tasks and orders them by priority."""

    def __init__(
        self,
        cooldowns: dict[IntentType, timedelta] | None = None,
        skill_registry: RuntimeSkillRegistry | None = None,
    ) -> None:
        self.cooldowns = cooldowns or {
            IntentType.HEALTH_SEDENTARY: timedelta(minutes=30),
            IntentType.UNKNOWN: timedelta(minutes=1),
            IntentType.AC_LEFT_ON_ALERT: timedelta(minutes=1),
            IntentType.CLIMATE_COMFORT: timedelta(minutes=10),
        }
        self.skill_registry = skill_registry or build_default_runtime_skills()
        self._last_scheduled: dict[IntentType, datetime] = {}
        self._heap: list[tuple[int, int, Task]] = []
        self._counter = 0

    def schedule(self, intent: Intent, now: datetime | None = None) -> Task | None:
        now = now or utc_now()
        cooldown = self.cooldowns.get(intent.intent_type)
        last = self._last_scheduled.get(intent.intent_type)
        if cooldown is not None and last is not None and now - last < cooldown:
            return None

        task = self._task_from_intent(intent, now)
        self._last_scheduled[intent.intent_type] = now
        self._counter += 1
        heapq.heappush(self._heap, (-task.priority, self._counter, task))
        return task

    def next_task(self) -> Task | None:
        if not self._heap:
            return None
        return heapq.heappop(self._heap)[2]

    def has_preemption(self) -> bool:
        task = self.peek()
        return task is not None and self.can_preempt(task)

    def peek(self) -> Task | None:
        return self._heap[0][2] if self._heap else None

    @staticmethod
    def can_preempt(task: Task) -> bool:
        return task.priority >= 100 and not task.interruptible

    def _task_from_intent(self, intent: Intent, now: datetime) -> Task:
        skill = self.skill_registry.resolve(intent.intent_type)
        interruptible = skill.interruptible if skill.interruptible is not None else intent.priority < 80
        return Task(
            uuid4().hex,
            skill.target_module,
            skill.action,
            dict(intent.slots),
            intent.priority,
            interruptible,
            intent.requires_confirmation,
            now,
            intent,
            skill.skill_id,
        )
