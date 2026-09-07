"""Rule-first query planning for minimal sufficient memory context."""

from __future__ import annotations

from typing import Any, Protocol

from .schemas import MemoryQueryPlan


class MemoryQueryPlanner(Protocol):
    def plan_memory_query(
        self,
        user_query: str,
        task_state: dict[str, Any] | None = None,
    ) -> MemoryQueryPlan: ...


class RuleBasedMemoryQueryPlanner:
    def plan_memory_query(
        self,
        user_query: str,
        task_state: dict[str, Any] | None = None,
    ) -> MemoryQueryPlan:
        text = user_query.casefold()
        climate = any(word in text for word in ("冷", "热", "温度", "空调", "cold", "hot"))
        night = any(word in text for word in ("今晚", "夜间", "半夜", "night"))
        health = any(word in text for word in ("累", "疲劳", "久坐", "健康"))
        return MemoryQueryPlan(
            query=user_query,
            fact_keys=("night_temperature_preference",) if climate and night else (),
            preference_keys=("night_temperature_preference",) if climate else (),
            need_recent_episodes=climate or health,
            need_current_state=climate or health,
            need_relationship_state=any(word in text for word in ("我们", "关系", "信任")),
            need_skill_evidence=climate or health,
            boundary_keys=("physical_action_safety", "non_critical_wake_policy")
            if climate
            else (),
        )
