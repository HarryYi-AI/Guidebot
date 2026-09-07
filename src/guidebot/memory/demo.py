"""Offline demonstration of temporal memory, consolidation and retrieval."""

from __future__ import annotations

import json
from datetime import timedelta

from ..events import Event
from ..logbook import to_jsonable
from ..models import utc_now
from .memory_service import MemoryService
from .schemas import EpisodeMemory, PreferenceMemory


def _feedback(
    text: str,
    temperature: float,
    comfortable: bool,
    *,
    offset_days: int,
) -> Event:
    return Event(
        "climate.feedback",
        "voice",
        {
            "text": text,
            "period": "night",
            "target_temperature": temperature,
            "comfortable": comfortable,
            "skill_name": "NightComfortSkill",
            "applicability": "night temperature comfort",
            "procedure": ["read_temperature", f"suggest_{temperature}C", "ask_feedback"],
            "success": comfortable,
            "outcome": text,
        },
        timestamp=utc_now() + timedelta(days=offset_days),
    )


def main() -> None:
    service = MemoryService()
    user = "demo-user"
    service.update_memory(
        PreferenceMemory(
            user,
            "night_temperature_preference",
            {"preferred_c": 23.0, "range_c": [22.5, 23.0]},
            0.7,
            ("legacy-profile",),
        ),
        actor="system",
    )
    sequence = (
        _feedback("27℃时觉得热", 27.0, False, offset_days=0),
        _feedback("调到23℃后半夜太冷", 23.0, False, offset_days=1),
        _feedback("第二晚25℃舒适", 25.0, True, offset_days=2),
        _feedback("再次使用25℃仍然舒适", 25.0, True, offset_days=3),
    )
    for event in sequence:
        service.record_event(
            event,
            user_id=user,
            action={"target_temperature": event.payload["target_temperature"]},
            outcome={
                "period": "night",
                "target_temperature": event.payload["target_temperature"],
                "comfortable": event.payload["comfortable"],
            },
            importance=0.8,
        )
    report = service.run_consolidation(user, timedelta(days=60))
    context = service.get_context(user, "今晚又有点热")
    payload = {
        "active_preferences": service.list_active_memories(PreferenceMemory, user),
        "historical_preferences": service.list_history(PreferenceMemory, user),
        "recent_episodes": service.store.list(EpisodeMemory, user, limit=3),
        "generated_skill_candidates": report.skill_candidates,
        "reconstructed_context": context,
    }
    print(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
