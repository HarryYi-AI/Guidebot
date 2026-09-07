"""Vendor-neutral extraction of structured memory candidates from events."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, Callable

from ..events import RuntimeEvent
from .schemas import FactMemory, MemoryObject, SkillEvidenceMemory, TemporaryStateMemory


class BaseMemoryExtractor(ABC):
    @abstractmethod
    def extract_memory_candidates(
        self,
        event: RuntimeEvent,
        *,
        user_id: str,
        source_episode_id: str,
    ) -> list[MemoryObject]: ...


class RuleBasedMemoryExtractor(BaseMemoryExtractor):
    """Conservative rules: one utterance may create state/evidence, never a preference."""

    def __init__(self, *, state_ttl: timedelta = timedelta(hours=4)) -> None:
        self.state_ttl = state_ttl

    def extract_memory_candidates(
        self,
        event: RuntimeEvent,
        *,
        user_id: str,
        source_episode_id: str,
    ) -> list[MemoryObject]:
        text = str(event.payload.get("text", "")).casefold()
        result: list[MemoryObject] = []
        for needles, state, cause in (
            (("累", "疲劳", "tired"), "tired", "user_report"),
            (("焦虑", "紧张", "anxious"), "anxious", "user_report"),
            (("冷", "cold"), "feels_cold", "temperature"),
            (("热", "hot"), "feels_hot", "temperature"),
        ):
            if any(needle in text for needle in needles):
                result.append(
                    TemporaryStateMemory(
                        user_id,
                        state,
                        "user",
                        cause,
                        event.confidence,
                        event.timestamp,
                        event.timestamp + self.state_ttl,
                        source_episode_id=source_episode_id,
                    )
                )
        if "fact_key" in event.payload:
            result.append(
                FactMemory(
                    user_id,
                    str(event.payload["fact_key"]),
                    event.payload.get("fact_value"),
                    event.confidence,
                    valid_from=event.timestamp,
                    source_episode_id=source_episode_id,
                )
            )
        if event.payload.get("skill_name") and "success" in event.payload:
            procedure = event.payload.get("procedure", ())
            result.append(
                SkillEvidenceMemory(
                    user_id,
                    str(event.payload["skill_name"]),
                    str(event.payload.get("applicability", event.event_type)),
                    tuple(str(step) for step in procedure),
                    bool(event.payload["success"]),
                    str(event.payload.get("outcome", "unknown")),
                    source_episode_id,
                    event.confidence,
                    timestamp=event.timestamp,
                )
            )
        return result


class LLMMemoryExtractor(BaseMemoryExtractor):
    """Strict-JSON extension point; no model vendor is embedded in memory code."""

    def __init__(self, completion: Callable[[dict[str, Any]], str]) -> None:
        self.completion = completion

    def extract_memory_candidates(
        self,
        event: RuntimeEvent,
        *,
        user_id: str,
        source_episode_id: str,
    ) -> list[MemoryObject]:
        raw = json.loads(self.completion({"event": event.payload, "type": event.event_type}))
        if not isinstance(raw, dict) or set(raw) != {"candidates"}:
            raise ValueError("LLM memory extraction must return {'candidates': [...]} JSON")
        if not isinstance(raw["candidates"], list):
            raise ValueError("candidates must be a list")
        # First release accepts only atomic facts from the LLM boundary.
        result: list[MemoryObject] = []
        for item in raw["candidates"]:
            if not isinstance(item, dict) or set(item) != {"type", "key", "value", "confidence"}:
                raise ValueError("invalid memory candidate schema")
            if item["type"] != "fact":
                raise ValueError("unsupported LLM candidate type")
            result.append(
                FactMemory(
                    user_id,
                    str(item["key"]),
                    item["value"],
                    float(item["confidence"]),
                    valid_from=event.timestamp,
                    source_episode_id=source_episode_id,
                )
            )
        return result


class MockLLMMemoryExtractor(LLMMemoryExtractor):
    def __init__(self, response: str = '{"candidates": []}') -> None:
        super().__init__(lambda _: response)
