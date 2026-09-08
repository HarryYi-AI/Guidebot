"""Deterministic token-budget allocation for structured Agent context parts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Protocol

from ..logbook import to_jsonable


CORE_MEMORY = "core_memory"
CURRENT_STATE = "current_state"
ACTIVE_TASK_PATH = "active_task_path"
RECENT_OBSERVATIONS = "recent_observations"
RETRIEVED_HISTORY = "retrieved_history"
SAFETY = "safety"

_SHARES = {
    CORE_MEMORY: 0.15,
    CURRENT_STATE: 0.20,
    ACTIVE_TASK_PATH: 0.30,
    RECENT_OBSERVATIONS: 0.20,
    RETRIEVED_HISTORY: 0.10,
    SAFETY: 0.05,
}


class TokenCounter(Protocol):
    def count(self, value: Any) -> int: ...


class ApproximateTokenCounter:
    """Dependency-free estimate: CJK chars ~= 1 token, other text ~= 4 chars/token."""

    def count(self, value: Any) -> int:
        text = value if isinstance(value, str) else json.dumps(
            to_jsonable(value), ensure_ascii=False, sort_keys=True
        )
        cjk = sum("\u4e00" <= char <= "\u9fff" for char in text)
        return max(1, cjk + math.ceil((len(text) - cjk) / 4)) if text else 0


@dataclass(frozen=True, slots=True)
class BudgetUsage:
    cap: int
    used: int
    truncated: bool


class ContextBudgetAllocator:
    """Allocates a fixed context window without borrowing the safety reserve."""

    def __init__(
        self,
        total_tokens: int = 4_096,
        *,
        max_caps: dict[str, int] | None = None,
        counter: TokenCounter | None = None,
    ) -> None:
        if total_tokens < 20:
            raise ValueError("total token budget must be at least 20")
        self.total_tokens = total_tokens
        overrides = max_caps or {}
        unknown = set(overrides) - set(_SHARES)
        if unknown:
            raise ValueError(f"unknown context budget sections: {sorted(unknown)}")
        if any(cap < 0 for cap in overrides.values()):
            raise ValueError("context budget caps must be non-negative")
        self.caps = {
            name: min(int(total_tokens * share), overrides.get(name, total_tokens))
            for name, share in _SHARES.items()
        }
        self.counter = counter or ApproximateTokenCounter()
        self.last_usage: dict[str, BudgetUsage] = {}

    def allocate(self, context_parts: dict[str, Any]) -> dict[str, Any]:
        unknown = set(context_parts) - set(_SHARES)
        if unknown:
            raise ValueError(f"unknown context parts: {sorted(unknown)}")
        allocated: dict[str, Any] = {}
        self.last_usage = {}
        for section, value in context_parts.items():
            cap = self.caps[section]
            before = self.counter.count(value)
            fitted = self._fit(
                value,
                cap,
                prefer_recent=section == RECENT_OBSERVATIONS,
                preserve_endpoints=section == ACTIVE_TASK_PATH,
            )
            used = self.counter.count(fitted)
            allocated[section] = fitted
            self.last_usage[section] = BudgetUsage(cap, used, used < before)
        return allocated

    def _fit(
        self,
        value: Any,
        cap: int,
        *,
        prefer_recent: bool = False,
        preserve_endpoints: bool = False,
    ) -> Any:
        if cap <= 0:
            return None
        normalized = to_jsonable(value)
        if self.counter.count(normalized) <= cap:
            return normalized
        if isinstance(normalized, str):
            return self._fit_text(normalized, cap)
        if isinstance(normalized, list):
            if preserve_endpoints and normalized:
                endpoints = normalized[:1]
                if len(normalized) > 1:
                    endpoints.append(normalized[-1])
                per_item = max(1, (cap - 1) // len(endpoints))
                fitted = [self._fit(item, per_item) for item in endpoints]
                while fitted and self.counter.count(fitted) > cap:
                    fitted.pop(0 if len(fitted) == 1 else 1)
                return fitted
            source = list(reversed(normalized)) if prefer_recent else normalized
            kept: list[Any] = []
            for item in source:
                candidate = [*kept, item]
                if self.counter.count(candidate) <= cap:
                    kept.append(item)
            if not kept and source:
                fitted = self._fit(source[0], max(1, cap - 1))
                if self.counter.count([fitted]) <= cap:
                    kept.append(fitted)
            return list(reversed(kept)) if prefer_recent else kept
        if isinstance(normalized, dict):
            kept_dict: dict[str, Any] = {}
            for key, item in normalized.items():
                candidate = {**kept_dict, key: item}
                if self.counter.count(candidate) <= cap:
                    kept_dict[key] = item
                    continue
                fitted = self._fit(item, max(0, cap - self.counter.count(kept_dict) - 2))
                candidate = {**kept_dict, key: fitted}
                if fitted is not None and self.counter.count(candidate) <= cap:
                    kept_dict[key] = fitted
            return kept_dict
        return None

    def _fit_text(self, text: str, cap: int) -> str:
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if self.counter.count(text[:middle]) <= cap:
                low = middle
            else:
                high = middle - 1
        return text[:low]
