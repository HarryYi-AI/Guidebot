"""Bounded in-memory state for the current task."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from typing import Any


class WorkingMemory:
    def __init__(self, capacity: int = 6) -> None:
        if capacity < 1:
            raise ValueError("working-memory capacity must be positive")
        self.capacity = capacity
        self._steps: deque[Any] = deque(maxlen=capacity)

    def append(self, step: Any) -> None:
        self._steps.append(step)

    def extend(self, steps: Iterable[Any]) -> None:
        self._steps.extend(steps)

    def clear(self) -> None:
        self._steps.clear()

    def items(self) -> tuple[Any, ...]:
        return tuple(self._steps)

    def __len__(self) -> int:
        return len(self._steps)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._steps)
