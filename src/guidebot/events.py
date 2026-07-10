"""Runtime event primitives for multimodal Guidebot inputs."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from .models import utc_now


@dataclass(frozen=True, slots=True)
class Event:
    event_type: str
    source: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=utc_now)
    confidence: float = 1.0
    priority_hint: int = 0
    event_id: str = field(default_factory=lambda: uuid4().hex)


EventHandler = Callable[[Event], None]


class EventBus:
    """In-process FIFO event bus with optional type subscribers."""

    def __init__(self) -> None:
        self._queue: deque[Event] = deque()
        self._subscribers: dict[str, list[EventHandler]] = defaultdict(list)

    def publish(self, event: Event) -> None:
        self._queue.append(event)
        for handler in self._subscribers.get(event.event_type, ()):
            handler(event)
        for handler in self._subscribers.get("*", ()):
            handler(event)

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._subscribers[event_type].append(handler)

    def poll(self) -> Event | None:
        return self._queue.popleft() if self._queue else None
