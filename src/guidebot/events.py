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
class RuntimeEvent:
    event_type: str
    source: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=utc_now)
    confidence: float = 1.0
    priority_hint: int = 0
    event_id: str = field(default_factory=lambda: uuid4().hex)
    session_id: str | None = None


Event = RuntimeEvent
# Backward-compatible alias; new online-runtime code should say RuntimeEvent.


EventHandler = Callable[[RuntimeEvent], None]


class EventBus:
    """In-process FIFO event bus with optional type subscribers."""

    def __init__(self) -> None:
        self._queue: deque[RuntimeEvent] = deque()
        self._queued_ids: set[str] = set()
        self._subscribers: dict[str, list[EventHandler]] = defaultdict(list)

    def publish(self, event: RuntimeEvent) -> None:
        if event.event_id in self._queued_ids:
            return
        self._queue.append(event)
        self._queued_ids.add(event.event_id)
        for handler in self._subscribers.get(event.event_type, ()):
            handler(event)
        for handler in self._subscribers.get("*", ()):
            handler(event)

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._subscribers[event_type].append(handler)

    def poll(self, event_id: str | None = None) -> RuntimeEvent | None:
        if not self._queue:
            return None
        if event_id is None:
            event = self._queue.popleft()
        else:
            event = next((item for item in self._queue if item.event_id == event_id), None)
            if event is None:
                return None
            self._queue.remove(event)
        self._queued_ids.discard(event.event_id)
        return event
