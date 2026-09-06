"""Small async event bus; replaceable with MQTT or ROS 2 without changing the hub."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable

from .models import DomainEvent

EventHandler = Callable[[DomainEvent], Awaitable[None]]


class AsyncEventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[EventHandler]] = defaultdict(list)
        self._history: list[DomainEvent] = []

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        self._subscribers[topic].append(handler)

    async def publish(self, event: DomainEvent) -> None:
        self._history.append(event)
        handlers = self._subscribers.get(event.topic, []) + self._subscribers.get("*", [])
        if handlers:
            await asyncio.gather(*(handler(event) for handler in handlers))

    @property
    def history(self) -> tuple[DomainEvent, ...]:
        return tuple(self._history)


EventBus = AsyncEventBus
# Backward-compatible alias for older integrations.
