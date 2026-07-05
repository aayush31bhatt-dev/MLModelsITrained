"""
In-process pub/sub bus for realtime events.

The bus is intentionally simple: a single asyncio event loop, a set of
connected WebSocket queues, and a `publish(event)` coroutine that fans out
to every subscriber. This keeps the Phase 2 release dependency-light
(no Redis/Kafka) while still letting clients stay in sync with backend
state changes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[Any]] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue[Any]:
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=256)
        async with self._lock:
            self._subscribers.add(queue)
        logger.info("event_bus: subscriber added (total=%d)", len(self._subscribers))
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[Any]) -> None:
        async with self._lock:
            self._subscribers.discard(queue)
        logger.info("event_bus: subscriber removed (total=%d)", len(self._subscribers))

    async def publish(self, event: dict[str, Any]) -> None:
        # Snapshot to avoid mutation during iteration
        async with self._lock:
            targets = list(self._subscribers)
        for q in targets:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("event_bus: dropping event for slow subscriber")

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


# Singleton bus used by the FastAPI app.
bus = EventBus()
