"""ConnectionPool — adapted from LiveKit Agents (Apache 2.0).

Manages a pool of persistent connections (e.g. WebSockets) with
automatic reconnection after max session duration.

Changes from LiveKit original:
- No OpenTelemetry dependency
- Simplified logging
- Type hints adapted for aiohttp.ClientWebSocketResponse
"""

from __future__ import annotations

import asyncio
import time
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ConnectionPool(Generic[T]):
    """Manage persistent connections like WebSockets.

    Handles connection pooling and reconnection after max duration.
    Use as an async context manager to automatically return connections.
    """

    def __init__(
        self,
        *,
        max_session_duration: float | None = None,
        mark_refreshed_on_get: bool = False,
        connect_cb: Callable[[float], Awaitable[T]] | None = None,
        close_cb: Callable[[T], Awaitable[None]] | None = None,
        connect_timeout: float = 10.0,
    ) -> None:
        self._max_session_duration = max_session_duration
        self._mark_refreshed_on_get = mark_refreshed_on_get
        self._connect_cb = connect_cb
        self._close_cb = close_cb
        self._connections: dict[T, float] = {}  # conn -> connected_at
        self._available: set[T] = set()
        self._connect_timeout = connect_timeout
        self._connect_lock = asyncio.Lock()
        self._to_close: set[T] = set()
        self._prewarm_task: asyncio.Task | None = None

        # Timing from last get()
        self.last_acquire_time: float = 0.0
        self.last_connection_reused: bool = False

    async def _connect(self, timeout: float) -> T:
        if self._connect_cb is None:
            raise NotImplementedError("Must provide connect_cb")
        conn = await self._connect_cb(timeout)
        self._connections[conn] = time.time()
        return conn

    async def _drain_to_close(self) -> None:
        while self._to_close:
            conn = self._to_close.pop()
            try:
                await self._close_connection(conn)
            except Exception:
                logger.warning("error closing connection", exc_info=True)

    async def _close_connection(self, conn: T) -> None:
        if self._close_cb is not None:
            await self._close_cb(conn)

    @asynccontextmanager
    async def connection(self, *, timeout: float) -> AsyncGenerator[T, None]:
        conn = await self.get(timeout=timeout)
        try:
            yield conn
        except BaseException:
            self.remove(conn)
            raise
        else:
            self.put(conn)

    async def get(self, *, timeout: float) -> T:
        async with self._connect_lock:
            await self._drain_to_close()
            now = time.time()

            # Try to reuse an available non-expired connection
            while self._available:
                conn = self._available.pop()
                if (
                    self._max_session_duration is None
                    or now - self._connections.get(conn, 0) <= self._max_session_duration
                ):
                    if self._mark_refreshed_on_get:
                        self._connections[conn] = now
                    self.last_acquire_time = 0.0
                    self.last_connection_reused = True
                    return conn
                self.remove(conn)

            t0 = time.perf_counter()
            conn = await self._connect(timeout)
            self.last_acquire_time = time.perf_counter() - t0
            self.last_connection_reused = False
            return conn

    def put(self, conn: T) -> None:
        if conn in self._connections:
            self._available.add(conn)

    def remove(self, conn: T) -> None:
        self._available.discard(conn)
        if conn in self._connections:
            self._to_close.add(conn)
            self._connections.pop(conn, None)

    def invalidate(self) -> None:
        for conn in list(self._connections.keys()):
            self._to_close.add(conn)
        self._connections.clear()
        self._available.clear()

    def prewarm(self) -> None:
        if self._prewarm_task is not None or self._connections:
            return

        async def _prewarm_impl() -> None:
            async with self._connect_lock:
                if not self._connections:
                    conn = await self._connect(timeout=self._connect_timeout)
                    self._available.add(conn)

        self._prewarm_task = asyncio.create_task(_prewarm_impl())

    async def aclose(self) -> None:
        if self._prewarm_task:
            if not self._prewarm_task.done():
                self._prewarm_task.cancel()
                try:
                    await self._prewarm_task
                except (asyncio.CancelledError, Exception):
                    pass
        self.invalidate()
        await self._drain_to_close()
