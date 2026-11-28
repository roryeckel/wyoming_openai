"""A limit on how many requests a backend is sent at once."""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

_LOGGER = logging.getLogger(__name__)


class ConcurrencySlot:
    """One slot of a ConcurrencyLimit, held until released."""

    def __init__(self, semaphore: asyncio.Semaphore | None) -> None:
        self._semaphore = semaphore

    def release(self) -> None:
        """Give the slot back; safe to call more than once."""
        semaphore, self._semaphore = self._semaphore, None
        if semaphore is not None:
            semaphore.release()


class ConcurrencyLimit:
    """
    At most `limit` slots held at once, where 0 means no limit. With a name, a request that has to wait for a slot
    is logged under it.
    """

    def __init__(self, limit: int, name: str | None = None) -> None:
        if limit < 0:
            raise ValueError(f"A concurrency limit must be at least 0, got {limit}")
        self.limit = limit
        self._name = name
        self._semaphore = asyncio.Semaphore(limit) if limit else None

    @property
    def full(self) -> bool:
        """Whether every slot is held, so the next acquire waits."""
        return self._semaphore is not None and self._semaphore.locked()

    async def acquire(self) -> ConcurrencySlot:
        """Wait for a free slot, for a request that ends somewhere else than where it starts."""
        if self._semaphore is not None:
            if self._name is not None and self.full:
                _LOGGER.info("Waiting for a free %s slot: all %d are in use", self._name, self.limit)
            await self._semaphore.acquire()
        return ConcurrencySlot(self._semaphore)

    @contextlib.asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Hold a slot for the duration of the block."""
        slot = await self.acquire()
        try:
            yield
        finally:
            slot.release()
