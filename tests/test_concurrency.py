import asyncio
import logging

import pytest

from wyoming_openai.concurrency import ConcurrencyLimit


async def _settle():
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_zero_means_no_limit():
    limit = ConcurrencyLimit(0)

    slots = [await asyncio.wait_for(limit.acquire(), timeout=1) for _ in range(100)]

    assert not limit.full
    for slot in slots:
        slot.release()


@pytest.mark.asyncio
async def test_a_request_beyond_the_limit_waits_for_a_slot():
    limit = ConcurrencyLimit(2)
    first = await limit.acquire()
    await limit.acquire()
    assert limit.full

    waiting = asyncio.create_task(limit.acquire())
    await _settle()
    assert not waiting.done()

    first.release()
    await asyncio.wait_for(waiting, timeout=1)


@pytest.mark.asyncio
async def test_releasing_a_slot_twice_frees_it_once():
    limit = ConcurrencyLimit(1)
    slot = await limit.acquire()

    slot.release()
    slot.release()

    await limit.acquire()
    assert limit.full


@pytest.mark.asyncio
async def test_slot_block_releases_its_slot_when_it_fails():
    limit = ConcurrencyLimit(1)

    with pytest.raises(RuntimeError):
        async with limit.slot():
            assert limit.full
            raise RuntimeError("request failed")

    assert not limit.full


@pytest.mark.asyncio
async def test_a_cancelled_wait_takes_no_slot():
    limit = ConcurrencyLimit(1)
    held = await limit.acquire()
    waiting = asyncio.create_task(limit.acquire())
    await _settle()

    waiting.cancel()
    await asyncio.gather(waiting, return_exceptions=True)
    held.release()

    assert not limit.full


@pytest.mark.asyncio
async def test_a_named_limit_logs_a_request_that_has_to_wait(caplog):
    limit = ConcurrencyLimit(1, name="STT request")
    held = await limit.acquire()

    with caplog.at_level(logging.INFO, logger="wyoming_openai.concurrency"):
        waiting = asyncio.create_task(limit.acquire())
        await _settle()
        held.release()
        await asyncio.wait_for(waiting, timeout=1)

    assert "Waiting for a free STT request slot: all 1 are in use" in caplog.text


def test_a_negative_limit_is_rejected():
    with pytest.raises(ValueError, match="must be at least 0"):
        ConcurrencyLimit(-1)
