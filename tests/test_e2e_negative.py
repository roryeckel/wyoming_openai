"""Negative end-to-end tests: the server's rejection contract.

wyoming_openai rejects unknown models/voices by returning False from its event
handler, which makes the wyoming server close the TCP connection without any
error event. These tests codify that contract by requiring the connection
closure specifically: if a future change makes the server answer with an error
event, hang, or otherwise stop closing the socket, they fail loudly
(WyomingTimeoutError or an unexpected-event error) so the contract change is a
conscious one.

Each test uses a fresh connection (a dead connection must not poison the
shared client fixture) and a short event timeout.
"""

from __future__ import annotations

import io
import wave

import pytest
from e2e_helpers import WyomingServerClosedError, WyomingTestClient

pytestmark = pytest.mark.integration

NEGATIVE_EVENT_TIMEOUT_SECONDS = 10.0


@pytest.mark.asyncio
async def test_unknown_voice_closes_connection(wyoming_endpoint: tuple[str, int]) -> None:
    host, port = wyoming_endpoint
    async with WyomingTestClient(
        host=host, port=port, event_timeout=NEGATIVE_EVENT_TIMEOUT_SECONDS
    ) as client:
        with pytest.raises(WyomingServerClosedError):
            await client.synthesize("hello", voice="definitely-not-a-voice (nope)")


@pytest.mark.asyncio
async def test_unknown_stt_model_closes_connection(wyoming_endpoint: tuple[str, int]) -> None:
    host, port = wyoming_endpoint
    # Any WAV payload will do; the server should reject at the Transcribe event.
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)

    async with WyomingTestClient(
        host=host, port=port, event_timeout=NEGATIVE_EVENT_TIMEOUT_SECONDS
    ) as client:
        with pytest.raises(WyomingServerClosedError):
            await client.transcribe(buffer.getvalue(), model="definitely-not-a-model")
