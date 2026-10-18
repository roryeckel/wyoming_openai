"""Negative end-to-end tests: the server's rejection contract.

wyoming_openai rejects unknown models/voices with a Wyoming error event, then
returns False from its event handler, which makes the wyoming server close the
TCP connection. Home Assistant reports either as a failure. These tests codify
that contract by requiring the error event and then the closure: if a future
change drops the error, answers with something else, hangs, or stops closing
the socket, they fail loudly so the contract change is a conscious one.

Each test uses a fresh connection (a dead connection must not poison the
shared client fixture) and a short event timeout.
"""

from __future__ import annotations

import pytest
from e2e_helpers import WyomingTestClient
from wyoming.asr import Transcribe
from wyoming.tts import Synthesize, SynthesizeVoice

pytestmark = pytest.mark.integration

NEGATIVE_EVENT_TIMEOUT_SECONDS = 10.0


@pytest.mark.asyncio
async def test_unknown_voice_closes_connection(wyoming_endpoint: tuple[str, int]) -> None:
    host, port = wyoming_endpoint
    async with WyomingTestClient(
        host=host, port=port, event_timeout=NEGATIVE_EVENT_TIMEOUT_SECONDS
    ) as client:
        await client.expect_rejection(
            Synthesize(text="hello", voice=SynthesizeVoice(name="definitely-not-a-voice (nope)")).event()
        )


@pytest.mark.asyncio
async def test_unknown_stt_model_closes_connection(wyoming_endpoint: tuple[str, int]) -> None:
    host, port = wyoming_endpoint
    async with WyomingTestClient(
        host=host, port=port, event_timeout=NEGATIVE_EVENT_TIMEOUT_SECONDS
    ) as client:
        # Refused at transcribe, so the audio a client would send next is not needed
        await client.expect_rejection(Transcribe(name="definitely-not-a-model", language="en").event())


@pytest.mark.asyncio
async def test_select_program_unknown_name_is_noop(wyoming_endpoint: tuple[str, int]) -> None:
    """Unlike unknown models/voices, an unknown select-program name must NOT close
    the connection: the Wyoming protocol says servers drop unrecognized selections
    and keep using the default program."""
    host, port = wyoming_endpoint
    async with WyomingTestClient(
        host=host, port=port, event_timeout=NEGATIVE_EVENT_TIMEOUT_SECONDS
    ) as client:
        await client.select_program("definitely-not-a-program")
        info = await client.describe()
        voice = info.tts[0].voices[0].name if info.tts and info.tts[0].voices else None
        audio = await client.synthesize("Still alive after unknown selection.", voice=voice)
        assert audio.chunk_count >= 1
