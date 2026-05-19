"""End-to-end integration tests against a real Speaches backend.

Skipped by default. To run:

    docker compose -f docker-compose.speaches-cpu.yml up -d speaches
    pytest -m integration

Or, if Speaches is already running on localhost:8000, just `pytest -m integration`.
"""

from __future__ import annotations

import io
import wave

import pytest
from e2e_helpers import WyomingTestClient, fuzzy_text_match

pytestmark = pytest.mark.integration

MIN_TTS_PCM_BYTES = 8000


def _first_voice_name(info) -> str | None:
    for tts_program in info.tts:
        if tts_program.voices:
            return tts_program.voices[0].name
    return None


@pytest.mark.asyncio
async def test_describe_returns_speaches_models(wyoming_client: WyomingTestClient) -> None:
    info = await wyoming_client.describe()

    assert info.asr, "Info should advertise at least one ASR program"
    assert any(model.name for prog in info.asr for model in prog.models), "ASR program should list a model"
    assert info.tts, "Info should advertise at least one TTS program"
    assert _first_voice_name(info) is not None, "TTS program should list at least one voice"


@pytest.mark.asyncio
async def test_tts_structural(wyoming_client: WyomingTestClient) -> None:
    info = await wyoming_client.describe()
    voice = _first_voice_name(info)

    audio = await wyoming_client.synthesize("Hello, this is a test.", voice=voice)

    assert audio.chunk_count >= 1
    assert audio.rate > 0
    assert audio.width == 2
    assert audio.channels in (1, 2)
    assert len(audio.pcm) >= MIN_TTS_PCM_BYTES, (
        f"TTS produced suspiciously little audio: {len(audio.pcm)} bytes"
    )

    with wave.open(io.BytesIO(audio.to_wav_bytes()), "rb") as wav:
        assert wav.getframerate() == audio.rate
        assert wav.getnchannels() == audio.channels
        assert wav.getsampwidth() == audio.width


@pytest.mark.asyncio
async def test_tts_streaming_structural(wyoming_client: WyomingTestClient) -> None:
    info = await wyoming_client.describe()
    voice = _first_voice_name(info)

    audio = await wyoming_client.synthesize_streaming("Hello, this is a streaming test.", voice=voice)

    assert audio.chunk_count >= 1
    assert len(audio.pcm) >= MIN_TTS_PCM_BYTES


@pytest.mark.asyncio
async def test_stt_from_synthesized_audio(wyoming_client: WyomingTestClient) -> None:
    """The killer test: synthesize text, transcribe it, assert the loop closes.

    No human listening required - if TTS produces silence or STT returns
    garbage, fuzzy_text_match will catch it.
    """
    info = await wyoming_client.describe()
    voice = _first_voice_name(info)
    phrase = "The quick brown fox jumps over the lazy dog."

    audio = await wyoming_client.synthesize(phrase, voice=voice)
    transcript = await wyoming_client.transcribe(audio.to_wav_bytes())

    assert transcript.strip(), "STT returned empty transcript"
    assert fuzzy_text_match(transcript, phrase), (
        f"Round-trip transcript did not match.\n  expected: {phrase!r}\n  got:      {transcript!r}"
    )


@pytest.mark.asyncio
async def test_stt_round_trip_short_phrase(wyoming_client: WyomingTestClient) -> None:
    info = await wyoming_client.describe()
    voice = _first_voice_name(info)
    phrase = "Turn on the kitchen lights."

    audio = await wyoming_client.synthesize(phrase, voice=voice)
    transcript = await wyoming_client.transcribe(audio.to_wav_bytes())

    assert fuzzy_text_match(transcript, phrase), (
        f"Round-trip transcript did not match.\n  expected: {phrase!r}\n  got:      {transcript!r}"
    )


@pytest.mark.asyncio
async def test_backend_autodetection_speaches(wyoming_client_autodetect: WyomingTestClient) -> None:
    """Server started without STT_BACKEND/TTS_BACKEND should still come up.

    The autodetection probe targets the real Speaches /models endpoint, so a
    successful Describe + working synthesis proves the SPEACHES branch was
    selected and configured correctly.
    """
    info = await wyoming_client_autodetect.describe()
    assert info.asr, "Autodetected server should still advertise ASR"
    assert info.tts, "Autodetected server should still advertise TTS"

    voice = _first_voice_name(info)
    audio = await wyoming_client_autodetect.synthesize("Autodetect smoke test.", voice=voice)
    assert len(audio.pcm) >= MIN_TTS_PCM_BYTES
