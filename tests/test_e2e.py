"""End-to-end integration tests against a real Speaches backend.

Skipped by default. To run:

    docker compose -f docker-compose.speaches-cpu.yml -f docker-compose.e2e.yml up -d speaches
    pytest -m integration

Or, if Speaches is already running on localhost:8000, just `pytest -m integration`.
Tests marked `smoke` additionally run in CI against the containerized proxy
image (see .github/workflows/integration.yml).
"""

from __future__ import annotations

import io
import wave

import pytest
from e2e_helpers import WyomingTestClient, assert_fuzzy_match, save_debug_wav

pytestmark = pytest.mark.integration

MIN_TTS_PCM_BYTES = 8000

ROUND_TRIP_PHRASES = [
    pytest.param("The quick brown fox jumps over the lazy dog.", 0.7, id="pangram", marks=pytest.mark.smoke),
    pytest.param("Turn on the kitchen lights.", 0.7, id="command"),
    # Digits/homophones ("five" vs "5") get a slightly looser threshold.
    pytest.param("Set a timer for five minutes.", 0.65, id="digits"),
    pytest.param("What's the weather like tomorrow in Berlin?", 0.7, id="question"),
]


def _first_voice_name(info) -> str | None:
    for tts_program in info.tts:
        if tts_program.voices:
            return tts_program.voices[0].name
    return None


def _first_asr_model_name(info) -> str | None:
    for asr_program in info.asr:
        if asr_program.models:
            return asr_program.models[0].name
    return None


@pytest.mark.smoke
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

    # Event ordering: audio-start, then only chunks, then audio-stop.
    assert audio.event_types[0] == "audio-start"
    assert audio.event_types[-1] == "audio-stop"
    middle = audio.event_types[1:-1]
    assert middle, "expected at least one audio-chunk between start and stop"
    assert all(event_type == "audio-chunk" for event_type in middle), audio.event_types

    with wave.open(io.BytesIO(audio.to_wav_bytes()), "rb") as wav:
        assert wav.getframerate() == audio.rate
        assert wav.getnchannels() == audio.channels
        assert wav.getsampwidth() == audio.width


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_tts_streaming_structural(wyoming_client: WyomingTestClient) -> None:
    info = await wyoming_client.describe()
    voice = _first_voice_name(info)

    audio = await wyoming_client.synthesize_streaming("Hello, this is a streaming test.", voice=voice)

    assert audio.chunk_count >= 1
    assert len(audio.pcm) >= MIN_TTS_PCM_BYTES

    # Streaming synthesis ends with synthesize-stopped after the audio stream.
    assert audio.event_types[0] == "audio-start"
    assert audio.event_types[-1] == "synthesize-stopped"
    assert audio.event_types[-2] == "audio-stop"


@pytest.mark.asyncio
@pytest.mark.parametrize(("phrase", "threshold"), ROUND_TRIP_PHRASES)
async def test_stt_round_trip(wyoming_client: WyomingTestClient, phrase: str, threshold: float) -> None:
    """The killer test: synthesize text, transcribe it, assert the loop closes.

    No human listening required - if TTS produces silence or STT returns
    garbage, the fuzzy match will catch it.
    """
    info = await wyoming_client.describe()
    voice = _first_voice_name(info)

    audio = await wyoming_client.synthesize(phrase, voice=voice)
    save_debug_wav(f"round_trip_{phrase[:24]}", audio)
    result = await wyoming_client.transcribe(audio.to_wav_bytes())

    assert result.text.strip(), "STT returned empty transcript"
    assert_fuzzy_match(result.text, phrase, threshold=threshold)

    # Transcript event contract: starts with transcript-start, exactly one
    # final transcript, ends with transcript-stop; any transcript-chunks
    # (streaming models) fall in between.
    assert result.event_types[0] == "transcript-start", result.event_types
    assert result.event_types[-1] == "transcript-stop", result.event_types
    assert result.event_types.count("transcript") == 1, result.event_types


@pytest.mark.asyncio
async def test_stt_round_trip_streaming_synthesis(wyoming_client: WyomingTestClient) -> None:
    """Round trip via the streaming TTS path (SynthesizeStart/Chunk/Stop)."""
    info = await wyoming_client.describe()
    voice = _first_voice_name(info)
    phrase = "Streaming synthesis should still be understandable."

    audio = await wyoming_client.synthesize_streaming(phrase, voice=voice)
    save_debug_wav("round_trip_streaming", audio)
    result = await wyoming_client.transcribe(audio.to_wav_bytes())

    assert result.text.strip(), "STT returned empty transcript"
    assert_fuzzy_match(result.text, phrase)


@pytest.mark.asyncio
async def test_stt_with_explicit_model(wyoming_client: WyomingTestClient) -> None:
    """Transcribe with an explicit model name (exercises the Transcribe(name=...) path).

    The model name is read from Describe so this works in both subprocess and
    external (containerized) modes.
    """
    info = await wyoming_client.describe()
    voice = _first_voice_name(info)
    model = _first_asr_model_name(info)
    assert model is not None, "No ASR model advertised"
    phrase = "Explicit model selection works."

    audio = await wyoming_client.synthesize(phrase, voice=voice)
    result = await wyoming_client.transcribe(audio.to_wav_bytes(), model=model)

    assert_fuzzy_match(result.text, phrase)


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


@pytest.mark.asyncio
async def test_select_program_round_trip(wyoming_client: WyomingTestClient) -> None:
    """select-program (wyoming 1.10.0) pins the advertised program; flows still work.

    The e2e server advertises one program per domain, so this proves real-socket
    dispatch of the new event; cross-program restriction is covered by unit tests.
    """
    info = await wyoming_client.describe()
    program_name = info.asr[0].name
    assert program_name is not None

    await wyoming_client.select_program(program_name)

    phrase = "Program selection round trip."
    voice = _first_voice_name(info)
    audio = await wyoming_client.synthesize(phrase, voice=voice)
    result = await wyoming_client.transcribe(audio.to_wav_bytes())

    assert_fuzzy_match(result.text, phrase)


@pytest.mark.asyncio
async def test_synthesize_ssml_round_trip(wyoming_client: WyomingTestClient) -> None:
    """SSML tags must be stripped, not spoken: spoken markup would tank the match."""
    info = await wyoming_client.describe()
    voice = _first_voice_name(info)
    plain_phrase = "The quick brown fox jumps over the lazy dog."
    ssml_text = f'<speak>{plain_phrase}<break time="300ms"/></speak>'

    audio = await wyoming_client.synthesize(ssml_text, voice=voice, text_format="ssml")
    save_debug_wav("ssml_round_trip", audio)
    result = await wyoming_client.transcribe(audio.to_wav_bytes())

    assert_fuzzy_match(result.text, plain_phrase)


@pytest.mark.asyncio
async def test_transcribe_with_new_fields_round_trip(wyoming_client: WyomingTestClient) -> None:
    """vad_sensitivity/transcript_names/transcript_terms are ignored without disturbing STT."""
    info = await wyoming_client.describe()
    voice = _first_voice_name(info)
    phrase = "What's the weather like tomorrow in Berlin?"

    audio = await wyoming_client.synthesize(phrase, voice=voice)
    result = await wyoming_client.transcribe(
        audio.to_wav_bytes(),
        vad_sensitivity="aggressive",
        transcript_names=["Berlin"],
        transcript_terms=["weather"],
    )

    assert_fuzzy_match(result.text, phrase)
