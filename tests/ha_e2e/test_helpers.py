"""Unit tests for the HA E2E helpers (run in the default suite)."""

from __future__ import annotations

import io
import wave

import pytest

from ha_e2e.helpers import (
    WavData,
    assert_plausible_duration,
    normalize_numbers,
    parse_wav_bytes,
)


def _streaming_wav(rate: int, channels: int, pcm: bytes) -> bytes:
    """Build the nframes=0 WAV header Home Assistant's streaming TTS emits."""

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
    return buffer.getvalue() + pcm


def test_normalize_numbers_handles_stt_word_forms():
    assert normalize_numbers("There are 3 apples, 42 oranges, and 7 pears.") == (
        "There are three apples, forty two oranges, and seven pears."
    )
    assert normalize_numbers("The oven is at 350 degrees.") == "The oven is at three hundred fifty degrees."


def test_parse_wav_bytes_keeps_pcm_after_a_zero_frame_header():
    pcm = b"\x01\x00" * 1600
    audio = parse_wav_bytes(_streaming_wav(16000, 1, pcm))

    assert audio.rate == 16000
    assert audio.channels == 1
    assert audio.width == 2
    assert audio.pcm == pcm
    assert audio.duration_seconds == pytest.approx(0.1)


def test_parse_wav_bytes_handles_a_complete_wav():
    audio = WavData(rate=24000, width=2, channels=1, pcm=b"\x02\x00" * 2400)
    parsed = parse_wav_bytes(audio.to_wav_bytes())

    assert parsed == audio


def test_parse_wav_bytes_rejects_non_wav():
    with pytest.raises(ValueError):
        parse_wav_bytes(b"not a wav")


def test_assert_plausible_duration_rejects_truncation():
    audio = WavData(rate=24000, width=2, channels=1, pcm=b"\x00\x00" * 2400)  # 0.1s
    with pytest.raises(AssertionError):
        assert_plausible_duration(audio, " ".join(["word"] * 100))
