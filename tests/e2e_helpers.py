"""End-to-end test helpers for the Wyoming OpenAI proxy.

These utilities drive a real running wyoming_openai TCP server from pytest,
standing in for Home Assistant. They are only used by tests under the
`integration` marker.
"""

from __future__ import annotations

import asyncio
import io
import os
import re
import string
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from rapidfuzz import fuzz
from wyoming.asr import Transcribe, Transcript, TranscriptChunk, TranscriptStart, TranscriptStop
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
    SynthesizeVoice,
)

AUDIO_CHUNK_FRAMES = 1024
DEFAULT_EVENT_TIMEOUT_SECONDS = 30.0
DEFAULT_OPERATION_TIMEOUT_SECONDS = 120.0

# When set, save_debug_wav() writes synthesized audio here so CI can upload it
# as an artifact for human listening when a round-trip assertion fails.
ARTIFACT_DIR_ENV_VAR = "WYOMING_E2E_ARTIFACT_DIR"


class WyomingServerClosedError(RuntimeError):
    """Server closed the TCP connection before sending the expected event.

    wyoming_openai rejects unknown models/voices by returning False from its
    event handler, which makes the wyoming server close the socket without any
    error event - so this is the expected failure mode for negative tests.
    """


class WyomingTimeoutError(TimeoutError):
    """Server did not send the expected event in time."""


@dataclass
class SynthesizedAudio:
    """A complete audio response captured from the TTS pipeline."""

    rate: int
    width: int
    channels: int
    pcm: bytes
    chunk_count: int
    event_types: list[str] = field(default_factory=list)

    def to_wav_bytes(self) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(self.channels)
            wav.setsampwidth(self.width)
            wav.setframerate(self.rate)
            wav.writeframes(self.pcm)
        return buffer.getvalue()


@dataclass
class TranscriptionResult:
    """A complete transcription response captured from the STT pipeline."""

    text: str
    event_types: list[str] = field(default_factory=list)


class WyomingTestClient:
    """Convenience wrapper around `wyoming.client.AsyncTcpClient`.

    Exposes describe/transcribe/synthesize methods that mirror what a Wyoming
    consumer like Home Assistant would do, so tests can verify the full TCP
    surface without depending on HA itself.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 10300,
        event_timeout: float = DEFAULT_EVENT_TIMEOUT_SECONDS,
        operation_timeout: float = DEFAULT_OPERATION_TIMEOUT_SECONDS,
    ) -> None:
        self.host = host
        self.port = port
        self._event_timeout = event_timeout
        self._operation_timeout = operation_timeout
        self._client: AsyncTcpClient | None = None

    async def __aenter__(self) -> WyomingTestClient:
        self._client = AsyncTcpClient(self.host, self.port)
        await self._client.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            await self._client.disconnect()
            self._client = None

    @property
    def client(self) -> AsyncTcpClient:
        if self._client is None:
            raise RuntimeError("WyomingTestClient must be used as an async context manager")
        return self._client

    async def _write_event(self, event: Event) -> None:
        """Write an event, normalizing write-side disconnects.

        When the server rejects an earlier event it closes the connection
        immediately; depending on timing, a subsequent write can then raise a
        connection error before any read observes the closure. Translate those
        into WyomingServerClosedError so callers (especially negative tests)
        see one consistent failure mode.
        """
        try:
            await self.client.write_event(event)
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError) as exc:
            raise WyomingServerClosedError(f"Connection closed while writing {event.type}") from exc

    async def _read_required(self, expecting: str) -> Event:
        try:
            event = await asyncio.wait_for(self.client.read_event(), timeout=self._event_timeout)
        except TimeoutError as exc:
            raise WyomingTimeoutError(
                f"No event within {self._event_timeout}s while waiting for {expecting}"
            ) from exc
        if event is None:
            raise WyomingServerClosedError(f"Connection closed while waiting for {expecting}")
        return event

    async def describe(self) -> Info:
        await self._write_event(Describe().event())
        async with asyncio.timeout(self._operation_timeout):
            while True:
                event = await self._read_required("info")
                if Info.is_type(event.type):
                    return Info.from_event(event)

    async def transcribe(
        self, wav_bytes: bytes, model: str | None = None, language: str = "en"
    ) -> TranscriptionResult:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            rate = wav.getframerate()
            width = wav.getsampwidth()
            channels = wav.getnchannels()
            pcm = wav.readframes(wav.getnframes())

        await self._write_event(Transcribe(name=model, language=language).event())
        await self._write_event(AudioStart(rate=rate, width=width, channels=channels).event())

        bytes_per_frame = width * channels
        chunk_size = AUDIO_CHUNK_FRAMES * bytes_per_frame
        for start in range(0, len(pcm), chunk_size):
            chunk = pcm[start : start + chunk_size]
            await self._write_event(
                AudioChunk(rate=rate, width=width, channels=channels, audio=chunk).event()
            )
        await self._write_event(AudioStop().event())

        event_types: list[str] = []
        streamed_text = ""
        final_text: str | None = None
        async with asyncio.timeout(self._operation_timeout):
            while True:
                event = await self._read_required("transcript events")
                event_types.append(event.type)
                if TranscriptChunk.is_type(event.type):
                    streamed_text += TranscriptChunk.from_event(event).text
                elif Transcript.is_type(event.type):
                    final_text = Transcript.from_event(event).text
                elif TranscriptStop.is_type(event.type):
                    break
                elif not TranscriptStart.is_type(event.type):
                    raise RuntimeError(f"Unexpected event during transcription: {event.type}")
        text = final_text if final_text is not None else streamed_text
        return TranscriptionResult(text=text, event_types=event_types)

    async def synthesize(self, text: str, voice: str | None = None) -> SynthesizedAudio:
        synth_voice = SynthesizeVoice(name=voice) if voice else None
        await self._write_event(Synthesize(text=text, voice=synth_voice).event())
        return await self._collect_audio()

    async def synthesize_streaming(self, text: str, voice: str | None = None) -> SynthesizedAudio:
        synth_voice = SynthesizeVoice(name=voice) if voice else None
        await self._write_event(SynthesizeStart(voice=synth_voice).event())
        await self._write_event(SynthesizeChunk(text=text).event())
        await self._write_event(SynthesizeStop().event())
        return await self._collect_audio(expect_synthesize_stopped=True)

    async def _collect_audio(self, expect_synthesize_stopped: bool = False) -> SynthesizedAudio:
        rate: int | None = None
        width: int | None = None
        channels: int | None = None
        pcm = bytearray()
        chunk_count = 0
        event_types: list[str] = []
        async with asyncio.timeout(self._operation_timeout):
            while True:
                event = await self._read_required("audio events")
                event_types.append(event.type)
                if AudioStart.is_type(event.type):
                    start = AudioStart.from_event(event)
                    rate, width, channels = start.rate, start.width, start.channels
                    continue
                if AudioChunk.is_type(event.type):
                    chunk = AudioChunk.from_event(event)
                    pcm.extend(chunk.audio)
                    chunk_count += 1
                    if rate is None:
                        rate, width, channels = chunk.rate, chunk.width, chunk.channels
                    continue
                if AudioStop.is_type(event.type):
                    break
            if expect_synthesize_stopped:
                while True:
                    event = await self._read_required("synthesize-stopped")
                    event_types.append(event.type)
                    if SynthesizeStopped.is_type(event.type):
                        break
        if rate is None or width is None or channels is None:
            raise RuntimeError("TTS response did not include audio format information")
        return SynthesizedAudio(
            rate=rate,
            width=width,
            channels=channels,
            pcm=bytes(pcm),
            chunk_count=chunk_count,
            event_types=event_types,
        )

    async def stream_audio_chunks(self, text: str, voice: str | None = None) -> AsyncIterator[Event]:
        synth_voice = SynthesizeVoice(name=voice) if voice else None
        await self._write_event(Synthesize(text=text, voice=synth_voice).event())
        while True:
            event = await self._read_required("audio events")
            yield event
            if AudioStop.is_type(event.type):
                return


_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


@dataclass
class FuzzyMatchResult:
    matched: bool
    score: float  # rapidfuzz partial_ratio, 0-100
    coverage: float
    normalized_transcript: str
    normalized_expected: str


def fuzzy_match(
    transcript: str, expected: str, threshold: float = 0.7, coverage_threshold: float = 0.75
) -> FuzzyMatchResult:
    """Compare `transcript` to `expected` after lowercasing/punctuation strip.

    Uses rapidfuzz partial_ratio so leading/trailing artifacts (e.g. Whisper
    capitalizing the first word, dropping a trailing period, or hallucinating
    a leading "..." token) don't fail the match. Thresholds are 0-1 fractions.
    The coverage check rejects short substrings that would otherwise get a
    perfect partial_ratio score.
    """
    normalized_transcript = _normalize(transcript) if transcript else ""
    normalized_expected = _normalize(expected) if expected else ""
    if not normalized_transcript or not normalized_expected:
        return FuzzyMatchResult(
            matched=False,
            score=0.0,
            coverage=0.0,
            normalized_transcript=normalized_transcript,
            normalized_expected=normalized_expected,
        )

    coverage = min(len(normalized_transcript), len(normalized_expected)) / len(normalized_expected)
    score = float(fuzz.partial_ratio(normalized_transcript, normalized_expected))
    matched = coverage >= coverage_threshold and score >= threshold * 100
    return FuzzyMatchResult(
        matched=matched,
        score=score,
        coverage=coverage,
        normalized_transcript=normalized_transcript,
        normalized_expected=normalized_expected,
    )


def fuzzy_text_match(
    transcript: str, expected: str, threshold: float = 0.7, coverage_threshold: float = 0.75
) -> bool:
    """True if `transcript` matches `expected`. See fuzzy_match for details."""
    return fuzzy_match(transcript, expected, threshold, coverage_threshold).matched


def assert_fuzzy_match(
    transcript: str, expected: str, threshold: float = 0.7, coverage_threshold: float = 0.75
) -> None:
    """Assert a fuzzy match, reporting score/coverage details on failure."""
    result = fuzzy_match(transcript, expected, threshold, coverage_threshold)
    assert result.matched, (
        f"Round-trip transcript did not match "
        f"(score={result.score:.1f}, need>={threshold * 100:.0f}, "
        f"coverage={result.coverage:.2f}, need>={coverage_threshold:.2f})\n"
        f"  expected: {result.normalized_expected!r}\n"
        f"  got:      {result.normalized_transcript!r}"
    )


def save_debug_wav(name: str, audio: SynthesizedAudio) -> None:
    """Write synthesized audio to $WYOMING_E2E_ARTIFACT_DIR for CI artifact upload.

    No-op when the environment variable is unset (local runs).
    """
    artifact_dir = os.environ.get(ARTIFACT_DIR_ENV_VAR)
    if not artifact_dir:
        return
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "audio"
    path = Path(artifact_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{safe_name}.wav").write_bytes(audio.to_wav_bytes())
