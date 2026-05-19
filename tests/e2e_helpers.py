"""End-to-end test helpers for the Wyoming OpenAI proxy.

These utilities drive a real running wyoming_openai TCP server from pytest,
standing in for Home Assistant. They are only used by tests under the
`integration` marker.
"""

from __future__ import annotations

import io
import re
import string
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass

from rapidfuzz import fuzz
from wyoming.asr import Transcribe, Transcript, TranscriptChunk, TranscriptStart, TranscriptStop
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeVoice

AUDIO_CHUNK_FRAMES = 1024


@dataclass
class SynthesizedAudio:
    """A complete audio response captured from the TTS pipeline."""

    rate: int
    width: int
    channels: int
    pcm: bytes
    chunk_count: int

    def to_wav_bytes(self) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(self.channels)
            wav.setsampwidth(self.width)
            wav.setframerate(self.rate)
            wav.writeframes(self.pcm)
        return buffer.getvalue()


class WyomingTestClient:
    """Convenience wrapper around `wyoming.client.AsyncTcpClient`.

    Exposes describe/transcribe/synthesize methods that mirror what a Wyoming
    consumer like Home Assistant would do, so tests can verify the full TCP
    surface without depending on HA itself.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 10300) -> None:
        self.host = host
        self.port = port
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

    async def _read_required(self) -> Event:
        event = await self.client.read_event()
        if event is None:
            raise RuntimeError("Server closed the connection before sending the expected event")
        return event

    async def describe(self) -> Info:
        await self.client.write_event(Describe().event())
        while True:
            event = await self._read_required()
            if Info.is_type(event.type):
                return Info.from_event(event)

    async def transcribe(self, wav_bytes: bytes, model: str | None = None, language: str = "en") -> str:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            rate = wav.getframerate()
            width = wav.getsampwidth()
            channels = wav.getnchannels()
            pcm = wav.readframes(wav.getnframes())

        await self.client.write_event(Transcribe(name=model, language=language).event())
        await self.client.write_event(AudioStart(rate=rate, width=width, channels=channels).event())

        bytes_per_frame = width * channels
        chunk_size = AUDIO_CHUNK_FRAMES * bytes_per_frame
        for start in range(0, len(pcm), chunk_size):
            chunk = pcm[start : start + chunk_size]
            await self.client.write_event(
                AudioChunk(rate=rate, width=width, channels=channels, audio=chunk).event()
            )
        await self.client.write_event(AudioStop().event())

        streamed_text = ""
        while True:
            event = await self._read_required()
            if TranscriptStart.is_type(event.type):
                continue
            if TranscriptChunk.is_type(event.type):
                streamed_text += TranscriptChunk.from_event(event).text
                continue
            if Transcript.is_type(event.type):
                return Transcript.from_event(event).text
            if TranscriptStop.is_type(event.type):
                return streamed_text

    async def synthesize(self, text: str, voice: str | None = None) -> SynthesizedAudio:
        synth_voice = SynthesizeVoice(name=voice) if voice else None
        await self.client.write_event(Synthesize(text=text, voice=synth_voice).event())
        return await self._collect_audio()

    async def synthesize_streaming(self, text: str, voice: str | None = None) -> SynthesizedAudio:
        synth_voice = SynthesizeVoice(name=voice) if voice else None
        await self.client.write_event(SynthesizeStart(voice=synth_voice).event())
        await self.client.write_event(SynthesizeChunk(text=text).event())
        await self.client.write_event(SynthesizeStop().event())
        return await self._collect_audio()

    async def _collect_audio(self) -> SynthesizedAudio:
        rate: int | None = None
        width: int | None = None
        channels: int | None = None
        pcm = bytearray()
        chunk_count = 0
        while True:
            event = await self._read_required()
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
        if rate is None or width is None or channels is None:
            raise RuntimeError("TTS response did not include audio format information")
        return SynthesizedAudio(rate=rate, width=width, channels=channels, pcm=bytes(pcm), chunk_count=chunk_count)

    async def stream_audio_chunks(self, text: str, voice: str | None = None) -> AsyncIterator[Event]:
        synth_voice = SynthesizeVoice(name=voice) if voice else None
        await self.client.write_event(Synthesize(text=text, voice=synth_voice).event())
        while True:
            event = await self._read_required()
            yield event
            if AudioStop.is_type(event.type):
                return


_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def fuzzy_text_match(transcript: str, expected: str, threshold: float = 0.7, coverage_threshold: float = 0.75) -> bool:
    """True if `transcript` matches `expected` after lowercasing/punctuation strip.

    Uses rapidfuzz partial_ratio so leading/trailing artifacts (e.g. Whisper
    capitalizing the first word, dropping a trailing period, or hallucinating
    a leading "..." token) don't fail the match. Thresholds are 0-1 fractions.
    """
    if not transcript or not expected:
        return False

    normalized_transcript = _normalize(transcript)
    normalized_expected = _normalize(expected)
    if not normalized_transcript or not normalized_expected:
        return False

    coverage = min(len(normalized_transcript), len(normalized_expected)) / len(normalized_expected)
    if coverage < coverage_threshold:
        return False

    score = fuzz.partial_ratio(normalized_transcript, normalized_expected)
    return score >= threshold * 100
