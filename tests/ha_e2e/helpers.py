"""Pure helpers for the Home Assistant end-to-end harness."""

from __future__ import annotations

import io
import re
import shutil
import struct
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from e2e_helpers import FuzzyMatchResult, fuzzy_match

WAV_STREAM_SENTINEL = 0xFFFFFFFF

_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_NUMBER_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class WavData:
    """PCM audio and its format."""

    rate: int
    width: int
    channels: int
    pcm: bytes

    @property
    def duration_seconds(self) -> float:
        bytes_per_second = self.rate * self.width * self.channels
        if bytes_per_second <= 0:
            return 0.0
        return len(self.pcm) / bytes_per_second

    def to_wav_bytes(self) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(self.channels)
            wav.setsampwidth(self.width)
            wav.setframerate(self.rate)
            wav.writeframes(self.pcm)
        return buffer.getvalue()


def read_wav(path: Path) -> WavData:
    """Read a complete, conventional WAV file."""

    with wave.open(str(path), "rb") as wav:
        return WavData(
            rate=wav.getframerate(),
            width=wav.getsampwidth(),
            channels=wav.getnchannels(),
            pcm=wav.readframes(wav.getnframes()),
        )


def parse_wav_bytes(data: bytes) -> WavData:
    """Parse a WAV body, including the nframes=0 header HA uses for streaming TTS.

    ``wave`` trusts the data-chunk length in the header.  Home Assistant's
    streaming TTS proxy forwards the Wyoming WAV header, which declares zero
    frames, and then the PCM follows as chunked HTTP data.  Walking the RIFF
    chunks ourselves lets us keep that PCM.
    """

    if len(data) < 12 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("response is not a RIFF/WAVE file")

    pos = 12
    rate: int | None = None
    width: int | None = None
    channels: int | None = None
    data_offset: int | None = None
    declared_data_size: int | None = None

    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        chunk_size = struct.unpack_from("<I", data, pos + 4)[0]
        body = pos + 8
        if chunk_id == b"fmt ":
            if chunk_size < 16 or body + 16 > len(data):
                raise ValueError("truncated WAV fmt chunk")
            audio_format, channel_count, sample_rate, _, _, bits = struct.unpack_from("<HHIIHH", data, body)
            if audio_format != 1:
                raise ValueError(f"unsupported WAV format {audio_format}")
            if channel_count <= 0 or sample_rate <= 0 or bits % 8:
                raise ValueError("invalid WAV format fields")
            rate = sample_rate
            channels = channel_count
            width = bits // 8
        elif chunk_id == b"data":
            data_offset = body
            declared_data_size = chunk_size
            break

        if chunk_size == WAV_STREAM_SENTINEL:
            raise ValueError("unbounded RIFF chunk before the data chunk")
        pos = body + chunk_size + (chunk_size & 1)

    if rate is None or width is None or channels is None:
        raise ValueError("WAV file has no fmt chunk")
    if data_offset is None:
        raise ValueError("WAV file has no data chunk")

    pcm = data[data_offset:]
    if declared_data_size is not None and 0 < declared_data_size <= len(pcm):
        pcm = pcm[:declared_data_size]

    if not pcm:
        raise ValueError("WAV file has no PCM data")

    return WavData(rate=rate, width=width, channels=channels, pcm=pcm)


def save_wav(path: Path, audio: WavData) -> None:
    """Write a canonical WAV file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audio.to_wav_bytes())


def decode_audio_bytes(data: bytes) -> WavData:
    """Decode a WAV or a Home Assistant mp3 TTS response into PCM.

    Home Assistant's TTS proxy defaults to mp3 unless the pipeline asks for a
    preferred format.  The Assist websocket API does not expose a preferred
    format for pipeline runs, so the harness decodes the fetched response with
    ffmpeg (available on the maintainer's machine and on GitHub's Ubuntu
    runners).
    """

    if data.startswith(b"RIFF"):
        return parse_wav_bytes(data)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "Home Assistant returned non-WAV TTS audio and ffmpeg is not installed; "
            "install ffmpeg or make the pipeline return WAV"
        )
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-f",
            "wav",
            "pipe:1",
        ],
        input=data,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode(errors="replace")[:1000]
        raise ValueError(f"ffmpeg could not decode the TTS response: {detail}")
    return parse_wav_bytes(result.stdout)


def _int_to_words(value: int) -> str:
    if value < 0:
        return "minus " + _int_to_words(-value)
    if value < 20:
        return _ONES[value]
    if value < 100:
        tens, ones = divmod(value, 10)
        return _TENS[tens] if not ones else f"{_TENS[tens]} {_ONES[ones]}"
    if value < 1000:
        hundreds, remainder = divmod(value, 100)
        text = f"{_ONES[hundreds]} hundred"
        return text if not remainder else f"{text} {_int_to_words(remainder)}"
    if value < 1_000_000:
        thousands, remainder = divmod(value, 1000)
        text = f"{_int_to_words(thousands)} thousand"
        return text if not remainder else f"{text} {_int_to_words(remainder)}"
    return str(value)


def normalize_numbers(text: str) -> str:
    """Replace digit runs with English words so STT may return either form."""

    return _NUMBER_RE.sub(lambda match: _int_to_words(int(match.group())), text)


def assert_transcript_match(transcript: str, expected: str) -> FuzzyMatchResult:
    """Fuzzy-match an STT transcript against the fixture phrase."""

    result = fuzzy_match(transcript, expected)
    assert result.matched, (
        f"STT transcript did not match (score={result.score:.1f}, coverage={result.coverage:.2f})\n"
        f"  expected: {result.normalized_expected!r}\n"
        f"  got:      {result.normalized_transcript!r}"
    )
    return result


def assert_reply_match(transcript: str, reply: str, *, long_reply: bool = False) -> FuzzyMatchResult:
    """Fuzzy-match the TTS round-trip transcript against the scripted reply."""

    normalized_transcript = normalize_numbers(transcript)
    normalized_reply = normalize_numbers(reply)
    coverage_threshold = 0.9 if long_reply else 0.85
    result = fuzzy_match(
        normalized_transcript,
        normalized_reply,
        threshold=0.6,
        coverage_threshold=coverage_threshold,
    )
    assert result.matched, (
        f"TTS round-trip transcript did not match the reply "
        f"(score={result.score:.1f}, coverage={result.coverage:.2f})\n"
        f"  expected: {result.normalized_expected!r}\n"
        f"  got:      {result.normalized_transcript!r}"
    )

    if long_reply:
        expected_tail = " ".join(normalized_reply.split()[-80:])
        transcript_tail = " ".join(normalized_transcript.split()[-80:])
        tail = fuzzy_match(transcript_tail, expected_tail, threshold=0.6, coverage_threshold=0.75)
        assert tail.matched, (
            "TTS round-trip transcript is missing the end of the long reply "
            f"(tail score={tail.score:.1f}, coverage={tail.coverage:.2f})"
        )

    return result


def assert_plausible_duration(audio: WavData, text: str) -> float:
    """Return the duration and reject empty, truncated, or absurd TTS output.

    Bounds are deliberately generous: 0.08 s/word catches a stream cut off in
    the first sentence of a long reply, while 1.5 s/word allows very slow
    speech.
    """

    duration = audio.duration_seconds
    words = max(1, len(text.split()))
    minimum = max(0.35, words * 0.08)
    maximum = max(5.0, words * 1.5)
    assert minimum <= duration <= maximum, (
        f"TTS audio duration {duration:.2f}s is not plausible for {words} words "
        f"(expected {minimum:.2f}s..{maximum:.2f}s)"
    )
    return duration


def short_text(text: str, limit: int = 90) -> str:
    """Shorten text for a markdown table cell."""

    single_line = " ".join(text.split())
    if len(single_line) <= limit:
        return single_line
    return single_line[: limit - 1] + "…"
