"""
Early transcription: start the speech-to-text request when the speaker pauses instead of at Wyoming's ``AudioStop``.

Home Assistant only sends ``AudioStop`` after its own end-of-speech detection, which by default waits for 0.7 s of
silence, and a plain proxy only then sends the recording to the speech-to-text backend. ``EarlyTranscription`` listens
to the audio while it arrives. After speech followed by a short pause it sends the audio received so far to the backend
right away. At ``AudioStop`` that answer is used if nothing but silence arrived since the request started. If the
speaker carried on, the early answer is dropped and the handler sends its normal request with the complete recording.

The pause detector compares the loudness of each chunk with the background level. It needs no model or extra package.
"""

import array
import asyncio
import logging
import math
import operator
import sys
import time
import wave
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any

from .utilities import NamedBytesIO

_LOGGER = logging.getLogger(__name__)

EARLY_SILENCE_MS = 300  # Quiet time after speech that starts an early request
EARLY_MIN_SPEECH_MS = 250  # Loud time needed before a pause counts as the end of speech
EARLY_RESUME_MS = 100  # Sound after an early request started that makes the request stale
EARLY_MAX_REQUESTS = 3  # Early requests per recording (every pause after speech can start one)
REFERENCE_CHUNK_MS = 32  # Chunk length that the background-level adaptation rates below are expressed in


class Signal(Enum):
    """What a chunk of audio changed about the recording."""

    NONE = "none"
    PAUSE = "pause"  # Speech was followed by EARLY_SILENCE_MS of quiet: an early request can start
    RESUMED = "resumed"  # Sound came back after a pause: the early request no longer covers the recording


def _rms(audio: bytes) -> float:
    """Root-mean-square level of little-endian 16-bit PCM (0.0 for empty input)."""
    if len(audio) < 2:
        return 0.0
    samples = array.array("h")
    samples.frombytes(audio[: len(audio) // 2 * 2])
    if sys.byteorder == "big":
        samples.byteswap()
    return math.sqrt(sum(map(operator.mul, samples, samples)) / len(samples))


class PauseDetector:
    """
    Finds pauses after speech in a stream of 16-bit PCM audio.

    Two thresholds are used on purpose. A chunk is *loud* when it is clearly above the background level (speech) and
    *quiet* when it is clearly at the background level (silence). A chunk in between, such as a soft word ending or a
    breath, is neither: it can never start an early request and it makes a started one stale. The background level
    follows the quietest recent chunk and the speech peak fades slowly, so no calibration is needed.
    """

    def __init__(self, *, rate: int, width: int, channels: int) -> None:
        if width != 2 or rate <= 0 or channels <= 0:
            raise ValueError("Pause detection needs 16-bit PCM audio")
        self._ms_per_byte = 1000 / (rate * width * channels)
        self._noise_floor: float | None = None
        self._peak = 0.0
        self._loud_ms = 0.0
        self._quiet_ms = 0.0
        self._sound_after_pause_ms = 0.0
        self._pauses = 0
        self._in_pause = False

    def feed(self, audio: bytes) -> Signal:
        """Process the next chunk of audio and report whether it started a pause or ended one."""
        if not audio:
            return Signal.NONE

        ms = len(audio) * self._ms_per_byte
        level = _rms(audio)

        # The adaptation rates are per REFERENCE_CHUNK_MS, so they do not depend on how the client chunks the audio
        steps = ms / REFERENCE_CHUNK_MS
        if self._noise_floor is None:
            self._noise_floor = level
        else:
            self._noise_floor = min(level, self._noise_floor * 1.002**steps + 0.2 * steps)
        self._peak = max(level, self._peak * 0.998**steps)

        loud = level > max(2.5 * self._noise_floor + 30, 0.12 * self._peak, 100)
        quiet = level <= max(1.8 * self._noise_floor + 20, 0.05 * self._peak, 60)

        signal = Signal.NONE
        if loud:
            self._loud_ms += ms
        if quiet:
            self._quiet_ms += ms
        else:
            self._quiet_ms = 0.0
            if self._in_pause:
                self._sound_after_pause_ms += ms
                if self._sound_after_pause_ms >= EARLY_RESUME_MS:
                    self._in_pause = False
                    signal = Signal.RESUMED

        if (
            quiet
            and not self._in_pause
            and self._loud_ms >= EARLY_MIN_SPEECH_MS
            and self._quiet_ms >= EARLY_SILENCE_MS
            and self._pauses < EARLY_MAX_REQUESTS
        ):
            self._pauses += 1
            self._in_pause = True
            self._sound_after_pause_ms = 0.0
            signal = Signal.PAUSE

        return signal


def _consume_exception(task: asyncio.Task[Any]) -> None:
    """Mark the error of a finished early request as seen; only requests that are still valid are ever awaited."""
    if not task.cancelled():
        task.exception()


class EarlyTranscription:
    """
    Transcribes one recording early, while it is still being received.

    Args:
        request: Coroutine function that transcribes a WAV recording and returns the backend's response.
        rate: Sample rate of the incoming audio.
        width: Bytes per sample of the incoming audio (must be 2).
        channels: Number of channels of the incoming audio.
    """

    def __init__(
        self, request: Callable[[NamedBytesIO], Awaitable[Any]], *, rate: int, width: int, channels: int
    ) -> None:
        self._request = request
        self._rate = rate
        self._width = width
        self._channels = channels
        self._detector = PauseDetector(rate=rate, width=width, channels=channels)
        self._pcm = bytearray()
        self._task: asyncio.Task[Any] | None = None
        self._started_at = 0.0

    def feed(self, audio: bytes) -> None:
        """Take the next chunk of the recording; start the early request on a pause and drop it if sound resumes."""
        self._pcm += audio
        signal = self._detector.feed(audio)

        if signal is Signal.PAUSE:
            self._start()
        elif signal is Signal.RESUMED:
            _LOGGER.debug(
                "Dropping early transcription: sound resumed %.0f ms after it started",
                (time.monotonic() - self._started_at) * 1000,
            )
            self.cancel()

    async def result(self) -> Any | None:
        """
        Return the early answer if it still covers the whole recording, otherwise None.

        Call this when the recording has ended. A request that is still running is awaited: it started before a new
        request could, so it also finishes before one. None means "send the normal request": nothing was started,
        sound came back after the request started, or the early request failed.
        """
        task = self._task
        if task is None:
            return None

        self._task = None
        lead_ms = (time.monotonic() - self._started_at) * 1000
        try:
            transcription = await task
        except Exception as exc:
            _LOGGER.warning("Early transcription failed (%s); transcribing the recording normally", exc)
            return None

        _LOGGER.info("Using early transcription (its request started %.0f ms before the audio stopped)", lead_ms)
        return transcription

    def cancel(self) -> None:
        """Stop an early request that is still running. Safe to call at any time."""
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def _start(self) -> None:
        pcm = bytes(self._pcm)
        self._started_at = time.monotonic()
        self._task = asyncio.create_task(self._transcribe(pcm), name="early_transcription")
        self._task.add_done_callback(_consume_exception)
        _LOGGER.debug(
            "Starting early transcription after a pause (%.0f ms of audio so far)",
            len(pcm) / (self._rate * self._width * self._channels) * 1000,
        )

    async def _transcribe(self, pcm: bytes) -> Any:
        wav_buffer = NamedBytesIO(name="recording.wav")
        with wave.open(wav_buffer, "wb") as wav_writer:
            wav_writer.setnchannels(self._channels)
            wav_writer.setsampwidth(self._width)
            wav_writer.setframerate(self._rate)
            wav_writer.writeframes(pcm)
        wav_buffer.seek(0)
        return await self._request(wav_buffer)
