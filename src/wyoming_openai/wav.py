import io
import logging
import struct
import wave
from dataclasses import dataclass

from .const import DEFAULT_AUDIO_CHANNELS, DEFAULT_AUDIO_WIDTH

_LOGGER = logging.getLogger(__name__)

TTS_WAV_HEADER_MAX_BYTES = 65536  # Bound header buffering if a backend never yields a complete WAV header
WAV_UNBOUNDED_SIZE = 0xFFFFFFFF  # Streaming WAV data chunk size sentinel
# RIFF sizes that stand in for a length the writer did not know: the largest unsigned and signed 32-bit values.
# Any other RIFF size is a real length, however large
WAV_PLACEHOLDER_RIFF_SIZES = frozenset((WAV_UNBOUNDED_SIZE, 0x7FFFFFFF))


@dataclass(frozen=True)
class TtsAudioFormat:
    """How the audio bytes a voice's transport yields are framed."""

    headerless: bool  # Raw PCM16 mono rather than a WAV file
    rate: int  # Hz; for WAV only a fallback until the header is parsed


def parse_wav_header(wav_data: bytes) -> tuple[int, int, int, int, int | None] | None:
    """
    Parse WAV header to extract the PCM format, data offset, and data size.
    Returns (sample_rate, channels, sample_width, data_offset, data_size) or None if parsing fails.
    The data size is None when the WAV data chunk uses the unbounded-size sentinel.
    """
    try:
        # Create a BytesIO object from the data
        wav_io = io.BytesIO(wav_data)

        # Some streaming writers leave a RIFF length too small even to contain WAVE.
        # Retry that narrow case once; the framer also treats the original size as unknown.
        try:
            wav_file = wave.open(wav_io, "rb")
        except (wave.Error, EOFError):
            if len(wav_data) < 8 or struct.unpack_from("<I", wav_data, 4)[0] >= 4:
                raise
            wav_io = io.BytesIO(wav_data[:4] + struct.pack("<I", WAV_UNBOUNDED_SIZE) + wav_data[8:])
            wav_file = wave.open(wav_io, "rb")
        with wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()

            # Get the current position which should be at the start of audio data
            data_offset = wav_io.tell()
            declared_data_size = struct.unpack("<I", wav_data[data_offset - 4 : data_offset])[0]
            data_size = (
                None if declared_data_size == WAV_UNBOUNDED_SIZE else wav_file.getnframes() * channels * sample_width
            )

            return sample_rate, channels, sample_width, data_offset, data_size
    except Exception as e:
        _LOGGER.debug("Failed to parse WAV header: %s", e)
        return None


class WavFramer:
    """Turns the bytes of a TTS response into PCM, for the streamed and the buffered path alike."""

    def __init__(self, audio_format: TtsAudioFormat) -> None:
        self.rate = audio_format.rate
        self.width = DEFAULT_AUDIO_WIDTH
        self.channels = DEFAULT_AUDIO_CHANNELS
        self.awaiting_header = not audio_format.headerless
        self.header_missing = False  # No WAV header could be parsed, so the bytes were released as raw PCM
        self._pending = b""  # Bytes held until the header is understood
        self._remaining: int | None = None  # Declared PCM bytes not received yet; None when unbounded

    @property
    def missing_bytes(self) -> int:
        """Declared PCM bytes that have not arrived."""
        return self._remaining or 0

    def feed(self, chunk: bytes) -> bytes:
        """Take the next response bytes and return the PCM to play, which is empty while the header is incomplete."""
        if not self.awaiting_header:
            return self._bound(chunk)

        self._pending += chunk
        wav_params = parse_wav_header(self._pending)
        if wav_params:
            self.rate, self.channels, self.width, data_offset, data_size = wav_params
            riff_size = struct.unpack_from("<I", self._pending, 4)[0]
            available_audio = self._pending[data_offset:]
            self._pending = b""
            self.awaiting_header = False
            frame_size = self.width * self.channels
            if (
                data_size
                and riff_size in WAV_PLACEHOLDER_RIFF_SIZES
                and data_offset - 8 + data_size > riff_size - frame_size
            ):
                # A data length that fills a placeholder container was derived from it, so it is as
                # unknown as the RIFF size. Any shorter length is a real one and still bounds the audio
                self._remaining = None
            elif data_size == 0 and riff_size not in WAV_PLACEHOLDER_RIFF_SIZES and 8 + riff_size > data_offset:
                # The RIFF container goes on after an empty data chunk: an empty file, followed by metadata
                # of whatever length the container declares
                self._remaining = 0
            else:
                # A zero size is also what servers write when streaming audio of unknown length: the RIFF
                # size is then a placeholder or covers nothing past the header
                self._remaining = data_size or None
            _LOGGER.debug(
                "Detected audio format: %d Hz, %d channels, %d bytes/sample, header offset: %d, PCM size: %s",
                self.rate,
                self.channels,
                self.width,
                data_offset,
                "unknown" if self._remaining is None else f"{self._remaining} bytes",
            )
            return self._bound(available_audio)

        if len(self._pending) <= TTS_WAV_HEADER_MAX_BYTES:
            return b""

        # Callers report `header_missing`: only they know if this was a stream or a whole buffered response
        return self.finish()

    def finish(self) -> bytes:
        """Return the bytes still held when the response ends without a WAV header, to be played as raw PCM."""
        audio, self._pending = self._pending, b""
        if self.awaiting_header and audio:
            self.header_missing = True
        self.awaiting_header = False
        return audio

    def _bound(self, audio: bytes) -> bytes:
        """Keep only the PCM the header covers."""
        if self._remaining is None:
            return audio
        audio = audio[: self._remaining]
        self._remaining -= len(audio)
        return audio
