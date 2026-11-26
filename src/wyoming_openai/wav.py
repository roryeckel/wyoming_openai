import array
import logging
import struct
import sys
import uuid
from dataclasses import dataclass
from typing import NamedTuple

from .const import DEFAULT_AUDIO_CHANNELS, DEFAULT_AUDIO_WIDTH
from .utilities import take_whole_frames

_LOGGER = logging.getLogger(__name__)

TTS_WAV_HEADER_MAX_BYTES = 65536  # Bound header buffering if a backend never yields a complete WAV header
WAV_UNBOUNDED_SIZE = 0xFFFFFFFF  # Streaming WAV data chunk size sentinel
# RIFF sizes that stand in for a length the writer did not know: the largest unsigned and signed 32-bit values.
# Any other RIFF size is a real length, however large
WAV_PLACEHOLDER_RIFF_SIZES = frozenset((WAV_UNBOUNDED_SIZE, 0x7FFFFFFF))

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE
# The GUID of an extensible subformat is its format tag followed by these bytes
_KSDATAFORMAT_SUFFIX = bytes.fromhex("000000001000800000aa00389b71")
# How to get playable audio from a backend whose WAV output is in a format the proxy does not play
_UNSUPPORTED_FORMAT_HINT = (
    "; if the backend can return raw 24 kHz 16-bit mono PCM instead, "
    """set TTS_EXTRA_BODY='{"response_format": "pcm"}'"""
)
# Formats seen from TTS backends, named so an error says what was received
_WAVE_FORMAT_NAMES = {
    0x0002: "Microsoft ADPCM",
    0x0006: "A-law",
    0x0007: "mu-law",
    0x0011: "IMA ADPCM",
    0x0031: "GSM 6.10",
    0x0050: "MPEG",
    0x0055: "MPEG Layer 3",
}
# How a response starts when it is audio of a kind the proxy does not play, named so it fails saying what it is
# instead of playing as noise. Raw PCM starts with one of these only by a chance too small to matter. MP3 without
# an ID3 tag is not recognized: its frame sync, eleven set bits, is also how near-silent 16-bit PCM starts
_FOREIGN_AUDIO_SIGNATURES = (
    (b"RF64", "RF64 WAV"),
    (b"BW64", "BW64 WAV"),
    (b"RIFX", "big-endian RIFX WAV"),
    (b"fLaC", "FLAC"),
    (b"OggS", "Ogg (Opus or Vorbis)"),
    # An ID3v2 tag, followed by its major version
    (b"ID3\x02", "MP3"),
    (b"ID3\x03", "MP3"),
    (b"ID3\x04", "MP3"),
)
_SIGNATURE_BYTES = 4  # Enough of a response to tell a WAV file from raw PCM or another kind of audio


@dataclass(frozen=True)
class TtsAudioFormat:
    """How the audio bytes a voice's transport yields are framed."""

    headerless: bool  # Raw PCM16 mono rather than a WAV file
    rate: int  # Hz; for WAV only a fallback until the header is parsed
    # Whether a headerless response may be a WAV file anyway, from a backend that ignores response_format, and is
    # then read by its header. True for /v1/audio/speech; never for Realtime, which only carries raw PCM
    may_be_wav: bool = False


class WavFormatError(ValueError):
    """A response that starts as a WAV file but whose header cannot be read or whose audio cannot be played."""


class WavHeader(NamedTuple):
    rate: int
    channels: int
    width: int  # Bytes per sample in the file, before any conversion
    data_offset: int
    data_size: int | None  # Whole frames the data chunk declares; None when it uses the unbounded-size sentinel
    sample_format: int  # WAVE_FORMAT_PCM or WAVE_FORMAT_IEEE_FLOAT, after unwrapping EXTENSIBLE


def _reject_foreign_audio(start: bytes) -> None:
    """Raise for a response whose first bytes show it is a kind of audio the proxy does not play."""
    for signature, name in _FOREIGN_AUDIO_SIGNATURES:
        if start.startswith(signature):
            raise WavFormatError(f"TTS response is {name} audio, which the proxy does not play (only WAV and raw PCM)")


def _describe_format(tag: int) -> str:
    name = _WAVE_FORMAT_NAMES.get(tag)
    return f"0x{tag:04X} ({name})" if name else f"0x{tag:04X}"


def _parse_fmt_chunk(fmt: bytes) -> tuple[int, int, int, int]:
    """Read (sample format, channels, rate, bytes per sample) from a fmt chunk, rejecting what cannot be played."""
    if len(fmt) < 16:
        raise WavFormatError(f"WAV fmt chunk is {len(fmt)} bytes, too short to describe the audio (needs 16)")
    tag, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", fmt)
    if tag == WAVE_FORMAT_EXTENSIBLE:
        if len(fmt) < 40:
            raise WavFormatError(f"WAV fmt chunk is {len(fmt)} bytes, too short for WAVE_FORMAT_EXTENSIBLE (needs 40)")
        subformat = fmt[24:40]
        if subformat[2:] != _KSDATAFORMAT_SUFFIX:
            raise WavFormatError(
                f"WAV extensible subformat {uuid.UUID(bytes_le=subformat)} is not supported{_UNSUPPORTED_FORMAT_HINT}"
            )
        tag = struct.unpack_from("<H", subformat)[0]
    if tag not in (WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT):
        raise WavFormatError(
            f"WAV audio format {_describe_format(tag)} is not supported; "
            f"the proxy plays integer PCM and 32- or 64-bit IEEE float WAV{_UNSUPPORTED_FORMAT_HINT}"
        )
    if not channels:
        raise WavFormatError("WAV header declares 0 channels")
    if not rate:
        raise WavFormatError("WAV header declares a sample rate of 0 Hz")
    if tag == WAVE_FORMAT_IEEE_FLOAT and bits not in (32, 64):
        raise WavFormatError(
            f"WAV IEEE float audio with {bits}-bit samples is not supported (only 32 or 64){_UNSUPPORTED_FORMAT_HINT}"
        )
    width = (bits + 7) // 8
    if not width:
        raise WavFormatError("WAV header declares 0 bits per sample")
    return tag, channels, rate, width


def parse_wav_header(wav_data: bytes) -> WavHeader | None:
    """
    Parse a WAV header to find the audio format, data offset, and data size.

    Returns None while the bytes could still be the start of a WAV header that has not fully arrived, and for
    bytes that are not a RIFF file at all, which callers may play as raw PCM.

    Raises:
        WavFormatError: The bytes are a RIFF file, but not a WAV file whose audio can be played.
    """
    if len(wav_data) < 12 or wav_data[:4] != b"RIFF":
        return None
    if wav_data[8:12] != b"WAVE":
        raise WavFormatError(f"RIFF file of form {wav_data[8:12]!r} is not WAVE audio")

    fmt: tuple[int, int, int, int] | None = None
    position = 12
    while position + 8 <= len(wav_data):
        chunk_id = wav_data[position : position + 4]
        chunk_size = struct.unpack_from("<I", wav_data, position + 4)[0]
        body = position + 8
        if chunk_id == b"data":
            if fmt is None:
                raise WavFormatError("WAV data chunk comes before any fmt chunk, so its format is unknown")
            tag, channels, rate, width = fmt
            frame_size = width * channels
            data_size = None if chunk_size == WAV_UNBOUNDED_SIZE else chunk_size - chunk_size % frame_size
            return WavHeader(rate, channels, width, body, data_size, tag)
        # RIFF pads chunks of odd length to an even one
        end = body + chunk_size + chunk_size % 2
        if end > len(wav_data):
            return None
        if chunk_id == b"fmt ":
            fmt = _parse_fmt_chunk(wav_data[body : body + chunk_size])
        position = end
    return None


def _float_to_pcm16(audio: bytes, width: int) -> bytes:
    """Convert little-endian IEEE float samples, nominally within [-1, 1], to 16-bit PCM, clipping beyond it."""
    samples = array.array("f" if width == 4 else "d")
    samples.frombytes(audio)
    if sys.byteorder == "big":
        samples.byteswap()
    # NaN fails every comparison and becomes silence rather than a full-scale click. One list comprehension over an
    # array runs over twice as fast as unpacking with `struct`, which matters on a Raspberry Pi
    pcm = array.array(
        "h", [0 if x != x else 32767 if x >= 1.0 else -32767 if x <= -1.0 else round(x * 32767) for x in samples]
    )
    if sys.byteorder == "big":
        pcm.byteswap()
    return pcm.tobytes()


class WavFramer:
    """Turns the bytes of a TTS response into PCM, for the streamed and the buffered path alike."""

    def __init__(self, audio_format: TtsAudioFormat) -> None:
        self.rate = audio_format.rate
        self.width = DEFAULT_AUDIO_WIDTH  # Of the PCM handed out, which float audio is converted to
        self.channels = DEFAULT_AUDIO_CHANNELS
        # Until the first bytes show whether the response is a WAV file, and then until its header is understood
        self.awaiting_header = not audio_format.headerless or audio_format.may_be_wav
        self.header_missing = False  # A WAV file was requested but not returned, so the bytes were played as raw PCM
        self.unrequested_wav = False  # Raw PCM was requested but a WAV file returned, which is read by its header
        self._expects_wav = not audio_format.headerless
        self._pending = b""  # Bytes held until the header is understood
        self._remaining: int | None = None  # Declared PCM bytes not received yet; None when unbounded
        self._partial = b""  # The start of a frame whose rest has not arrived, since chunks hold whole frames
        self._source_width = DEFAULT_AUDIO_WIDTH  # Bytes per sample in the response
        self._float = False  # The response holds IEEE float samples, converted to 16-bit PCM

    @property
    def missing_bytes(self) -> int:
        """Declared PCM bytes that have not arrived."""
        return self._remaining or 0

    def feed(self, chunk: bytes) -> bytes:
        """
        Take the next response bytes and return the PCM to play, which is empty while the header is incomplete.

        Raises:
            WavFormatError: The response is a WAV file whose header cannot be read or whose audio cannot be played,
                or audio of another kind the proxy recognizes, such as MP3 or FLAC.
        """
        if not self.awaiting_header:
            return self._convert(self._whole_frames(self._bound(chunk)))

        self._pending += chunk
        if len(self._pending) < _SIGNATURE_BYTES:
            return b""
        if not self._pending.startswith(b"RIFF"):
            _reject_foreign_audio(self._pending)
            # Raw PCM, which is played as it arrives rather than after a wait for a header that will not come.
            # Callers report `header_missing`: only they know if this was a stream or a whole buffered response
            return self._release_pending()

        header = parse_wav_header(self._pending)
        if header:
            self.unrequested_wav = not self._expects_wav
            self.rate, self.channels, self._source_width = header.rate, header.channels, header.width
            data_offset, data_size = header.data_offset, header.data_size
            self._float = header.sample_format == WAVE_FORMAT_IEEE_FLOAT
            self.width = DEFAULT_AUDIO_WIDTH if self._float else self._source_width
            riff_size = struct.unpack_from("<I", self._pending, 4)[0]
            available_audio = self._pending[data_offset:]
            self._pending = b""
            self.awaiting_header = False
            frame_size = self._source_width * self.channels
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
                "Detected audio format: %s%d Hz, %d channels, %d bytes/sample, header offset: %d, audio size: %s",
                "IEEE float converted to 16-bit PCM, " if self._float else "",
                self.rate,
                self.channels,
                self._source_width,
                data_offset,
                "unknown" if self._remaining is None else f"{self._remaining} bytes",
            )
            return self._convert(self._whole_frames(self._bound(available_audio)))

        if len(self._pending) <= TTS_WAV_HEADER_MAX_BYTES:
            return b""
        raise WavFormatError(
            f"TTS response starts as a WAV file but has no data chunk in its first {TTS_WAV_HEADER_MAX_BYTES} bytes"
        )

    def finish(self) -> bytes:
        """
        Return the bytes still held when the response ends too short to show it is a WAV file, to be played as raw
        PCM. A frame the response ends in the middle of is dropped, since its samples are incomplete.

        Raises:
            WavFormatError: The response ended inside a WAV header.
        """
        if self.awaiting_header and self._pending.startswith(b"RIFF"):
            raise WavFormatError(f"TTS response ended after {len(self._pending)} bytes, inside its WAV header")
        audio = self._release_pending()
        if self._partial:
            _LOGGER.debug("Dropping %d bytes that end the audio in the middle of a frame", len(self._partial))
            self._partial = b""
        return audio

    def _release_pending(self) -> bytes:
        """Give up on a WAV header and release the bytes held for it as raw PCM."""
        audio, self._pending = self._pending, b""
        if self.awaiting_header and audio and self._expects_wav:
            self.header_missing = True
        self.awaiting_header = False
        return self._whole_frames(audio)

    def _whole_frames(self, audio: bytes) -> bytes:
        """Return the whole frames of the audio so far, carrying a partial frame to the next bytes."""
        audio, self._partial = take_whole_frames(
            partial=self._partial, audio=audio, frame_size=self._source_width * self.channels
        )
        return audio

    def _convert(self, audio: bytes) -> bytes:
        """Turn whole frames of the response into the PCM handed out."""
        return _float_to_pcm16(audio, self._source_width) if self._float else audio

    def _bound(self, audio: bytes) -> bytes:
        """Keep only the PCM the header covers."""
        if self._remaining is None:
            return audio
        audio = audio[: self._remaining]
        self._remaining -= len(audio)
        return audio
