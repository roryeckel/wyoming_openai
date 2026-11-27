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

WAV_UNBOUNDED_SIZE = 0xFFFFFFFF  # Streaming WAV data chunk size sentinel
# RIFF sizes that stand in for a length the writer did not know: the largest unsigned and signed 32-bit values.
# Any other RIFF size is a real length, however large
WAV_PLACEHOLDER_RIFF_SIZES = frozenset((WAV_UNBOUNDED_SIZE, 0x7FFFFFFF))

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE
# The GUID of an extensible subformat is its format tag followed by these bytes
_KSDATAFORMAT_SUFFIX = bytes.fromhex("000000001000800000aa00389b71")
# How to get playable audio from a backend whose WAV output is in a format the proxy does not play. Only given when
# WAV was requested: a backend that answers a request for raw PCM with WAV anyway has already ignored that setting
_UNSUPPORTED_FORMAT_HINT = (
    "; if the backend can return raw 24 kHz 16-bit mono PCM instead, "
    'add "response_format": "pcm" to TTS_EXTRA_BODY or --tts-extra-body'
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
# Where a response that is audio of a kind the proxy does not play has which bytes, named so it fails saying what it
# is instead of playing as noise. Raw PCM has one of these only by a chance too small to matter. MP3 without an ID3
# tag and ADTS AAC have no signature and are recognized by their first frame header instead
_FOREIGN_AUDIO_SIGNATURES = (
    (0, b"RF64", "RF64 (64-bit WAV)"),
    (0, b"BW64", "BW64 (64-bit WAV)"),
    (0, b"RIFX", "big-endian RIFX WAV"),
    (0, b"FORM", "AIFF"),
    (0, b"caff", "CAF"),
    (0, b"fLaC", "FLAC"),
    (0, b"OggS", "Ogg (Opus or Vorbis)"),
    (0, b"\x1a\x45\xdf\xa3", "WebM or Matroska"),
    (4, b"ftyp", "MP4 or M4A"),
    # An ID3v2 tag, followed by its major version
    (0, b"ID3\x02", "MP3"),
    (0, b"ID3\x03", "MP3"),
    (0, b"ID3\x04", "MP3"),
)
_SIGNATURE_BYTES = 8  # Enough of a response to tell a WAV file from raw PCM or another kind of audio


@dataclass(frozen=True)
class TtsAudioFormat:
    """How the audio bytes a voice's transport yields are framed."""

    headerless: bool  # Raw PCM16 mono was requested, or is all the transport carries, rather than a WAV file
    rate: int  # Hz; for WAV only a fallback until the header is parsed
    # Whether a headerless response may be a WAV file anyway, from a backend that ignores response_format, and is
    # then read by its header. True for /v1/audio/speech; never for Realtime, which only carries raw PCM
    may_be_wav: bool = False


class WavFormatError(ValueError):
    """
    A TTS response the proxy recognizes but cannot play: a WAV file whose header cannot be read or whose audio is in
    a format it does not play, or audio of another kind, such as MP3 or FLAC.
    """


class UnsupportedWavFormatError(WavFormatError):
    """A well-formed WAV file whose audio is in a format the proxy does not play, such as a compressed codec."""


class WavHeader(NamedTuple):
    rate: int
    channels: int
    width: int  # Bytes per sample in the file, before any conversion
    data_offset: int
    data_size: int | None  # Bytes the data chunk declares, rounded down to whole frames; None when unbounded
    sample_format: int  # WAVE_FORMAT_PCM or WAVE_FORMAT_IEEE_FLOAT, after unwrapping EXTENSIBLE


def _frame_header_audio(start: bytes) -> str | None:
    """
    Name the audio a response is when it starts with the header of an MP3 or ADTS AAC frame.

    Both start with a run of set bits, as near-silent 16-bit PCM does, so the fields after it are checked too.
    16-bit PCM then matches only when its first sample is one of ten values from -1025 to -7425, -30 dBFS or
    louder, which speech that starts from silence does not have.
    """
    if len(start) < 3 or start[0] != 0xFF:
        return None
    second, third = start[1], start[2]
    # MPEG 1, 2 or 2.5 Layer III: an 11-bit sync, a version that is not reserved, layer bits 01, and a bitrate and
    # sample rate that are neither free-format nor reserved
    if second & 0xE0 == 0xE0 and (second >> 3) & 3 != 1 and (second >> 1) & 3 == 1:
        if (third >> 4) not in (0, 15) and (third >> 2) & 3 != 3:
            return "MP3"
    # ADTS AAC: a 12-bit sync, layer bits 00, and a sampling frequency index that is defined
    if second & 0xF6 == 0xF0 and (third >> 2) & 0xF < 13:
        return "AAC (ADTS)"
    return None


def _reject_foreign_audio(start: bytes) -> None:
    """Raise for a response whose first bytes show it is a kind of audio the proxy does not play."""
    names = [name for offset, signature, name in _FOREIGN_AUDIO_SIGNATURES if start.startswith(signature, offset)]
    name = names[0] if names else _frame_header_audio(start)
    if name:
        raise WavFormatError(f"TTS response is {name} audio, which the proxy does not play (only RIFF WAV and raw PCM)")


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
            raise UnsupportedWavFormatError(
                f"WAV extensible subformat {uuid.UUID(bytes_le=subformat)} is not supported"
            )
        tag = struct.unpack_from("<H", subformat)[0]
    if tag not in (WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT):
        raise UnsupportedWavFormatError(
            f"WAV audio format {_describe_format(tag)} is not supported; "
            "the proxy plays integer PCM and 32- or 64-bit IEEE float WAV"
        )
    if not channels:
        raise WavFormatError("WAV header declares 0 channels")
    if not rate:
        raise WavFormatError("WAV header declares a sample rate of 0 Hz")
    if tag == WAVE_FORMAT_IEEE_FLOAT and bits not in (32, 64):
        raise UnsupportedWavFormatError(
            f"WAV IEEE float audio with {bits}-bit samples is not supported (only 32 or 64)"
        )
    width = (bits + 7) // 8
    if not width:
        raise WavFormatError("WAV header declares 0 bits per sample")
    if tag == WAVE_FORMAT_PCM and width > 4:
        # Wyoming clients play integer PCM of at most 4 bytes per sample
        raise UnsupportedWavFormatError(f"WAV integer PCM with {bits}-bit samples is not supported (at most 32)")
    return tag, channels, rate, width


class _WavHeaderReader:
    """
    Reads the header of a WAV file as its bytes arrive, chunk by chunk, up to the start of the data chunk.

    It holds only what it still has to read: a chunk header, or the part of the fmt chunk that describes the audio.
    Other chunks before the audio, such as metadata or padding of any length, are dropped as they arrive.
    """

    def __init__(self) -> None:
        self.held = b""  # Bytes not read yet; once the header is read, the start of the audio
        self.offset = 0  # Position in the file of the first held byte
        self.riff_size: int | None = None  # Read from the RIFF header once it has arrived and named WAVE
        self._skip = 0  # Bytes of the current chunk still to drop
        self._fmt: tuple[int, int, int, int] | None = None

    @property
    def received(self) -> int:
        """Bytes of the file received so far."""
        return self.offset + len(self.held)

    def feed(self, data: bytes) -> WavHeader | None:
        """
        Take the next bytes of a file that starts with `RIFF`, and return its header once the data chunk starts.

        Raises:
            WavFormatError: The file is not a WAV file whose audio can be played.
        """
        self.held += data
        if self.riff_size is None:
            if len(self.held) < 12:
                return None
            if self.held[8:12] != b"WAVE":
                raise WavFormatError(f"RIFF file of form {self.held[8:12]!r} is not WAVE audio")
            self.riff_size = struct.unpack_from("<I", self.held, 4)[0]
            self._consume(12)

        while True:
            if self._skip:
                dropped = min(self._skip, len(self.held))
                self._consume(dropped)
                self._skip -= dropped
                if self._skip:
                    return None
            if len(self.held) < 8:
                return None
            chunk_id = self.held[:4]
            chunk_size = struct.unpack_from("<I", self.held, 4)[0]
            if chunk_id == b"data":
                if self._fmt is None:
                    raise WavFormatError("WAV data chunk comes before any fmt chunk, so its format is unknown")
                tag, channels, rate, width = self._fmt
                frame_size = width * channels
                data_size = None if chunk_size == WAV_UNBOUNDED_SIZE else chunk_size - chunk_size % frame_size
                self._consume(8)
                return WavHeader(rate, channels, width, self.offset, data_size, tag)
            # RIFF pads chunks of odd length to an even one
            padded_size = chunk_size + chunk_size % 2
            read_size = 0
            if chunk_id == b"fmt ":
                # Only the first 40 bytes describe the audio, however long the chunk says it is
                read_size = min(chunk_size, 40)
                if len(self.held) < 8 + read_size:
                    return None
                self._fmt = _parse_fmt_chunk(self.held[8 : 8 + read_size])
            else:
                _LOGGER.debug("Skipping a %d-byte %r chunk before the WAV audio", chunk_size, chunk_id)
            self._consume(8 + read_size)
            self._skip = padded_size - read_size

    def _consume(self, size: int) -> None:
        self.held = self.held[size:]
        self.offset += size


def parse_wav_header(wav_data: bytes) -> WavHeader | None:
    """
    Parse the WAV header at the start of a buffer to find the audio format, data offset, and data size.

    Returns None for bytes that are not a RIFF file, and while a WAV header has not fully arrived. Unlike
    `WavFramer`, it does not look for other kinds of audio such as FLAC or MP3, so None does not mean the bytes
    are raw PCM.

    Raises:
        WavFormatError: The bytes are a RIFF file, but not a WAV file whose audio can be played.
    """
    if not wav_data.startswith(b"RIFF"):
        return None
    return _WavHeaderReader().feed(wav_data)


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
        self._pending = b""  # The first bytes, held until they show whether the response is a WAV file
        self._header: _WavHeaderReader | None = None  # Reading the header of a response that is a WAV file
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

        if self._header is None:
            self._pending += chunk
            if not self._pending.startswith(b"RIFF"):
                if len(self._pending) < _SIGNATURE_BYTES:
                    return b""
                try:
                    _reject_foreign_audio(self._pending)
                except WavFormatError as e:
                    raise WavFormatError(f"{e}{self._error_note(pcm_may_help=True)}") from e
                # Raw PCM, which is played as it arrives rather than after a wait for a header that will not come.
                # Callers report `header_missing`, once for all the responses of a synthesis
                return self._release_pending()
            self._header = _WavHeaderReader()
            chunk, self._pending = self._pending, b""

        try:
            header = self._header.feed(chunk)
        except WavFormatError as e:
            if note := self._error_note(pcm_may_help=isinstance(e, UnsupportedWavFormatError)):
                raise type(e)(f"{e}{note}") from e
            raise
        if header is None:
            return b""

        riff_size = self._header.riff_size or 0
        available_audio = self._header.held
        self._header = None
        self.awaiting_header = False
        self.unrequested_wav = not self._expects_wav
        self.rate, self.channels, self._source_width = header.rate, header.channels, header.width
        data_offset, data_size = header.data_offset, header.data_size
        self._float = header.sample_format == WAVE_FORMAT_IEEE_FLOAT
        self.width = DEFAULT_AUDIO_WIDTH if self._float else self._source_width
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

    def finish(self) -> bytes:
        """
        Return the bytes still held when the response ends too short to show it is a WAV file, to be played as raw
        PCM. A frame the response ends in the middle of is dropped, since its samples are incomplete.

        Raises:
            WavFormatError: The response is a WAV file that ended before its header reached the audio: it was cut off,
                or a chunk size in its header is wrong, so the chunks after it were misread.
        """
        if self._header is not None:
            raise WavFormatError(
                f"TTS response ended after {self._header.received} bytes, before its WAV header reached the audio; "
                f"the header is truncated or malformed{self._error_note()}"
            )
        audio = self._release_pending()
        if self._partial:
            _LOGGER.debug("Dropping %d bytes that end the audio in the middle of a frame", len(self._partial))
            self._partial = b""
        return audio

    def _error_note(self, *, pcm_may_help: bool = False) -> str:
        """
        What to add to an error about a response, given what was requested; the error already says what it is.

        `pcm_may_help` is for audio a backend may be able to send as raw PCM instead, such as a WAV codec or MP3.
        """
        if not self._expects_wav:
            # The backend ignores response_format, so asking for raw PCM is not a way around the error
            return "; raw PCM was requested, so the backend ignores response_format"
        return _UNSUPPORTED_FORMAT_HINT if pcm_may_help else ""

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
