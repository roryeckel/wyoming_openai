"""Builders for WAV files in formats the stdlib `wave` module cannot write, such as IEEE float."""

import struct

# Written out here rather than imported from wyoming_openai.wav, so a mistyped constant there fails the tests
WAVE_FORMAT_EXTENSIBLE = 0xFFFE
KSDATAFORMAT_SUFFIX = bytes.fromhex("000000001000800000aa00389b71")


def fmt_chunk_body(tag: int, channels: int = 1, rate: int = 24000, bits: int = 32, extra: bytes = b"") -> bytes:
    """The body of a fmt chunk, with the byte rate and block alignment derived from the rest."""
    block_align = channels * bits // 8
    return struct.pack("<HHIIHH", tag, channels, rate, rate * block_align, block_align, bits) + extra


def extensible_fmt_chunk_body(subformat_tag: int, bits: int = 32, channels: int = 1, rate: int = 24000) -> bytes:
    """The body of a WAVE_FORMAT_EXTENSIBLE fmt chunk whose subformat GUID carries the given format tag."""
    guid = struct.pack("<H", subformat_tag) + KSDATAFORMAT_SUFFIX
    extra = struct.pack("<HHI", 22, bits, 0) + guid
    return fmt_chunk_body(WAVE_FORMAT_EXTENSIBLE, channels=channels, rate=rate, bits=bits, extra=extra)


def riff_chunk(chunk_id: bytes, payload: bytes) -> bytes:
    """Build a word-aligned RIFF chunk."""
    padding = b"\x00" if len(payload) % 2 else b""
    return chunk_id + struct.pack("<I", len(payload)) + payload + padding


def wav_file(
    fmt: bytes,
    data: bytes = b"",
    chunks: bytes = b"",
    *,
    before_fmt: bytes = b"",
    riff_size: int | None = None,
    data_size: int | None = None,
) -> bytes:
    """
    A WAV file of a fmt chunk body and audio data, with any other chunks before the fmt chunk or between the two.
    The RIFF and data sizes are the real ones unless given, as streaming writers leave placeholders there.
    """
    data_header = b"data" + struct.pack("<I", len(data) if data_size is None else data_size)
    body = b"WAVE" + before_fmt + riff_chunk(b"fmt ", fmt) + chunks + data_header + data
    return b"RIFF" + struct.pack("<I", len(body) if riff_size is None else riff_size) + body
