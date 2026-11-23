import io
import struct
import wave

import pytest
from wav_helpers import extensible_fmt_chunk_body, fmt_chunk_body, wav_file

from wyoming_openai.utilities import take_whole_frames
from wyoming_openai.wav import (
    TTS_WAV_HEADER_MAX_BYTES,
    WAVE_FORMAT_IEEE_FLOAT,
    TtsAudioFormat,
    WavFormatError,
    WavFramer,
    WavHeader,
    parse_wav_header,
)

PCM = b"\x00\x01" * 240


def _wav(pcm: bytes = PCM, rate: int = 16000, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


def test_parse_wav_header_reads_format_offset_and_size():
    assert parse_wav_header(_wav(rate=22050, channels=2)) == WavHeader(22050, 2, 2, 44, len(PCM))


def test_parse_wav_header_reports_an_unbounded_data_size_as_unknown():
    wav = _wav()
    unbounded = wav[:40] + struct.pack("<I", 0xFFFFFFFF) + wav[44:]
    assert parse_wav_header(unbounded) == WavHeader(16000, 1, 2, 44, None)


@pytest.mark.parametrize("data", [b"", b"RIFF", _wav()[:43], b"not a wav file at all" * 4])
def test_parse_wav_header_returns_none_without_a_complete_header(data):
    assert parse_wav_header(data) is None


@pytest.mark.parametrize("chunk_size", [1, 7, 44, 4096])
def test_framer_strips_the_header_however_the_response_is_chunked(chunk_size):
    wav = _wav()
    framer = WavFramer(TtsAudioFormat(headerless=False, rate=24000))
    pcm = b"".join(framer.feed(wav[i : i + chunk_size]) for i in range(0, len(wav), chunk_size))

    assert pcm + framer.finish() == PCM
    assert (framer.rate, framer.width, framer.channels) == (16000, 2, 1)
    assert framer.missing_bytes == 0
    assert not framer.header_missing


def _wav_with_extended_fmt(pcm: bytes) -> bytes:
    """A stereo PCM WAV whose fmt chunk has the 18-byte form, which leaves the audio at an odd offset of 46 bytes."""
    return wav_file(fmt_chunk_body(1, channels=2, rate=22050, bits=16, extra=b"\x00\x00"), pcm)


@pytest.mark.parametrize("chunk_size", [1, 7, 2048])
def test_framer_hands_out_whole_frames_after_a_header_of_odd_length(chunk_size):
    """Stereo 16-bit frames are four bytes, so a 46-byte header would split them at every read."""
    pcm = bytes(range(256)) * 32
    wav = _wav_with_extended_fmt(pcm)
    framer = WavFramer(TtsAudioFormat(headerless=False, rate=24000))

    chunks = [framer.feed(wav[i : i + chunk_size]) for i in range(0, len(wav), chunk_size)]

    assert all(len(chunk) % 4 == 0 for chunk in chunks)
    assert b"".join(chunks) + framer.finish() == pcm
    assert (framer.rate, framer.width, framer.channels) == (22050, 2, 2)


def test_framer_drops_a_frame_the_response_ends_in():
    """Headerless PCM16 read in odd pieces keeps whole samples, and a last half sample is not played."""
    framer = WavFramer(TtsAudioFormat(headerless=True, rate=24000))

    assert framer.feed(PCM[:3]) == PCM[:2]
    assert framer.feed(PCM[3:]) == PCM[2:]
    assert framer.feed(b"\x7f") == b""
    assert framer.finish() == b""


def test_framer_passes_headerless_audio_through_at_the_declared_rate():
    framer = WavFramer(TtsAudioFormat(headerless=True, rate=24000))

    assert not framer.awaiting_header
    assert framer.feed(PCM) == PCM
    assert framer.finish() == b""
    assert framer.rate == 24000


def test_framer_reports_declared_audio_that_did_not_arrive():
    framer = WavFramer(TtsAudioFormat(headerless=False, rate=24000))

    assert framer.feed(_wav()[:-100]) == PCM[:-100]
    assert framer.missing_bytes == 100


def test_framer_releases_bytes_as_raw_pcm_when_no_header_shows_up():
    framer = WavFramer(TtsAudioFormat(headerless=False, rate=24000))
    raw = b"\x00\x01" * (TTS_WAV_HEADER_MAX_BYTES // 2)

    assert framer.feed(raw) == b""
    assert framer.feed(b"\x00\x01") == raw + b"\x00\x01"
    assert framer.header_missing
    assert framer.rate == 24000
    # Later bytes pass straight through
    assert framer.feed(PCM) == PCM


def test_framer_finish_hands_back_a_short_response_that_is_not_a_wav_file():
    framer = WavFramer(TtsAudioFormat(headerless=False, rate=24000))

    assert framer.feed(PCM) == b""
    assert framer.finish() == PCM
    assert framer.header_missing


def test_framer_rejects_a_response_that_ended_inside_the_wav_header():
    framer = WavFramer(TtsAudioFormat(headerless=False, rate=24000))

    assert framer.feed(_wav()[:30]) == b""
    with pytest.raises(WavFormatError, match="ended after 30 bytes, inside its WAV header"):
        framer.finish()


FLOATS = [0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -3.0, float("nan")]
FLOATS_AS_PCM16 = [0, 16384, -16384, 32767, -32767, 32767, -32767, 0]


@pytest.mark.parametrize(
    ("fmt", "code"),
    [
        (fmt_chunk_body(WAVE_FORMAT_IEEE_FLOAT), "<f"),
        (fmt_chunk_body(WAVE_FORMAT_IEEE_FLOAT, bits=64), "<d"),
        (extensible_fmt_chunk_body(3), "<f"),
    ],
    ids=["float32", "float64", "extensible-float32"],
)
@pytest.mark.parametrize("chunk_size", [1, 5, 4096])
def test_framer_converts_float_wav_to_pcm16(fmt, code, chunk_size):
    data = b"".join(struct.pack(code, x) for x in FLOATS)
    wav = wav_file(fmt, data)
    framer = WavFramer(TtsAudioFormat(headerless=False, rate=16000))

    chunks = [framer.feed(wav[i : i + chunk_size]) for i in range(0, len(wav), chunk_size)]
    pcm = b"".join(chunks) + framer.finish()

    assert all(len(chunk) % 2 == 0 for chunk in chunks)
    assert list(struct.unpack(f"<{len(FLOATS)}h", pcm)) == FLOATS_AS_PCM16
    assert (framer.rate, framer.width, framer.channels) == (24000, 2, 1)
    assert framer.missing_bytes == 0


def test_framer_bounds_float_audio_by_the_declared_data_size():
    data = struct.pack("<2f", 0.5, -0.5)
    wav = wav_file(fmt_chunk_body(WAVE_FORMAT_IEEE_FLOAT, channels=2), data) + b"LIST\x04\x00\x00\x00meta"
    framer = WavFramer(TtsAudioFormat(headerless=False, rate=24000))

    assert struct.unpack("<2h", framer.feed(wav)) == (16384, -16384)
    assert framer.channels == 2


def test_parse_wav_header_reads_extensible_pcm_and_skips_chunks_before_the_data():
    wav = wav_file(extensible_fmt_chunk_body(1, bits=16, channels=2), PCM, chunks=b"LIST\x03\x00\x00\x00abc\x00")
    header = parse_wav_header(wav)

    assert header == WavHeader(24000, 2, 2, len(wav) - len(PCM), len(PCM))


@pytest.mark.parametrize(
    ("wav", "message"),
    [
        (wav_file(fmt_chunk_body(0x55, bits=0)), r"format 0x0055 \(MPEG Layer 3\) is not supported"),
        (wav_file(fmt_chunk_body(7, bits=8)), r"format 0x0007 \(mu-law\) is not supported"),
        (wav_file(fmt_chunk_body(0x1234, bits=16)), r"format 0x1234 is not supported"),
        (wav_file(fmt_chunk_body(WAVE_FORMAT_IEEE_FLOAT, bits=16)), "16-bit samples is not supported"),
        (wav_file(extensible_fmt_chunk_body(0x55)), r"format 0x0055 \(MPEG Layer 3\) is not supported"),
        (wav_file(extensible_fmt_chunk_body(3)[:-14] + b"\x00" * 14), "extensible subformat .* is not supported"),
        (wav_file(fmt_chunk_body(1, channels=0, bits=16)), "0 channels"),
        (wav_file(fmt_chunk_body(1, rate=0, bits=16)), "0 Hz"),
        (wav_file(fmt_chunk_body(1, bits=16)[:14]), "too short"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "not WAVE audio"),
        (b"RIFF\x00\x00\x00\x00WAVEdata\x00\x00\x00\x00", "before any fmt chunk"),
    ],
    ids=[
        "mp3",
        "mu-law",
        "unknown-tag",
        "float16",
        "extensible-mp3",
        "extensible-unknown-guid",
        "no-channels",
        "no-rate",
        "short-fmt",
        "not-wave",
        "data-before-fmt",
    ],
)
def test_framer_rejects_wav_it_cannot_play(wav, message):
    framer = WavFramer(TtsAudioFormat(headerless=False, rate=24000))

    with pytest.raises(WavFormatError, match=message):
        framer.feed(wav)


def test_framer_rejects_a_wav_file_without_a_data_chunk_in_reach():
    wav = wav_file(fmt_chunk_body(1, bits=16))[:-8]
    padding = b"LIST" + struct.pack("<I", TTS_WAV_HEADER_MAX_BYTES) + b"\x00" * TTS_WAV_HEADER_MAX_BYTES
    framer = WavFramer(TtsAudioFormat(headerless=False, rate=24000))

    with pytest.raises(WavFormatError, match="no data chunk"):
        framer.feed(wav + padding)


def test_take_whole_frames_prepends_a_carried_partial_frame():
    whole, partial = take_whole_frames(partial=b"\x01\x02", audio=b"\x03\x04\x05", frame_size=2)

    assert whole == b"\x01\x02\x03\x04"
    assert partial == b"\x05"


def test_take_whole_frames_keeps_a_partial_frame_when_frames_are_single_bytes():
    whole, partial = take_whole_frames(partial=b"\x01", audio=b"\x02", frame_size=1)

    assert whole == b"\x01\x02"
    assert partial == b""


def test_unsupported_format_error_says_how_to_work_around_it():
    framer = WavFramer(TtsAudioFormat(headerless=False, rate=24000))

    with pytest.raises(WavFormatError, match=r"""set TTS_EXTRA_BODY='\{"response_format": "pcm"\}'$"""):
        framer.feed(wav_file(fmt_chunk_body(7, bits=8)))
