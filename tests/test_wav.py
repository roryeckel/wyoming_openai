import io
import struct
import wave

import pytest

from wyoming_openai.utilities import take_whole_frames
from wyoming_openai.wav import TTS_WAV_HEADER_MAX_BYTES, TtsAudioFormat, WavFramer, parse_wav_header

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
    assert parse_wav_header(_wav(rate=22050, channels=2)) == (22050, 2, 2, 44, len(PCM))


def test_parse_wav_header_reports_an_unbounded_data_size_as_unknown():
    wav = _wav()
    unbounded = wav[:40] + struct.pack("<I", 0xFFFFFFFF) + wav[44:]
    assert parse_wav_header(unbounded) == (16000, 1, 2, 44, None)


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


def _wav_with_extended_fmt(pcm: bytes, rate: int = 22050, channels: int = 2) -> bytes:
    """A PCM WAV whose fmt chunk has the 18-byte form, which leaves the audio at an odd offset of 46 bytes."""
    block_align = channels * 2
    fmt = struct.pack("<HHIIHHH", 1, channels, rate, rate * block_align, block_align, 16, 0)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(pcm)) + pcm
    return b"RIFF" + struct.pack("<I", len(body)) + body


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


def test_framer_finish_hands_back_a_response_that_ended_inside_the_header():
    framer = WavFramer(TtsAudioFormat(headerless=False, rate=24000))

    assert framer.feed(b"RIFF\x00\x00") == b""
    assert framer.finish() == b"RIFF\x00\x00"
    assert framer.header_missing


def test_take_whole_frames_prepends_a_carried_partial_frame():
    whole, partial = take_whole_frames(partial=b"\x01\x02", audio=b"\x03\x04\x05", frame_size=2)

    assert whole == b"\x01\x02\x03\x04"
    assert partial == b"\x05"


def test_take_whole_frames_keeps_a_partial_frame_when_frames_are_single_bytes():
    whole, partial = take_whole_frames(partial=b"\x01", audio=b"\x02", frame_size=1)

    assert whole == b"\x01\x02"
    assert partial == b""
