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


def test_framer_plays_a_response_that_is_not_a_wav_file_as_raw_pcm_from_its_first_bytes():
    framer = WavFramer(TtsAudioFormat(headerless=False, rate=24000))

    # Held only until there are enough bytes to see it is not a WAV file
    assert framer.feed(PCM[:3]) == b""
    assert framer.feed(PCM[3:7]) == PCM[:6]
    assert framer.header_missing
    assert not framer.awaiting_header
    assert framer.rate == 24000
    # Later bytes pass straight through
    assert framer.feed(PCM[7:]) == PCM[6:]
    assert framer.finish() == b""


def test_framer_finish_hands_back_a_response_too_short_to_be_a_wav_file():
    framer = WavFramer(TtsAudioFormat(headerless=False, rate=24000))

    assert framer.feed(PCM[:3]) == b""
    assert framer.finish() == PCM[:2]
    assert framer.header_missing


WAV_FORMAT = TtsAudioFormat(headerless=False, rate=24000)
SPEECH_PCM_FORMAT = TtsAudioFormat(headerless=True, rate=24000, may_be_wav=True)
REALTIME_FORMAT = TtsAudioFormat(headerless=True, rate=24000)


@pytest.mark.parametrize("chunk_size", [1, 5, 4096])
def test_framer_reads_a_wav_file_returned_for_a_raw_pcm_request(chunk_size):
    """A backend that ignores `response_format: pcm` and answers with float WAV still plays, by its header."""
    data = b"".join(struct.pack("<f", x) for x in FLOATS)
    wav = wav_file(fmt_chunk_body(WAVE_FORMAT_IEEE_FLOAT, rate=16000), data)
    framer = WavFramer(SPEECH_PCM_FORMAT)

    pcm = b"".join(framer.feed(wav[i : i + chunk_size]) for i in range(0, len(wav), chunk_size)) + framer.finish()

    assert list(struct.unpack(f"<{len(FLOATS)}h", pcm)) == FLOATS_AS_PCM16
    assert (framer.rate, framer.width, framer.channels) == (16000, 2, 1)
    assert framer.unrequested_wav
    assert not framer.header_missing


def test_framer_plays_raw_pcm_returned_for_a_raw_pcm_request_without_reporting_it():
    framer = WavFramer(SPEECH_PCM_FORMAT)

    assert framer.feed(PCM[:3]) == b""
    assert framer.feed(PCM[3:]) == PCM
    assert framer.finish() == b""
    assert not framer.header_missing
    assert not framer.unrequested_wav


def test_framer_never_reads_realtime_audio_as_a_wav_file():
    """Realtime only carries raw PCM, so even bytes that spell a WAV header are played as they come."""
    wav = _wav()
    framer = WavFramer(REALTIME_FORMAT)

    assert framer.feed(wav) == wav
    assert not framer.unrequested_wav


FOREIGN_AUDIO = [
    (b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\xff\xfb\x90\x64" * 8, "MP3"),
    (b"fLaC\x00\x00\x00\x22" + bytes(34), "FLAC"),
    (b"OggS\x00\x02" + bytes(22), r"Ogg \(Opus or Vorbis\)"),
    (b"RF64\xff\xff\xff\xffWAVEds64" + bytes(28), "RF64 WAV"),
    (b"BW64\xff\xff\xff\xffWAVEds64" + bytes(28), "BW64 WAV"),
    (b"RIFX" + _wav()[4:], "big-endian RIFX WAV"),
]


@pytest.mark.parametrize(("audio", "name"), FOREIGN_AUDIO, ids=["mp3", "flac", "ogg", "rf64", "bw64", "rifx"])
@pytest.mark.parametrize("audio_format", [WAV_FORMAT, SPEECH_PCM_FORMAT], ids=["wav-requested", "pcm-requested"])
def test_framer_names_audio_of_another_kind_instead_of_playing_it(audio, name, audio_format):
    framer = WavFramer(audio_format)

    assert framer.feed(audio[:2]) == b""
    with pytest.raises(WavFormatError, match=f"^TTS response is {name} audio, which the proxy does not play"):
        framer.feed(audio[2:])


@pytest.mark.parametrize(
    "pcm",
    [b"\xff\xff\xfb\xff" * 8, b"\xff\xfb\x90\x64" * 8, b"ID3!" * 8, b"RIF\x00" * 8],
    ids=["near-silence", "mp3-frame-sync-without-id3", "id3-without-a-version", "riff-prefix-only"],
)
def test_framer_plays_raw_pcm_that_only_resembles_another_kind_of_audio(pcm):
    """Only whole signatures count: near-silent PCM starts with the same bits as an MP3 frame."""
    framer = WavFramer(WAV_FORMAT)

    assert framer.feed(pcm) + framer.finish() == pcm
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
