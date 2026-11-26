import io
import logging
import math
import random
import struct
import wave

import pytest
from wav_helpers import extensible_fmt_chunk_body, fmt_chunk_body, riff_chunk, wav_file

from wyoming_openai.utilities import take_whole_frames
from wyoming_openai.wav import (
    WAVE_FORMAT_IEEE_FLOAT,
    WAVE_FORMAT_PCM,
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
    assert parse_wav_header(_wav(rate=22050, channels=2)) == WavHeader(22050, 2, 2, 44, len(PCM), WAVE_FORMAT_PCM)


def test_parse_wav_header_reports_an_unbounded_data_size_as_unknown():
    wav = _wav()
    unbounded = wav[:40] + struct.pack("<I", 0xFFFFFFFF) + wav[44:]
    assert parse_wav_header(unbounded) == WavHeader(16000, 1, 2, 44, None, WAVE_FORMAT_PCM)


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

    # Held only until there are enough bytes to see it is neither a WAV file nor audio of another kind
    assert framer.feed(PCM[:7]) == b""
    assert framer.feed(PCM[7:9]) == PCM[:8]
    assert framer.header_missing
    assert not framer.awaiting_header
    assert framer.rate == 24000
    # Later bytes pass straight through
    assert framer.feed(PCM[9:]) == PCM[8:]
    assert framer.finish() == b""


def test_framer_finish_hands_back_a_response_too_short_to_be_a_wav_file():
    framer = WavFramer(TtsAudioFormat(headerless=False, rate=24000))

    assert framer.feed(PCM[:7]) == b""
    assert framer.finish() == PCM[:6]
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
    (b"FORM\x00\x00\x00\x2eAIFFCOMM" + bytes(26), "AIFF"),
    (b"caff\x00\x01\x00\x00desc" + bytes(40), "CAF"),
    (b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01" + bytes(24), "WebM or Matroska"),
    (b"\x00\x00\x00\x1cftypM4A \x00\x00\x00\x00" + bytes(16), "MP4 or M4A"),
]


@pytest.mark.parametrize(
    ("audio", "name"),
    FOREIGN_AUDIO,
    ids=["mp3", "flac", "ogg", "rf64", "bw64", "rifx", "aiff", "caf", "webm", "mp4"],
)
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


@pytest.mark.parametrize(
    "wav",
    [_wav()[:30], wav_file(fmt_chunk_body(1, bits=16))[:-8] + riff_chunk(b"LIST", b"meta")],
    ids=["ended-inside-the-header", "no-data-chunk"],
)
def test_framer_finds_no_audio_in_a_response_that_ended_inside_the_wav_header(wav, caplog):
    framer = WavFramer(TtsAudioFormat(headerless=False, rate=24000))

    assert framer.feed(wav) == b""
    with caplog.at_level(logging.WARNING, logger="wyoming_openai.wav"):
        assert framer.finish() == b""

    assert f"TTS response ended after {len(wav)} bytes, inside its WAV header" in caplog.text
    assert not framer.awaiting_header
    assert not framer.header_missing
    assert framer.missing_bytes == 0


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

    assert header == WavHeader(24000, 2, 2, len(wav) - len(PCM), len(PCM), WAVE_FORMAT_PCM)


@pytest.mark.parametrize(
    ("wav", "message"),
    [
        (wav_file(fmt_chunk_body(0x55, bits=0)), r"format 0x0055 \(MPEG Layer 3\) is not supported"),
        (wav_file(fmt_chunk_body(7, bits=8)), r"format 0x0007 \(mu-law\) is not supported"),
        (wav_file(fmt_chunk_body(0x1234, bits=16)), r"format 0x1234 is not supported"),
        (wav_file(fmt_chunk_body(WAVE_FORMAT_IEEE_FLOAT, bits=16)), "16-bit samples is not supported"),
        (wav_file(fmt_chunk_body(WAVE_FORMAT_PCM, bits=40)), r"integer PCM with 40-bit samples is not supported"),
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
        "int40",
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


@pytest.mark.parametrize("chunk_size", [7, 2048])
def test_framer_drops_metadata_before_the_audio_as_it_arrives(chunk_size):
    """Metadata of any length before the data chunk, such as embedded artwork, is not held while it is skipped."""
    artwork = riff_chunk(b"LIST", b"INFO" + bytes(200_001))
    wav = wav_file(fmt_chunk_body(WAVE_FORMAT_PCM, bits=16), PCM, chunks=artwork)
    framer = WavFramer(WAV_FORMAT)
    pcm, most_held = b"", 0
    for i in range(0, len(wav), chunk_size):
        pcm += framer.feed(wav[i : i + chunk_size])
        if framer._header is not None:
            most_held = max(most_held, len(framer._header.held))

    assert pcm + framer.finish() == PCM
    # No more than one read and the chunk header or fmt chunk it completes
    assert most_held < chunk_size + 48


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


def test_unsupported_format_error_offers_no_workaround_when_raw_pcm_was_already_requested():
    framer = WavFramer(SPEECH_PCM_FORMAT)

    with pytest.raises(WavFormatError, match=r"format 0x0007 \(mu-law\) is not supported") as error:
        framer.feed(wav_file(fmt_chunk_body(7, bits=8)))
    assert "TTS_EXTRA_BODY" not in str(error.value)


def _wave_written(width: int, channels: int, rate: int, frames: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(bytes((i * 37) % 256 for i in range(frames * channels * width)))
    return buffer.getvalue()


def _with_chunks(wav: bytes, before_fmt: bytes = b"", before_data: bytes = b"") -> bytes:
    """Insert chunks into a file `wave` wrote, whose fmt chunk is the plain 16-byte one."""
    body = wav[8:12] + before_fmt + wav[12:36] + before_data + wav[36:]
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _assert_matches_the_wave_module(wav: bytes) -> None:
    """Check the parser and the framer read the format, data offset, size and audio that `wave` does."""
    buffer = io.BytesIO(wav)
    with wave.open(buffer, "rb") as reader:
        data_offset = buffer.tell()
        rate, channels, width, frames = (
            reader.getframerate(),
            reader.getnchannels(),
            reader.getsampwidth(),
            reader.getnframes(),
        )
        audio = reader.readframes(frames)

    assert parse_wav_header(wav) == WavHeader(
        rate, channels, width, data_offset, frames * channels * width, WAVE_FORMAT_PCM
    )
    framer = WavFramer(WAV_FORMAT)
    assert b"".join(framer.feed(wav[i : i + 7]) for i in range(0, len(wav), 7)) + framer.finish() == audio
    assert (framer.rate, framer.width, framer.channels) == (rate, width, channels)
    assert framer.missing_bytes == 0


WAVE_LAYOUTS = {
    "plain": lambda wav: wav,
    "list-before-fmt": lambda wav: _with_chunks(wav, before_fmt=riff_chunk(b"LIST", b"INFOISFTx\x00")),
    "chunks-before-data": lambda wav: _with_chunks(
        wav, before_data=riff_chunk(b"fact", struct.pack("<I", 0)) + riff_chunk(b"iXML", b"<a/> ")
    ),
}


@pytest.mark.parametrize("layout", WAVE_LAYOUTS)
@pytest.mark.parametrize("frames", [0, 1, 37])
@pytest.mark.parametrize("rate", [8000, 24000, 48000])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_files_the_wave_module_writes_are_read_as_it_reads_them(width, channels, rate, frames, layout):
    """The parser replaced the stdlib `wave` module, so every integer PCM file `wave` reads must read the same."""
    _assert_matches_the_wave_module(WAVE_LAYOUTS[layout](_wave_written(width, channels, rate, frames)))


@pytest.mark.parametrize(
    "wav",
    [
        wav_file(extensible_fmt_chunk_body(1, bits=16, channels=2), PCM),
        wav_file(extensible_fmt_chunk_body(1, bits=32, rate=48000), PCM[:480]),
        wav_file(fmt_chunk_body(1, bits=16, extra=b"\x00\x00"), PCM),
        wav_file(fmt_chunk_body(1, bits=16), PCM, before_fmt=riff_chunk(b"JUNK", bytes(28))),
        wav_file(fmt_chunk_body(1, bits=16), PCM[:-1]),
        wav_file(fmt_chunk_body(1, bits=16, extra=bytes(100)), PCM),
        wav_file(fmt_chunk_body(1, bits=16), PCM, chunks=riff_chunk(b"LIST", b"INFO" + bytes(100_001))),
    ],
    ids=[
        "extensible-stereo",
        "extensible-32-bit",
        "fmt-with-cbsize",
        "junk-before-fmt",
        "odd-data-size",
        "long-fmt",
        "metadata-past-64-kib",
    ],
)
def test_other_files_the_wave_module_reads_are_read_as_it_reads_them(wav):
    _assert_matches_the_wave_module(wav)


FLOAT_SAMPLES = [0.25, -0.25, 0.5, -1.0] * 300
FLOAT_SAMPLES_AS_PCM16 = struct.pack("<4h", 8192, -8192, 16384, -32767) * 300
PCM24 = bytes((i * 11) % 256 for i in range(6 * 200))
LAVF_INFO = riff_chunk(b"LIST", b"INFO" + riff_chunk(b"ISFT", b"Lavf61.7.100\x00"))
UNBOUNDED = 0xFFFFFFFF

# Headers as the backends that write them produce them, each with the audio and format it should play as
SERVER_RESPONSES = {
    # FFmpeg writing to a pipe (Speaches, Kokoro-FastAPI through PyAV): placeholder sizes and a LIST chunk
    "ffmpeg-pipe": (
        wav_file(fmt_chunk_body(1, bits=16), PCM, LAVF_INFO, riff_size=UNBOUNDED, data_size=UNBOUNDED),
        PCM,
        (24000, 2, 1),
    ),
    # FFmpeg switches to WAVE_FORMAT_EXTENSIBLE above 16-bit integer samples or 48 kHz
    "ffmpeg-extensible-24-bit": (
        wav_file(extensible_fmt_chunk_body(1, bits=24, channels=2, rate=48000), PCM24, LAVF_INFO),
        PCM24,
        (48000, 3, 2),
    ),
    # Kokoros and Lemonade's fork: float32 with placeholder sizes, even in a complete file
    "kokoros-float32": (
        wav_file(
            fmt_chunk_body(WAVE_FORMAT_IEEE_FLOAT),
            struct.pack(f"<{len(FLOAT_SAMPLES)}f", *FLOAT_SAMPLES),
            riff_size=UNBOUNDED,
            data_size=UNBOUNDED,
        ),
        FLOAT_SAMPLES_AS_PCM16,
        (24000, 2, 1),
    ),
    # OpenVINO Model Server up to 2026.4: float32 with real sizes, whatever format was requested
    "ovms-float32": (
        wav_file(fmt_chunk_body(WAVE_FORMAT_IEEE_FLOAT), struct.pack(f"<{len(FLOAT_SAMPLES)}f", *FLOAT_SAMPLES)),
        FLOAT_SAMPLES_AS_PCM16,
        (24000, 2, 1),
    ),
    # chatterbox-tts-api streaming: a RIFF size just short of the signed placeholder, and an unbounded data size
    "chatterbox-stream": (
        wav_file(fmt_chunk_body(1, bits=16), PCM, riff_size=0x7FFFFFDB, data_size=UNBOUNDED),
        PCM,
        (24000, 2, 1),
    ),
}


@pytest.mark.parametrize("chunk_size", [1, 7, 2048])
@pytest.mark.parametrize("server", SERVER_RESPONSES)
def test_framer_plays_wav_as_tts_servers_write_it(server, chunk_size):
    wav, audio, audio_format = SERVER_RESPONSES[server]
    framer = WavFramer(WAV_FORMAT)

    pcm = b"".join(framer.feed(wav[i : i + chunk_size]) for i in range(0, len(wav), chunk_size)) + framer.finish()

    assert pcm == audio
    assert (framer.rate, framer.width, framer.channels) == audio_format
    assert framer.missing_bytes == 0
    assert not framer.header_missing


@pytest.mark.parametrize(("code", "bits"), [("<f", 32), ("<d", 64)], ids=["float32", "float64"])
def test_float_conversion_clips_and_rounds_every_sample(code, bits):
    rng = random.Random(72)
    values = [rng.uniform(-1.5, 1.5) for _ in range(2000)]
    values += [math.inf, -math.inf, math.nan, 1.0, -1.0, 0.0, -0.0, 5e-324, 1 / 65534, -1 / 65534, 0.99999]
    data = b"".join(struct.pack(code, x) for x in values)
    # What the file holds, which for float32 is each value rounded to single precision
    stored = struct.unpack(f"<{len(values)}{code[1]}", data)
    expected = [0 if math.isnan(x) else round(32767 * max(-1.0, min(1.0, x))) for x in stored]
    framer = WavFramer(WAV_FORMAT)

    pcm = framer.feed(wav_file(fmt_chunk_body(WAVE_FORMAT_IEEE_FLOAT, bits=bits), data)) + framer.finish()

    assert list(struct.unpack(f"<{len(values)}h", pcm)) == expected
