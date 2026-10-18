"""Tests for early transcription (``--stt-early-transcribe``): the pause detector and its use by the handler."""

import asyncio
import io
import logging
import math
import struct
import wave
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import omit
from openai.types.audio import Transcription
from wyoming.asr import Transcript
from wyoming.event import Event

from wyoming_openai.compatibility import OpenAIBackend, create_asr_programs, create_info
from wyoming_openai.early_transcription import PauseDetector, Signal
from wyoming_openai.handler import SHUTDOWN_ERROR, STT_FAILED_ERROR, OpenAIEventHandler

RATE = 16000
BYTES_PER_MS = RATE * 2 // 1000  # 16-bit mono
CHUNK_MS = 32
AUDIO_FORMAT = {"rate": RATE, "width": 2, "channels": 1}

pytestmark = pytest.mark.timeout(30)  # a regression must not hang the suite while a test waits for the early request

# Peak amplitudes of the synthetic recordings (the RMS level is about 0.7 of these)
SPEECH = 4000  # clearly audible
SOFT = 250  # a soft word ending: above the background but far below speech, so neither loud nor quiet
HUSH = 30  # background of a quiet room


def tone(ms: int, amplitude: int) -> bytes:
    """16-bit mono PCM sine wave of the given length and peak amplitude."""
    samples = RATE * ms // 1000
    wave_form = (round(amplitude * math.sin(2 * math.pi * 220 * n / RATE)) for n in range(samples))
    return struct.pack(f"<{samples}h", *wave_form)


def run_detector(*parts: bytes, chunk_ms: int = CHUNK_MS) -> list[tuple[int, Signal]]:
    """Feed the parts in chunks; return (ms of audio fed so far, signal) for every signal."""
    detector = PauseDetector(rate=RATE, width=2, channels=1)
    chunk_bytes = chunk_ms * BYTES_PER_MS
    signals = []
    fed_bytes = 0
    for pcm in parts:
        for offset in range(0, len(pcm), chunk_bytes):
            chunk = pcm[offset : offset + chunk_bytes]
            fed_bytes += len(chunk)
            signal = detector.feed(chunk)
            if signal is not Signal.NONE:
                signals.append((fed_bytes // BYTES_PER_MS, signal))
    return signals


# ---------------------------------------------------------------------------------------------------------------------
# PauseDetector
# ---------------------------------------------------------------------------------------------------------------------


def test_pause_is_reported_after_speech_and_300_ms_of_quiet():
    signals = run_detector(tone(256, HUSH), tone(1024, SPEECH), tone(640, HUSH))

    # 10 quiet chunks of 32 ms are the first to reach 300 ms
    assert signals == [(256 + 1024 + 320, Signal.PAUSE)]


def test_no_pause_without_speech():
    assert run_detector(tone(2048, HUSH)) == []


def test_steady_background_noise_is_not_speech():
    assert run_detector(tone(2048, 200)) == []


def test_short_blip_is_not_speech():
    assert run_detector(tone(256, HUSH), tone(96, SPEECH), tone(1024, HUSH)) == []


def test_soft_sound_after_speech_is_not_a_pause():
    assert run_detector(tone(256, HUSH), tone(1024, SPEECH), tone(1024, SOFT)) == []


def test_pause_is_found_in_a_noisy_room():
    signals = run_detector(tone(512, 400), tone(1024, 6000), tone(640, 400))

    assert signals == [(512 + 1024 + 320, Signal.PAUSE)]


def test_sound_after_the_pause_makes_it_stale_and_a_later_pause_is_reported_again():
    signals = run_detector(tone(256, HUSH), tone(1024, SPEECH), tone(352, HUSH), tone(512, SPEECH), tone(640, HUSH))

    assert signals == [
        (256 + 1024 + 320, Signal.PAUSE),
        (256 + 1024 + 352 + 128, Signal.RESUMED),  # the fourth chunk of sound is the first to reach 100 ms
        (256 + 1024 + 352 + 512 + 320, Signal.PAUSE),
    ]


def test_soft_sound_after_the_pause_makes_it_stale():
    signals = run_detector(tone(256, HUSH), tone(1024, SPEECH), tone(320, HUSH), tone(128, SOFT))

    assert [signal for _, signal in signals] == [Signal.PAUSE, Signal.RESUMED]


def test_resume_signal_waits_for_100_ms_of_sound_after_the_pause():
    short_sound = run_detector(tone(256, HUSH), tone(1024, SPEECH), tone(320, HUSH), tone(96, SPEECH))
    longer_sound = run_detector(tone(256, HUSH), tone(1024, SPEECH), tone(320, HUSH), tone(128, SPEECH))

    assert [signal for _, signal in short_sound] == [Signal.PAUSE]
    assert [signal for _, signal in longer_sound] == [Signal.PAUSE, Signal.RESUMED]


def test_at_most_three_pauses_are_reported():
    utterance = (tone(512, SPEECH), tone(352, HUSH))
    signals = run_detector(tone(256, HUSH), *utterance * 5)

    assert [signal for _, signal in signals].count(Signal.PAUSE) == 3


@pytest.mark.parametrize("chunk_ms", [10, 32, 64])
def test_detection_does_not_depend_on_the_chunk_size(chunk_ms):
    """A long stretch of uniform speech must not raise the background level until the speech itself looks quiet."""
    speech_ms = 12000
    signals = run_detector(tone(256, HUSH), tone(speech_ms, 1000), tone(640, HUSH), chunk_ms=chunk_ms)

    assert [signal for _, signal in signals] == [Signal.PAUSE]
    pause_at_ms = signals[0][0]
    assert 256 + speech_ms + 300 <= pause_at_ms <= 256 + speech_ms + 300 + chunk_ms


def test_pause_detector_needs_16_bit_audio():
    with pytest.raises(ValueError, match="16-bit"):
        PauseDetector(rate=RATE, width=1, channels=1)


# ---------------------------------------------------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------------------------------------------------


def frames_of(wav_file) -> int:
    with wave.open(io.BytesIO(wav_file.getvalue())) as wav:
        return wav.getnframes()


class FakeStt:
    """STT backend double: records every request and answers with the number of audio frames the request carried."""

    def __init__(self, *, fail_first=False, hold_first=False):
        self.requests: list[dict] = []
        self.fail_first = fail_first
        self.hold_first = hold_first
        self.release = asyncio.Event()
        self.first_request_cancelled = False
        self.client = MagicMock()
        self.client.audio.transcriptions.create = AsyncMock(side_effect=self._create)

    @property
    def frames(self) -> list[int]:
        return [request["frames"] for request in self.requests]

    async def _create(self, **kwargs):
        request = {**kwargs, "frames": frames_of(kwargs["file"])}
        self.requests.append(request)
        is_first = len(self.requests) == 1
        if is_first and self.hold_first:
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.first_request_cancelled = True
                raise
        if is_first and self.fail_first:
            raise RuntimeError("backend unavailable")
        return Transcription(text=f"{request['frames']} frames")


def make_handler(stt: FakeStt, *, streaming_model=False, model_name="whisper-1", programs=None, **handler_kwargs):
    if programs is None:
        programs = create_asr_programs(
            [] if streaming_model else [model_name], [model_name] if streaming_model else [], "url", ["en", "de"]
        )
    handler = OpenAIEventHandler(
        MagicMock(),
        MagicMock(),
        info=create_info(programs, []),
        stt_client=stt.client,
        tts_client=None,
        **handler_kwargs,
    )
    handler.write_event = AsyncMock()
    return handler


async def start_transcribing(handler: OpenAIEventHandler) -> None:
    assert await handler.handle_event(Event(type="transcribe", data={"name": "whisper-1", "language": "en"})) is True


async def stream(handler: OpenAIEventHandler, *parts: bytes, audio_format=AUDIO_FORMAT) -> None:
    """Send audio-start and the audio as 32 ms chunks, giving the event loop a turn after each chunk like a socket."""
    await handler.handle_event(Event(type="audio-start", data=audio_format))
    chunk_bytes = CHUNK_MS * audio_format["rate"] * audio_format["width"] * audio_format["channels"] // 1000
    for pcm in parts:
        for offset in range(0, len(pcm), chunk_bytes):
            payload = pcm[offset : offset + chunk_bytes]
            await handler.handle_event(Event(type="audio-chunk", data=audio_format, payload=payload))
            await asyncio.sleep(0)


async def stop_streaming(handler: OpenAIEventHandler) -> None:
    await handler.handle_event(Event(type="audio-stop"))


def transcripts(handler) -> list[str]:
    events = [call.args[0] for call in cast(AsyncMock, handler.write_event).await_args_list]
    return [Transcript.from_event(event).text for event in events if Transcript.is_type(event.type)]


def frames_for(ms: int) -> int:
    return ms * RATE // 1000


QUIET_ROOM = tone(256, HUSH)
ONE_SECOND_OF_SPEECH = tone(1024, SPEECH)


@pytest.mark.asyncio
async def test_pause_sends_the_request_early_and_audio_stop_uses_its_answer():
    stt = FakeStt()
    handler = make_handler(stt, stt_early_transcribe=True)
    await start_transcribing(handler)

    await stream(handler, QUIET_ROOM, ONE_SECOND_OF_SPEECH, tone(640, HUSH))

    # The request has left during the pause: it carries the speech and the first 320 ms of the silence
    assert stt.frames == [frames_for(256 + 1024 + 320)]

    await stop_streaming(handler)

    assert stt.frames == [frames_for(256 + 1024 + 320)]
    assert transcripts(handler) == [f"{frames_for(256 + 1024 + 320)} frames"]


@pytest.mark.asyncio
async def test_early_request_carries_the_same_settings_as_the_normal_request():
    settings = {"stt_temperature": 0.2, "stt_prompt": "kitchen light", "stt_extra_body": {"vad_filter": True}}
    recording = (QUIET_ROOM, ONE_SECOND_OF_SPEECH, tone(640, HUSH))
    early, normal = FakeStt(), FakeStt()
    for stt, early_transcribe in ((early, True), (normal, False)):
        handler = make_handler(stt, stt_early_transcribe=early_transcribe, **settings)
        await start_transcribing(handler)
        await stream(handler, *recording)
        await stop_streaming(handler)

    assert len(early.requests) == len(normal.requests) == 1
    early_settings = {key: value for key, value in early.requests[0].items() if key not in ("file", "frames")}
    normal_settings = {key: value for key, value in normal.requests[0].items() if key not in ("file", "frames")}
    assert early_settings == normal_settings
    assert early_settings["language"] == "en"
    assert early_settings["temperature"] == 0.2
    assert early_settings["prompt"] == "kitchen light"
    assert early_settings["extra_body"] == {"vad_filter": True}
    assert early.requests[0]["file"].name == normal.requests[0]["file"].name == "recording.wav"


@pytest.mark.asyncio
async def test_sound_after_the_pause_discards_the_early_answer_and_sends_the_whole_recording():
    stt = FakeStt()
    handler = make_handler(stt, stt_early_transcribe=True)
    await start_transcribing(handler)

    await stream(handler, QUIET_ROOM, ONE_SECOND_OF_SPEECH, tone(352, HUSH), tone(256, SPEECH))
    await stop_streaming(handler)

    whole_recording = frames_for(256 + 1024 + 352 + 256)
    assert stt.frames == [frames_for(256 + 1024 + 320), whole_recording]
    assert transcripts(handler) == [f"{whole_recording} frames"]


@pytest.mark.asyncio
async def test_a_later_pause_gets_its_own_early_request_which_is_used():
    stt = FakeStt()
    handler = make_handler(stt, stt_early_transcribe=True)
    await start_transcribing(handler)

    await stream(handler, QUIET_ROOM, ONE_SECOND_OF_SPEECH, tone(352, HUSH), tone(512, SPEECH), tone(640, HUSH))
    await stop_streaming(handler)

    second_pause = frames_for(256 + 1024 + 352 + 512 + 320)
    assert stt.frames == [frames_for(256 + 1024 + 320), second_pause]
    assert transcripts(handler) == [f"{second_pause} frames"]


@pytest.mark.asyncio
async def test_failed_early_request_falls_back_to_the_normal_request(caplog):
    stt = FakeStt(fail_first=True)
    handler = make_handler(stt, stt_early_transcribe=True)
    await start_transcribing(handler)

    with caplog.at_level(logging.WARNING, logger="wyoming_openai.early_transcription"):
        await stream(handler, QUIET_ROOM, ONE_SECOND_OF_SPEECH, tone(640, HUSH))
        await stop_streaming(handler)

    whole_recording = frames_for(256 + 1024 + 640)
    assert stt.frames == [frames_for(256 + 1024 + 320), whole_recording]
    assert transcripts(handler) == [f"{whole_recording} frames"]
    assert "Early transcription failed" in caplog.text


@pytest.mark.asyncio
async def test_nothing_is_sent_early_unless_enabled():
    stt = FakeStt()
    handler = make_handler(stt)
    await start_transcribing(handler)

    await stream(handler, QUIET_ROOM, ONE_SECOND_OF_SPEECH, tone(640, HUSH))
    assert stt.requests == []

    await stop_streaming(handler)
    assert stt.frames == [frames_for(256 + 1024 + 640)]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["streaming_model", "stream_in_extra_body"])
async def test_streamed_transcription_is_never_sent_early(case):
    stt = FakeStt()
    if case == "streaming_model":
        handler = make_handler(stt, streaming_model=True, stt_early_transcribe=True)
    else:
        handler = make_handler(stt, stt_early_transcribe=True, stt_extra_body={"stream": True})
    await start_transcribing(handler)

    await stream(handler, QUIET_ROOM, ONE_SECOND_OF_SPEECH, tone(640, HUSH))

    assert stt.requests == []


@pytest.mark.asyncio
async def test_audio_that_is_not_16_bit_is_never_sent_early():
    stt = FakeStt()
    handler = make_handler(stt, stt_early_transcribe=True)
    await start_transcribing(handler)
    eight_bit_audio = {"rate": RATE, "width": 1, "channels": 1}

    await stream(handler, b"\x80" * RATE, audio_format=eight_bit_audio)
    assert stt.requests == []

    await stop_streaming(handler)
    assert stt.frames == [RATE]


@pytest.mark.asyncio
async def test_disconnect_cancels_a_running_early_request():
    stt = FakeStt(hold_first=True)
    handler = make_handler(stt, stt_early_transcribe=True)
    await start_transcribing(handler)
    await stream(handler, QUIET_ROOM, ONE_SECOND_OF_SPEECH, tone(640, HUSH))
    assert len(stt.requests) == 1

    await handler.disconnect()
    await asyncio.sleep(0)

    assert stt.first_request_cancelled is True


@pytest.mark.asyncio
async def test_new_transcribe_request_cancels_a_running_early_request():
    stt = FakeStt(hold_first=True)
    handler = make_handler(stt, stt_early_transcribe=True)
    await start_transcribing(handler)
    await stream(handler, QUIET_ROOM, ONE_SECOND_OF_SPEECH, tone(640, HUSH))
    assert len(stt.requests) == 1

    await start_transcribing(handler)
    await asyncio.sleep(0)
    await stop_streaming(handler)

    assert stt.first_request_cancelled is True
    assert stt.frames == [frames_for(256 + 1024 + 320), frames_for(256 + 1024 + 640)]
    assert transcripts(handler) == [f"{frames_for(256 + 1024 + 640)} frames"]


@pytest.mark.asyncio
async def test_audio_stop_waits_for_a_running_early_request_instead_of_sending_another():
    stt = FakeStt(hold_first=True)
    handler = make_handler(stt, stt_early_transcribe=True)
    await start_transcribing(handler)
    await stream(handler, QUIET_ROOM, ONE_SECOND_OF_SPEECH, tone(640, HUSH))

    stopping = asyncio.create_task(stop_streaming(handler))
    for _ in range(5):
        await asyncio.sleep(0)
    assert stopping.done() is False
    assert len(stt.requests) == 1

    stt.release.set()
    await stopping

    assert len(stt.requests) == 1
    assert transcripts(handler) == [f"{frames_for(256 + 1024 + 320)} frames"]


@pytest.mark.asyncio
@pytest.mark.parametrize("amplitude", [SPEECH, SOFT])
@pytest.mark.parametrize("trailing_quiet_ms", [0, 640])
async def test_even_a_brief_sound_after_the_snapshot_requires_the_whole_recording(amplitude, trailing_quiet_ms):
    stt = FakeStt()
    handler = make_handler(stt, stt_early_transcribe=True)
    await start_transcribing(handler)
    await stream(
        handler, QUIET_ROOM, ONE_SECOND_OF_SPEECH, tone(320, HUSH), tone(64, amplitude), tone(trailing_quiet_ms, HUSH)
    )
    await stop_streaming(handler)

    whole_recording = frames_for(256 + 1024 + 320 + 64 + trailing_quiet_ms)
    assert stt.frames == [frames_for(256 + 1024 + 320), whole_recording]
    assert transcripts(handler) == [f"{whole_recording} frames"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra_body,expected_languages,expected_extra",
    [
        (None, ["en"], None),
        ({"language": "de"}, ["de"], None),
        ({"languages": ["en", "de"]}, omit, {"languages": ["en", "de"]}),
    ],
)
async def test_early_request_preserves_plural_language_mapping(extra_body, expected_languages, expected_extra):
    stt = FakeStt()
    stt.client.backend = OpenAIBackend.OPENAI
    handler = make_handler(stt, model_name="gpt-transcribe", stt_early_transcribe=True, stt_extra_body=extra_body)
    assert await handler.handle_event(Event(type="transcribe", data={"name": "gpt-transcribe", "language": "en"}))

    await stream(handler, QUIET_ROOM, ONE_SECOND_OF_SPEECH, tone(640, HUSH))
    assert len(stt.requests) == 1  # The request was sent before AudioStop.
    await stop_streaming(handler)

    request = stt.requests[0]
    assert request["language"] is omit
    assert request["languages"] == expected_languages
    assert request.get("extra_body") == expected_extra
    assert len(stt.requests) == 1


@pytest.mark.asyncio
async def test_default_model_can_transcribe_early_without_a_transcribe_event():
    stt = FakeStt()
    handler = make_handler(stt, stt_early_transcribe=True)

    await stream(handler, QUIET_ROOM, ONE_SECOND_OF_SPEECH, tone(640, HUSH))
    assert stt.frames == [frames_for(256 + 1024 + 320)]
    await stop_streaming(handler)

    assert len(stt.requests) == 1
    assert stt.requests[0]["model"] == "whisper-1"


@pytest.mark.asyncio
async def test_new_model_and_language_replace_the_early_request():
    stt = FakeStt(hold_first=True)
    programs = create_asr_programs(["whisper-1", "gpt-transcribe"], [], "url", ["en", "de"])
    stt.client.backend = OpenAIBackend.OPENAI
    handler = make_handler(stt, programs=programs, stt_early_transcribe=True)
    await start_transcribing(handler)
    await stream(handler, QUIET_ROOM, ONE_SECOND_OF_SPEECH, tone(640, HUSH))

    assert await handler.handle_event(Event(type="transcribe", data={"name": "gpt-transcribe", "language": "de"}))
    await stop_streaming(handler)
    await asyncio.sleep(0)

    assert stt.first_request_cancelled
    assert stt.frames == [frames_for(256 + 1024 + 320), frames_for(256 + 1024 + 640)]
    assert stt.requests[1]["model"] == "gpt-transcribe"
    assert stt.requests[1]["language"] is omit
    assert stt.requests[1]["languages"] == ["de"]


@pytest.mark.asyncio
async def test_selecting_a_different_program_keeps_the_recordings_early_model():
    stt = FakeStt()
    programs = create_asr_programs(["whisper-1"], ["other-model"], "url", ["en"])
    handler = make_handler(stt, programs=programs, stt_early_transcribe=True)
    await start_transcribing(handler)
    await stream(handler, QUIET_ROOM, ONE_SECOND_OF_SPEECH, tone(640, HUSH))

    assert await handler.handle_event(Event(type="select-program", data={"name": "openai-streaming"}))
    await stop_streaming(handler)

    assert len(stt.requests) == 1
    assert stt.requests[0]["model"] == "whisper-1"
    assert transcripts(handler) == [f"{frames_for(256 + 1024 + 320)} frames"]


@pytest.mark.asyncio
async def test_stream_false_override_allows_an_early_http_request():
    stt = FakeStt()
    handler = make_handler(stt, streaming_model=True, stt_early_transcribe=True, stt_extra_body={"stream": False})
    await start_transcribing(handler)

    await stream(handler, QUIET_ROOM, ONE_SECOND_OF_SPEECH, tone(640, HUSH))
    assert len(stt.requests) == 1
    await stop_streaming(handler)

    assert len(stt.requests) == 1
    assert stt.requests[0]["stream"] is omit
    assert stt.requests[0]["extra_body"] == {"stream": False}


@pytest.mark.asyncio
async def test_realtime_audio_never_starts_an_early_http_request():
    stt = FakeStt()
    handler = make_handler(stt, stt_realtime_models=["whisper-1"], stt_early_transcribe=True)
    handler._handle_realtime_audio_start = AsyncMock()
    handler._handle_realtime_audio_chunk = AsyncMock()
    await start_transcribing(handler)

    await stream(handler, QUIET_ROOM, ONE_SECOND_OF_SPEECH, tone(640, HUSH))

    handler._handle_realtime_audio_start.assert_awaited_once()
    assert handler._handle_realtime_audio_chunk.await_count > 0
    assert stt.requests == []
    assert handler._early_transcription is None
    await handler.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["audio-start", "refused-transcribe"])
async def test_abandoning_a_recording_cancels_its_early_request(replacement):
    stt = FakeStt(hold_first=True)
    handler = make_handler(stt, stt_early_transcribe=True)
    await start_transcribing(handler)
    await stream(handler, QUIET_ROOM, ONE_SECOND_OF_SPEECH, tone(640, HUSH))

    if replacement == "audio-start":
        await handler.handle_event(Event(type="audio-start", data=AUDIO_FORMAT))
    else:
        assert not await handler.handle_event(Event(type="transcribe", data={"name": "missing-model"}))
    await asyncio.sleep(0)

    assert stt.first_request_cancelled
    assert len(stt.requests) == 1
    assert transcripts(handler) == []
    await handler.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("awaiting_stop", [False, True])
@pytest.mark.parametrize("cleanup", ["stop", "disconnect"])
async def test_cleanup_cancels_an_early_request_even_while_audio_stop_awaits_it(cleanup, awaiting_stop):
    stt = FakeStt(hold_first=True)
    handler = make_handler(stt, stt_early_transcribe=True)
    await start_transcribing(handler)
    await stream(handler, QUIET_ROOM, ONE_SECOND_OF_SPEECH, tone(640, HUSH))

    stopping = None
    if awaiting_stop:
        stopping = asyncio.create_task(handler.handle_event(Event(type="audio-stop")))
        for _ in range(3):
            await asyncio.sleep(0)
        assert not stopping.done()

    await getattr(handler, cleanup)()
    await asyncio.sleep(0)
    if stopping is not None:
        assert await stopping is False

    assert stt.first_request_cancelled
    assert len(stt.requests) == 1  # Cleanup must not trigger a fallback request.
    assert transcripts(handler) == []
    events = [call.args[0] for call in cast(AsyncMock, handler.write_event).await_args_list]
    assert [event.data["text"] for event in events if event.type == "error"] == (
        [SHUTDOWN_ERROR] if cleanup == "stop" else []
    )
    await handler.disconnect()


@pytest.mark.asyncio
async def test_cancelling_audio_stop_cancels_the_request_and_discards_its_recording():
    stt = FakeStt(hold_first=True)
    handler = make_handler(stt, stt_early_transcribe=True)
    await start_transcribing(handler)
    await stream(handler, QUIET_ROOM, ONE_SECOND_OF_SPEECH, tone(640, HUSH))
    stopping = asyncio.create_task(handler.handle_event(Event(type="audio-stop")))
    for _ in range(3):
        await asyncio.sleep(0)

    stopping.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stopping

    assert stt.first_request_cancelled
    assert len(stt.requests) == 1
    assert handler._wav_buffer is None
    assert handler._early_transcription is None
    assert transcripts(handler) == []


@pytest.mark.asyncio
async def test_failed_fallback_reports_the_current_stt_error_contract():
    stt = FakeStt()
    stt.client.audio.transcriptions.create.side_effect = RuntimeError("backend unavailable")
    handler = make_handler(stt, stt_early_transcribe=True)
    await start_transcribing(handler)
    await stream(handler, QUIET_ROOM, ONE_SECOND_OF_SPEECH, tone(640, HUSH))

    assert not await handler.handle_event(Event(type="audio-stop"))

    assert stt.client.audio.transcriptions.create.await_count == 2
    events = [call.args[0] for call in cast(AsyncMock, handler.write_event).await_args_list]
    assert [event.type for event in events] == ["error"]
    assert events[0].data["text"] == STT_FAILED_ERROR
    assert handler._wav_buffer is None
