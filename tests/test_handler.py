import asyncio
import base64
import builtins
import contextlib
import gc
import io
import json
import logging
import socket
import struct
import sys
import wave
from collections.abc import AsyncIterator, Callable
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import pytest_asyncio
from openai import omit
from pcm_reference import assert_samples_close, interleave_pcm, reference_resample, sine_samples
from wav_helpers import float_wav, fmt_chunk_body, riff_chunk, wav_file
from websockets.exceptions import ConnectionClosedOK
from websockets.frames import Close
from wyoming.asr import Transcript, TranscriptChunk
from wyoming.audio import AudioChunk
from wyoming.error import Error
from wyoming.event import Event, async_read_event, async_write_event
from wyoming.info import Attribution, TtsVoice
from wyoming.ping import Pong
from wyoming.server import AsyncTcpServer
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeVoice

from wyoming_openai.compatibility import (
    OpenAIBackend,
    TtsVoiceModel,
    create_asr_programs,
    create_info,
    create_tts_programs,
    create_tts_voices,
)
from wyoming_openai.concurrency import ConcurrencyLimit
from wyoming_openai.handler import (
    SHUTDOWN_ERROR,
    STT_FAILED_ERROR,
    TTS_CHUNK_SIZE,
    OpenAIEventHandler,
    TtsStreamError,
    _AudioStreamState,
    _has_speakable_content,
    _SynthesisState,
)
from wyoming_openai.utilities import (
    RealtimeAudioConverter,
    decode_pcm_samples,
    downmix_pcm_samples,
    encode_pcm16_samples,
)
from wyoming_openai.wav import WAVE_FORMAT_PCM, TtsAudioFormat, WavFramer, WavHeader, parse_wav_header


def _pcm_wav_with_trailing_metadata(pcm: bytes) -> bytes:
    """Build a valid PCM WAV with LIST/INFO and C2PA chunks after data."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(pcm)

    wav_data = wav_buffer.getvalue()
    metadata = riff_chunk(b"LIST", b"INFOISFT\x08\x00\x00\x00CrispASR") + riff_chunk(b"C2PA", b"manifest")
    complete_wav = wav_data + metadata
    return complete_wav[:4] + struct.pack("<I", len(complete_wav) - 8) + complete_wav[8:]


def _pcm_wav_with_unbounded_sizes(pcm: bytes) -> bytes:
    """Build a PCM WAV with unbounded RIFF and data chunk size fields."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(pcm)

    wav_data = wav_buffer.getvalue()
    return wav_data[:4] + struct.pack("<I", 0xFFFFFFFF) + wav_data[8:40] + struct.pack("<I", 0xFFFFFFFF) + wav_data[44:]


async def _play_buffered_sentence(handler, audio_data, is_first_chunk, start_timestamp, audio_format, text=""):
    """Play the buffered response of a sentence on a stream in the given state.

    Returns the timestamp after it, or None if playback failed.
    """
    handler._synthesis.audio_started = not is_first_chunk
    handler._synthesis.timestamp = start_timestamp
    if not await handler._stream_audio_to_wyoming(audio_data, audio_format, text):
        return None
    return handler._synthesis.timestamp


async def _stream_whole_request(handler, voice, text, send_audio_start=True, start_timestamp=0):
    """Synthesize text as one request on a stream in the given state.

    Returns the timestamp after it, or None if synthesis failed.
    """
    handler._synthesis.audio_started = not send_audio_start
    handler._synthesis.timestamp = start_timestamp
    if not await handler._stream_tts_audio(voice, text):
        return None
    return handler._synthesis.timestamp


@pytest.fixture
def dummy_info():
    class DummyModel:
        def __init__(self, name, languages=None):
            self.name = name
            self.languages = languages or ["en"]

    class DummyVoice:
        def __init__(self, name, languages=None, model_name=None):
            self.name = name
            self.languages = languages or ["en"]
            self.model_name = model_name or name
            self.backend_voice_name = name

    class DummyProgram:
        def __init__(self, models=None, voices=None, supports_transcript_streaming=False):
            self.models = models or []
            self.voices = voices or []
            self.supports_transcript_streaming = supports_transcript_streaming

    class DummyInfo:
        def __init__(self):
            self.asr = [DummyProgram([DummyModel("m1")])]
            self.tts = [DummyProgram(voices=[DummyVoice("voice1", ["en"], "m1")])]

        def event(self):
            return "event"

    return DummyInfo()


@pytest.fixture
def dummy_clients():
    stt_client = MagicMock()
    stt_client.close = AsyncMock()
    tts_client = MagicMock()
    tts_client.close = AsyncMock()
    return stt_client, tts_client


@pytest.fixture
def dummy_reader_writer():
    reader = MagicMock(name="reader")
    writer = MagicMock(name="writer")
    writer.wait_closed = AsyncMock()
    writer.transport = Mock()
    return reader, writer


@pytest.fixture
def handler(dummy_info, dummy_clients, dummy_reader_writer):
    stt_client, tts_client = dummy_clients
    reader, writer = dummy_reader_writer
    return OpenAIEventHandler(
        reader,
        writer,
        info=dummy_info,
        stt_client=stt_client,
        tts_client=tts_client,
    )


def _asr_model(name: str, languages: list[str]) -> Mock:
    """An ASR model double; Mock(name=...) would only name the mock, not set `.name`."""
    model = Mock(languages=languages)
    model.name = name
    return model


async def _stream_audio(handler, rate=16000, transcribe=None):
    """Send one audio stream, with a transcribe in the middle of it when given."""
    await handler.handle_event(Event(type="audio-start", data={"rate": rate, "width": 2, "channels": 1}))
    await handler.handle_event(
        Event(type="audio-chunk", data={"rate": rate, "width": 2, "channels": 1}, payload=b"\x00\x01" * 10)
    )
    if transcribe is not None:
        assert await handler.handle_event(Event(type="transcribe", data=transcribe))
    await handler.handle_event(Event(type="audio-stop"))


def _selected_realtime_stream(handler):
    """The Realtime audio stream that the handler's current selection would start."""
    return _AudioStreamState(
        model=handler._current_asr_model, language=handler._current_language, realtime=True, streaming=False
    )


def _realtime_stream(rate=24000, width=2, channels=1):
    """A Realtime audio stream with its converter, as audio-start creates it."""
    audio_stream = _AudioStreamState(model=None, language=None, realtime=True, streaming=False)
    audio_stream.converter = RealtimeAudioConverter(rate, width, channels)
    return audio_stream


def _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients, name="gpt-realtime-whisper"):
    """Select a Realtime STT model and return the handler's mocked STT client."""
    stt_client, _ = mock_clients
    mock_info.asr[0].models[0].name = name
    enhanced_handler._stt_realtime_models = {name}
    return stt_client


class _FakeTranscriptionStream:
    """Base of the stand-ins for openai.AsyncStream, which the handler closes as a context manager."""

    closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.closed = True


class _FakeRealtimeServerEvent:
    def __init__(self, event_type, **kwargs):
        self.type = event_type
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeRealtimeSession:
    def __init__(self):
        self.update = AsyncMock()


class _FakeRealtimeInputAudioBuffer:
    def __init__(self, connection):
        self._connection = connection
        self.appended_audio = []
        self.committed = False

    async def append(self, *, audio):
        self.appended_audio.append(audio)

    async def commit(self):
        self.committed = True
        for event in self._connection.commit_events:
            await self._connection.events.put(event)


class _FakeRealtimeResponse:
    def __init__(self, connection):
        self._connection = connection
        self.created = []

    async def create(self, *, response):
        self.created.append(response)
        for event in self._connection.commit_events:
            await self._connection.events.put(event)


class _FakeRealtimeConnection:
    def __init__(self, commit_events):
        self.commit_events = commit_events
        self.events = asyncio.Queue()
        self.session = _FakeRealtimeSession()
        self.input_audio_buffer = _FakeRealtimeInputAudioBuffer(self)
        self.response = _FakeRealtimeResponse(self)
        self.closed = False
        self._connection: Mock | None = None  # The websocket an SDK connection keeps

    async def __aiter__(self):
        # An async generator, as in the SDK, so cancelling a pending event closes the iterator
        while (event := await self.events.get()) is not None:
            yield event

    async def close(self):
        self.closed = True
        await self.events.put(None)


class _FakeRealtimeConnectionManager:
    def __init__(self, connection):
        self.connection = connection
        self.entered = False
        self.exited = False

    async def enter(self):
        self.entered = True
        return self.connection

    async def __aexit__(self, exc_type, exc, tb):
        self.exited = True
        await self.connection.close()


@pytest.mark.asyncio
async def test_init_and_stop(dummy_info, dummy_clients, dummy_reader_writer, handler):
    stt_client, tts_client = dummy_clients
    await handler.stop()
    stt_client.close.assert_not_called()
    tts_client.close.assert_not_called()


@pytest.mark.asyncio
async def test_shared_clients_remain_usable_after_handler_stop(dummy_info, dummy_reader_writer):
    stt_client = AsyncMock()
    tts_client = AsyncMock()

    stt_client.close = AsyncMock()
    tts_client.close = AsyncMock()

    mock_transcription = Mock()
    mock_transcription.text = "Shared client transcription"
    stt_client.audio.transcriptions.create = AsyncMock(return_value=mock_transcription)

    def new_handler():
        handler = OpenAIEventHandler(
            dummy_reader_writer[0],
            dummy_reader_writer[1],
            info=dummy_info,
            stt_client=stt_client,
            tts_client=tts_client,
        )
        handler.write_event = AsyncMock()
        return handler

    await new_handler().stop()

    # The handler of the next connection transcribes with the clients the stopped one shared
    handler = new_handler()
    transcribe_event = Event(type="transcribe", data={"language": "en", "name": "m1"})
    assert await handler.handle_event(transcribe_event) is True

    await handler.handle_event(Event(type="audio-start", data={"rate": 16000, "width": 2, "channels": 1}))
    await handler.handle_event(
        Event(type="audio-chunk", data={"rate": 16000, "width": 2, "channels": 1}, payload=b"\x00\x01" * 50)
    )

    with patch("wyoming_openai.handler.isinstance") as mock_isinstance:

        def isinstance_side_effect(obj, class_or_tuple):
            if obj is mock_transcription:
                from openai.types.audio.transcription_create_response import TranscriptionCreateResponse

                return class_or_tuple is TranscriptionCreateResponse
            return builtins.isinstance(obj, class_or_tuple)

        mock_isinstance.side_effect = isinstance_side_effect
        await handler.handle_event(Event(type="audio-stop"))

    stt_client.audio.transcriptions.create.assert_called_once()
    stt_client.close.assert_not_called()
    tts_client.close.assert_not_called()


def test_get_asr_model(handler):
    model = handler._get_asr_model("m1")
    assert model is not None
    assert model.name == "m1"


def test_get_voice(handler):
    voice = handler._get_voice("voice1")
    assert voice is not None
    assert voice.name == "voice1"


def test_is_asr_model_streaming(dummy_info, handler):
    dummy_info.asr[0].supports_transcript_streaming = True
    assert handler._is_asr_model_streaming("m1") is True


def test_is_asr_language_supported(handler):
    model = handler._get_asr_model("m1")
    assert handler._is_asr_language_supported("en", model)


@pytest.mark.parametrize(
    ("advertised", "requested", "expected"),
    [
        (["en"], "en-US", True),
        (["en"], "en_US", True),
        (["en"], "EN", True),
        (["en-US"], "en", True),
        (["en-US"], "en-GB", False),
        (["en"], "fr", False),
        (["en"], None, True),
        ([], "fr", True),
    ],
)
def test_asr_and_tts_language_checks_agree(handler, advertised, requested, expected):
    """WYOMING_LANGUAGES reads the same for a transcribe as for a synthesize."""
    model = handler._get_asr_model("m1")
    voice = handler._get_voice("voice1")
    model.languages = advertised
    voice.languages = advertised
    assert handler._is_asr_language_supported(requested, model) is expected
    assert handler._validate_tts_language(requested, voice) is expected


def test_validate_tts_language(handler):
    voice = handler._get_voice("voice1")
    assert handler._validate_tts_language("en", voice)


def test_init_rejects_unsupported_stt_response_format(dummy_info, dummy_clients, dummy_reader_writer):
    stt_client, tts_client = dummy_clients
    reader, writer = dummy_reader_writer

    with pytest.raises(ValueError, match="STT extra_body response_format must be one of 'json'"):
        OpenAIEventHandler(
            reader,
            writer,
            info=dummy_info,
            stt_client=stt_client,
            tts_client=tts_client,
            stt_extra_body={"response_format": "text"},
        )


def test_init_rejects_non_string_stt_response_format(dummy_info, dummy_clients, dummy_reader_writer):
    stt_client, tts_client = dummy_clients
    reader, writer = dummy_reader_writer

    with pytest.raises(ValueError, match="got \\['json'\\]"):
        OpenAIEventHandler(
            reader,
            writer,
            info=dummy_info,
            stt_client=stt_client,
            tts_client=tts_client,
            stt_extra_body={"response_format": ["json"]},
        )


def test_init_rejects_null_stt_response_format(dummy_info, dummy_clients, dummy_reader_writer):
    stt_client, tts_client = dummy_clients
    reader, writer = dummy_reader_writer

    with pytest.raises(ValueError, match="got None"):
        OpenAIEventHandler(
            reader,
            writer,
            info=dummy_info,
            stt_client=stt_client,
            tts_client=tts_client,
            stt_extra_body={"response_format": None},
        )


def test_init_rejects_non_boolean_stt_stream(dummy_info, dummy_clients, dummy_reader_writer):
    stt_client, tts_client = dummy_clients
    reader, writer = dummy_reader_writer

    with pytest.raises(ValueError, match="STT extra_body stream must be a boolean"):
        OpenAIEventHandler(
            reader,
            writer,
            info=dummy_info,
            stt_client=stt_client,
            tts_client=tts_client,
            stt_extra_body={"stream": "yes"},
        )


def test_init_allows_unused_stt_response_format_without_asr(dummy_info, dummy_clients, dummy_reader_writer):
    stt_client, tts_client = dummy_clients
    reader, writer = dummy_reader_writer
    dummy_info.asr = []

    OpenAIEventHandler(
        reader,
        writer,
        info=dummy_info,
        stt_client=stt_client,
        tts_client=tts_client,
        stt_extra_body={"response_format": "text"},
    )


def test_init_rejects_undecodable_tts_response_format(dummy_info, dummy_clients, dummy_reader_writer):
    stt_client, tts_client = dummy_clients
    reader, writer = dummy_reader_writer

    with pytest.raises(ValueError, match="TTS extra_body response_format must be one of 'pcm', 'wav'"):
        OpenAIEventHandler(
            reader,
            writer,
            info=dummy_info,
            stt_client=stt_client,
            tts_client=tts_client,
            tts_extra_body={"response_format": "mp3"},
        )


def test_init_rejects_non_string_tts_response_format(dummy_info, dummy_clients, dummy_reader_writer):
    stt_client, tts_client = dummy_clients
    reader, writer = dummy_reader_writer

    with pytest.raises(ValueError, match="got \\['wav'\\]"):
        OpenAIEventHandler(
            reader,
            writer,
            info=dummy_info,
            stt_client=stt_client,
            tts_client=tts_client,
            tts_extra_body={"response_format": ["wav"]},
        )


def test_init_rejects_null_tts_response_format(dummy_info, dummy_clients, dummy_reader_writer):
    stt_client, tts_client = dummy_clients
    reader, writer = dummy_reader_writer

    with pytest.raises(ValueError, match="got None"):
        OpenAIEventHandler(
            reader,
            writer,
            info=dummy_info,
            stt_client=stt_client,
            tts_client=tts_client,
            tts_extra_body={"response_format": None},
        )


def test_init_rejects_tts_stream_override(dummy_info, dummy_clients, dummy_reader_writer):
    stt_client, tts_client = dummy_clients
    reader, writer = dummy_reader_writer

    with pytest.raises(ValueError, match="TTS extra_body does not support overriding 'stream'"):
        OpenAIEventHandler(
            reader,
            writer,
            info=dummy_info,
            stt_client=stt_client,
            tts_client=tts_client,
            tts_extra_body={"stream": True},
        )


def test_init_rejects_tts_stream_format_override(dummy_info, dummy_clients, dummy_reader_writer):
    stt_client, tts_client = dummy_clients
    reader, writer = dummy_reader_writer

    with pytest.raises(ValueError, match="TTS extra_body does not support overriding 'stream_format'"):
        OpenAIEventHandler(
            reader,
            writer,
            info=dummy_info,
            stt_client=stt_client,
            tts_client=tts_client,
            tts_extra_body={"stream_format": "sse"},
        )


def test_init_allows_pcm_tts_response_format(dummy_info, dummy_clients, dummy_reader_writer):
    stt_client, tts_client = dummy_clients
    reader, writer = dummy_reader_writer

    OpenAIEventHandler(
        reader,
        writer,
        info=dummy_info,
        stt_client=stt_client,
        tts_client=tts_client,
        tts_extra_body={"response_format": "pcm"},
    )


def test_init_allows_unused_tts_response_format_without_tts(dummy_info, dummy_clients, dummy_reader_writer):
    stt_client, tts_client = dummy_clients
    reader, writer = dummy_reader_writer
    dummy_info.tts = []

    OpenAIEventHandler(
        reader,
        writer,
        info=dummy_info,
        stt_client=stt_client,
        tts_client=tts_client,
        tts_extra_body={"response_format": "mp3"},
    )


@pytest.mark.asyncio
async def test_streaming_chunk_failure_aborts(handler):
    handler._wyoming_info.tts[0].supports_synthesize_streaming = True
    handler.write_event = AsyncMock()

    start_voice = SynthesizeVoice(name="voice1", language="en")
    await handler.handle_event(SynthesizeStart(voice=start_voice).event())

    failing_stream = AsyncMock(side_effect=TtsStreamError("Forced failure", "Failure chunk.", "voice1"))
    with patch.object(handler, "_get_tts_audio_stream", failing_stream):
        result = await handler.handle_event(SynthesizeChunk(text="Failure chunk. Another sentence.").event())

    assert result is False
    event_types = [call.args[0].type for call in handler.write_event.call_args_list]
    assert "synthesize-stopped" in event_types
    assert handler._synthesis.active is False
    assert handler._allow_streaming_task_id is None


@pytest.mark.asyncio
async def test_buffered_synthesis_failure_aborts(handler):
    """Test that buffered synthesis failures (parallel tasks) properly abort synthesis."""
    handler._wyoming_info.tts[0].supports_synthesize_streaming = True
    handler.write_event = AsyncMock()

    start_voice = SynthesizeVoice(name="voice1", language="en")
    await handler.handle_event(SynthesizeStart(voice=start_voice).event())

    # Mock _get_tts_audio_stream to fail for buffered synthesis
    # (task_id exists but not currently allowed to stream)
    from wyoming_openai.handler import TtsStreamResult

    call_count = 0

    async def mock_buffered_failure(text, voice, task_id=None):
        nonlocal call_count
        call_count += 1
        # First task succeeds (but buffered - will wait to stream)
        # Second task fails to simulate a partial failure in parallel processing
        if call_count == 1:
            return TtsStreamResult(streamed=False, audio=b"\x00\x01" * 1000)
        raise TtsStreamError("Buffered synthesis failed", text[:50], voice.name)

    with patch.object(handler, "_get_tts_audio_stream", side_effect=mock_buffered_failure):
        # Send chunk with three sentences. Handler processes all but last,
        # so first two will be processed (first succeeds, second fails)
        result = await handler.handle_event(
            SynthesizeChunk(text="First sentence. Second sentence fails. Third sentence.").event()
        )

    # The chunk processing should fail when the second sentence fails
    assert result is False
    event_types = [call.args[0].type for call in handler.write_event.call_args_list]
    assert "synthesize-stopped" in event_types
    assert handler._synthesis.active is False
    assert handler._synthesis.buffer == []
    assert handler._synthesis.text_accumulator == ""


@pytest.mark.asyncio
async def test_sentence_playback_stops_once_the_synthesis_has_failed(handler):
    """A shutdown reported as a sentence finishes is not followed by the sentences already buffered."""
    handler._wyoming_info.tts[0].supports_synthesize_streaming = True
    handler.write_event = AsyncMock()
    await handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="voice1", language="en")).event())

    from wyoming_openai.handler import TtsStreamResult

    async def finish_as_the_server_stops(text, voice, task_id=None):
        if text.startswith("First"):
            # stop() reports the shutdown while this sentence finishes, when it has no request left to cancel
            handler._synthesis.failed = True
            return TtsStreamResult(streamed=True)
        return TtsStreamResult(streamed=False, audio=b"\x00\x01" * 1000)

    play = AsyncMock(return_value=True)
    with (
        patch.object(handler, "_get_tts_audio_stream", side_effect=finish_as_the_server_stops),
        patch.object(handler, "_stream_audio_to_wyoming", play),
    ):
        result = await handler.handle_event(
            SynthesizeChunk(text="First sentence. Second sentence. Third sentence.").event()
        )

    assert result is False
    play.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_synthesis_stopped_by_the_server_as_it_finishes_is_not_completed_after_the_error(
    enhanced_handler, mock_info, mock_clients, streaming
):
    """Test stop() landing after the last request finished does not let the synthesis send its stream end after."""
    from wyoming_openai.handler import SHUTDOWN_ERROR

    _, tts_client = mock_clients
    mock_info.tts[0].supports_synthesize_streaming = streaming
    enhanced_handler._tts_extra_body = {"response_format": "pcm"}
    _mock_speech_response(tts_client, [REALTIME_TTS_PCM])
    real_stream = enhanced_handler._stream_tts_audio
    real_get = enhanced_handler._get_tts_audio_stream
    stops = []

    async def stopped_as_it_finishes(*args, **kwargs):
        result = await (real_get if streaming else real_stream)(*args, **kwargs)
        # Runs once this request has finished and before the handler resumes, so it has nothing left to cancel
        stops.append(asyncio.ensure_future(enhanced_handler._stop_sentence_playback(SHUTDOWN_ERROR)))
        return result

    if streaming:
        with patch.object(enhanced_handler, "_get_tts_audio_stream", side_effect=stopped_as_it_finishes):
            assert await enhanced_handler.handle_event(SynthesizeStart().event())
            assert await enhanced_handler.handle_event(SynthesizeChunk(text="Only sentence.").event())
            await enhanced_handler.handle_event(Event(type="synthesize-stop"))
    else:
        with patch.object(enhanced_handler, "_stream_tts_audio", side_effect=stopped_as_it_finishes):
            assert await enhanced_handler.handle_event(Synthesize(text="Only sentence.").event()) is False

    await asyncio.gather(*stops)
    assert len(stops) == 1
    events = _event_types(enhanced_handler)
    after_error = events[events.index("error") + 1 :]
    # Nothing reopens or completes the stream once the client was told
    assert "audio-start" not in after_error
    assert after_error.count("audio-stop") <= (1 if streaming else 0)


@pytest.mark.asyncio
async def test_finishing_a_synthesis_that_stop_already_ended_sends_nothing(enhanced_handler, mock_info):
    """Test the stream end of a synthesis is not sent once stop() closed its stream and told the client."""
    voice = mock_info.tts[0].voices[0]
    # As stop() leaves it when it lands after the playback of the last, buffered sentence
    enhanced_handler._synthesis.failed = True
    enhanced_handler._synthesis.audio_started = False

    assert await enhanced_handler._finish_tts_audio_stream(voice) is False
    enhanced_handler.write_event.assert_not_called()


@pytest.mark.asyncio
async def test_stop_while_the_last_audio_stop_is_written_closes_the_stream_once(
    enhanced_handler, mock_info, mock_clients
):
    """Test stop() during the final audio stop write, to a slow client, does not send a second one."""
    _, tts_client = mock_clients
    mock_info.tts[0].supports_synthesize_streaming = True
    enhanced_handler._tts_extra_body = {"response_format": "pcm"}
    _mock_speech_response(tts_client, [REALTIME_TTS_PCM])
    written = []
    stopping = asyncio.Event()

    async def slow_client(event):
        written.append(event.type)
        if event.type == "audio-stop" and not stopping.is_set():
            stopping.set()
            await asyncio.sleep(0.05)

    enhanced_handler.write_event = AsyncMock(side_effect=slow_client)
    assert await enhanced_handler.handle_event(SynthesizeStart().event())
    assert await enhanced_handler.handle_event(SynthesizeChunk(text="Only sentence.").event())
    finishing = asyncio.create_task(enhanced_handler.handle_event(Event(type="synthesize-stop")))
    await stopping.wait()
    await enhanced_handler.stop()
    await finishing

    assert written.count("audio-stop") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [True, False])
async def test_disconnect_of_a_cancelled_handler_reports_the_synthesis_it_cuts_short(
    enhanced_handler, mock_info, mock_clients, cancelled
):
    """Test a streaming synthesis ended by cancellation, as on Ctrl+C, gets an error the client can still read."""
    _, tts_client = mock_clients
    mock_info.tts[0].supports_synthesize_streaming = True
    enhanced_handler._tts_extra_body = {"response_format": "pcm"}
    _mock_speech_response(tts_client, [REALTIME_TTS_PCM])
    assert await enhanced_handler.handle_event(SynthesizeStart().event())
    assert await enhanced_handler.handle_event(SynthesizeChunk(text="First sentence. Second").event())
    enhanced_handler.write_event.reset_mock()

    async def handler_task():
        if cancelled:
            task = asyncio.current_task()
            assert task is not None
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.sleep(1)
        # run() calls disconnect() as it unwinds, still marked as cancelled
        await enhanced_handler.disconnect()

    await asyncio.create_task(handler_task())

    written = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
    if cancelled:
        assert [event.type for event in written] == ["error", "audio-stop"]
        assert Error.from_event(written[0]).text == SHUTDOWN_ERROR
    else:
        assert [event.type for event in written] == ["audio-stop"]


@pytest.mark.asyncio
async def test_empty_audio_data_aborts(handler):
    """Test that empty audio data from synthesis properly aborts the session."""
    handler._wyoming_info.tts[0].supports_synthesize_streaming = True
    handler.write_event = AsyncMock()

    start_voice = SynthesizeVoice(name="voice1", language="en")
    await handler.handle_event(SynthesizeStart(voice=start_voice).event())

    # Mock _get_tts_audio_stream to return empty audio (buffered mode)
    from wyoming_openai.handler import TtsStreamResult

    async def mock_empty_audio(text, voice, task_id=None):
        # Return result with no audio data
        return TtsStreamResult(streamed=False, audio=b"")

    with patch.object(handler, "_get_tts_audio_stream", side_effect=mock_empty_audio):
        # Send two sentences so first one gets processed (and returns empty audio)
        assert await handler.handle_event(SynthesizeChunk(text="Test sentence. Another one.").event())
        handler.write_event.assert_not_called()
        result = await handler.handle_event(Event(type="synthesize-stop"))

    assert result is False
    event_types = [call.args[0].type for call in handler.write_event.call_args_list]
    assert "synthesize-stopped" in event_types
    assert handler._synthesis.active is False
    # Verify state was fully reset
    assert handler._synthesis.audio_started is False
    assert handler._synthesis.timestamp == 0
    assert handler._synthesis.voice is None
    assert handler._synthesis.language is None


@pytest.mark.asyncio
async def test_multiple_consecutive_chunk_failures(handler):
    """Test that multiple consecutive synthesis failures are handled gracefully."""
    handler._wyoming_info.tts[0].supports_synthesize_streaming = True
    handler.write_event = AsyncMock()

    start_voice = SynthesizeVoice(name="voice1", language="en")
    await handler.handle_event(SynthesizeStart(voice=start_voice).event())

    # Mock to always fail
    failing_stream = AsyncMock(side_effect=TtsStreamError("Persistent failure", "Test chunk", "voice1"))

    with patch.object(handler, "_get_tts_audio_stream", failing_stream):
        # First failure - send two sentences so first gets processed
        result1 = await handler.handle_event(SynthesizeChunk(text="First chunk. Second one.").event())
        assert result1 is False

        # Verify state was reset after first failure
        assert handler._synthesis.active is False

        # Try to start again - should work
        await handler.handle_event(SynthesizeStart(voice=start_voice).event())
        assert handler._synthesis.active is True

        # Second failure
        result2 = await handler.handle_event(SynthesizeChunk(text="Another chunk. And another.").event())
        assert result2 is False

        # Verify state is consistently reset
        assert handler._synthesis.active is False
        assert handler._synthesis.buffer == []
        assert handler._allow_streaming_task_id is None

    # Verify synthesize-stopped was called for each failure
    event_types = [call.args[0].type for call in handler.write_event.call_args_list]
    stopped_count = event_types.count("synthesize-stopped")
    assert stopped_count >= 2, f"Expected at least 2 synthesize-stopped events, got {stopped_count}"


@pytest.fixture
def mock_info():
    """Create a mock Info object with ASR and TTS programs."""
    mock_info = Mock()

    # Mock ASR model
    asr_model = Mock()
    asr_model.name = "whisper-1"
    asr_model.description = "OpenAI Whisper"
    asr_model.languages = ["en", "fr", "es"]

    # Mock ASR program
    asr_program = Mock()
    asr_program.models = [asr_model]
    asr_program.supports_transcript_streaming = False

    # Mock TTS voice
    tts_voice = Mock()
    tts_voice.name = "alloy"
    tts_voice.description = "Alloy voice"
    tts_voice.languages = ["en"]
    tts_voice.model_name = "tts-1"
    tts_voice.backend_voice_name = "alloy"

    # Mock TTS program
    tts_program = Mock()
    tts_program.voices = [tts_voice]

    mock_info.asr = [asr_program]
    mock_info.tts = [tts_program]

    # Mock event method
    mock_info.event = Mock(return_value=Event(type="info"))

    return mock_info


@pytest.fixture
def mock_clients():
    """Create mock STT and TTS clients."""
    stt_client = AsyncMock()
    tts_client = AsyncMock()

    # Mock close methods
    stt_client.close = AsyncMock()
    tts_client.close = AsyncMock()

    return stt_client, tts_client


@pytest.fixture
def enhanced_handler(mock_info, mock_clients, dummy_reader_writer):
    """Create an enhanced OpenAIEventHandler instance with comprehensive mocks."""
    stt_client, tts_client = mock_clients
    reader, writer = dummy_reader_writer

    handler = OpenAIEventHandler(
        reader,
        writer,
        info=mock_info,
        stt_client=stt_client,
        tts_client=tts_client,
        stt_temperature=0.5,
        stt_prompt="Test prompt",
        tts_speed=1.0,
        tts_instructions="Test instructions",
    )

    # Mock write_event as AsyncMock
    handler.write_event = AsyncMock()

    return handler


class TestOpenAIEventHandlerComprehensive:
    """Comprehensive tests for the OpenAIEventHandler class."""

    @pytest.mark.asyncio
    async def test_handle_describe_event(self, enhanced_handler, mock_info):
        """Test handling of Describe event."""
        event = Event(type="describe")

        result = await enhanced_handler.handle_event(event)

        assert result is True
        enhanced_handler.write_event.assert_called_once()
        # Check that the event written was the info event
        written_event = enhanced_handler.write_event.call_args[0][0]
        assert written_event.type == "info"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("data", [{"text": "are you there"}, {"text": None}, {}])
    async def test_ping_is_answered_with_a_pong_carrying_its_text(self, enhanced_handler, data):
        """A ping gets a pong with the same text, and leaves the connection open."""
        result = await enhanced_handler.handle_event(Event(type="ping", data=data))

        assert result is True
        enhanced_handler.write_event.assert_called_once()
        written_event = enhanced_handler.write_event.call_args[0][0]
        assert Pong.is_type(written_event.type)
        assert Pong.from_event(written_event).text == data.get("text")

    @pytest.mark.asyncio
    async def test_ping_during_a_recording_leaves_the_transcription_intact(self, enhanced_handler, mock_clients):
        """A ping that arrives while audio is streaming is answered, and the recording is still transcribed."""
        from openai.types.audio.transcription import Transcription

        stt_client, _ = mock_clients
        sent_frames = []

        async def create(**kwargs):
            with wave.open(io.BytesIO(kwargs["file"].getvalue()), "rb") as wav_file:
                sent_frames.append(wav_file.getnframes())
            return Transcription(text="Hello")

        stt_client.audio.transcriptions.create = AsyncMock(side_effect=create)
        audio_format = {"rate": 16000, "width": 2, "channels": 1}

        await enhanced_handler.handle_event(Event(type="audio-start", data=audio_format))
        await enhanced_handler.handle_event(Event(type="audio-chunk", data=audio_format, payload=b"\x00\x01" * 10))
        assert await enhanced_handler.handle_event(Event(type="ping", data={"text": "x"}))
        await enhanced_handler.handle_event(Event(type="audio-chunk", data=audio_format, payload=b"\x00\x01" * 10))
        assert await enhanced_handler.handle_event(Event(type="audio-stop"))

        assert [call.args[0].type for call in enhanced_handler.write_event.call_args_list] == [
            "pong", "transcript-start", "transcript", "transcript-stop"
        ]
        # Both chunks were recorded, the one after the ping too
        assert sent_frames == [20]

    @pytest.mark.asyncio
    async def test_handle_audio_start_event(self, enhanced_handler):
        """Test handling of AudioStart event."""
        event = Event(type="audio-start", data={"rate": 16000, "width": 2, "channels": 1})

        result = await enhanced_handler.handle_event(event)

        assert result is True
        assert enhanced_handler._is_recording is True
        assert enhanced_handler._wav_buffer is not None
        assert enhanced_handler._wav_write_buffer is not None

    @pytest.mark.asyncio
    async def test_handle_audio_chunk_event(self, enhanced_handler):
        """Test handling of AudioChunk event."""
        # First start recording
        start_event = Event(type="audio-start", data={"rate": 16000, "width": 2, "channels": 1})
        await enhanced_handler.handle_event(start_event)

        # Send audio chunk
        audio_data = b"\x00\x01" * 100
        chunk_event = Event(type="audio-chunk", data={"rate": 16000, "width": 2, "channels": 1}, payload=audio_data)

        result = await enhanced_handler.handle_event(chunk_event)

        assert result is True
        # Verify audio was written to buffer
        assert enhanced_handler._wav_buffer.tell() > 0

    @pytest.mark.asyncio
    async def test_handle_audio_stop_event(self, enhanced_handler, mock_clients):
        """Test handling of AudioStop event."""
        from openai.types.audio.transcription import Transcription

        stt_client, _ = mock_clients
        stt_client.audio.transcriptions.create = AsyncMock(return_value=Transcription(text="Hello"))
        # First start recording
        start_event = Event(type="audio-start")
        await enhanced_handler.handle_event(start_event)

        # Stop recording
        stop_event = Event(type="audio-stop")
        result = await enhanced_handler.handle_event(stop_event)

        assert result is True
        assert enhanced_handler._is_recording is False
        assert enhanced_handler._wav_write_buffer is None

    @pytest.mark.asyncio
    async def test_handle_transcribe_event(self, enhanced_handler, mock_clients):
        """Test handling of Transcribe event."""
        stt_client, _ = mock_clients

        # Mock transcription response
        mock_transcription = Mock()
        mock_transcription.text = "Test transcription"
        stt_client.audio.transcriptions.create = AsyncMock(return_value=mock_transcription)

        # First send the transcribe event to set the model
        transcribe_event = Event(type="transcribe", data={"language": "en", "name": "whisper-1"})
        result = await enhanced_handler.handle_event(transcribe_event)
        assert result is True

        # Now record some audio
        start_event = Event(type="audio-start", data={"rate": 16000, "width": 2, "channels": 1})
        await enhanced_handler.handle_event(start_event)

        # Add audio data
        chunk_event = Event(
            type="audio-chunk", data={"rate": 16000, "width": 2, "channels": 1}, payload=b"\x00\x01" * 1000
        )
        await enhanced_handler.handle_event(chunk_event)

        # Stop recording - this triggers transcription
        # Patch the isinstance check in the handler to accept our mock
        with patch("wyoming_openai.handler.isinstance") as mock_isinstance:

            def isinstance_side_effect(obj, class_or_tuple):
                if obj is mock_transcription:
                    from openai.types.audio.transcription_create_response import TranscriptionCreateResponse

                    return class_or_tuple is TranscriptionCreateResponse
                return builtins.isinstance(obj, class_or_tuple)

            mock_isinstance.side_effect = isinstance_side_effect

            stop_event = Event(type="audio-stop")
            await enhanced_handler.handle_event(stop_event)

        # Verify transcription was called
        stt_client.audio.transcriptions.create.assert_called_once()
        call_args = stt_client.audio.transcriptions.create.call_args[1]
        assert call_args["language"] == "en"

        # Find the Transcript event in the write_event calls
        transcript_found = False
        for call in enhanced_handler.write_event.call_args_list:
            event = call[0][0]
            if Transcript.is_type(event.type):
                transcript_found = True
                transcript = Transcript.from_event(event)
                assert transcript.text == "Test transcription"
                break
        assert transcript_found

    @pytest.mark.asyncio
    @pytest.mark.parametrize("late_model", [None, "gpt-realtime-whisper"])
    async def test_http_audio_start_uses_default_model_despite_late_transcribe(
        self, enhanced_handler, mock_info, mock_clients, late_model
    ):
        """An optional/late Transcribe cannot move an HTTP recording to Realtime."""
        from openai.types.audio.transcription import Transcription

        stt_client, _ = mock_clients
        mock_info.asr[0].models.append(_asr_model("gpt-realtime-whisper", ["en"]))
        enhanced_handler._stt_realtime_models = {"gpt-realtime-whisper"}
        stt_client.audio.transcriptions.create = AsyncMock(return_value=Transcription(text="Hello"))

        await _stream_audio(enhanced_handler, transcribe={"name": late_model} if late_model else None)

        assert stt_client.audio.transcriptions.create.call_args.kwargs["model"] == "whisper-1"
        stt_client.realtime.connect.assert_not_called()
        assert [call.args[0].type for call in enhanced_handler.write_event.call_args_list] == [
            "transcript-start", "transcript", "transcript-stop"
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("late_model", [None, "whisper-1"])
    async def test_realtime_audio_start_uses_default_model_despite_late_transcribe(
        self, enhanced_handler, mock_info, mock_clients, late_model
    ):
        """An optional/late Transcribe cannot move a Realtime recording to HTTP."""
        stt_client, _ = mock_clients
        mock_info.asr[0].models[0].name = "gpt-realtime-whisper"
        mock_info.asr[0].models.append(_asr_model("whisper-1", ["en"]))
        enhanced_handler._stt_realtime_models = {"gpt-realtime-whisper"}
        connection = _FakeRealtimeConnection(
            [_FakeRealtimeServerEvent("conversation.item.input_audio_transcription.completed", transcript="Hello")]
        )
        stt_client.realtime.connect = Mock(return_value=_FakeRealtimeConnectionManager(connection))
        stt_client.audio.transcriptions.create = AsyncMock()

        await _stream_audio(enhanced_handler, rate=24000, transcribe={"name": late_model} if late_model else None)

        assert connection.input_audio_buffer.committed is True
        assert connection.session.update.call_args.kwargs["session"]["audio"]["input"]["transcription"]["model"] == (
            "gpt-realtime-whisper"
        )
        stt_client.audio.transcriptions.create.assert_not_called()
        assert [call.args[0].type for call in enhanced_handler.write_event.call_args_list] == [
            "transcript-start", "transcript", "transcript-stop"
        ]

    @pytest.mark.asyncio
    async def test_late_transcribe_switches_an_http_recording_to_another_http_model(
        self, enhanced_handler, mock_info, mock_clients
    ):
        """A recording sent at audio-stop takes the model and language of a transcribe that arrives during it."""
        from openai.types.audio.transcription import Transcription

        stt_client, _ = mock_clients
        mock_info.asr[0].models.append(_asr_model("gpt-transcribe", ["en", "fr"]))
        stt_client.audio.transcriptions.create = AsyncMock(return_value=Transcription(text="Bonjour"))

        await _stream_audio(enhanced_handler, transcribe={"name": "gpt-transcribe", "language": "fr"})

        kwargs = stt_client.audio.transcriptions.create.call_args.kwargs
        assert (kwargs["model"], kwargs["language"]) == ("gpt-transcribe", "fr")

    @pytest.mark.asyncio
    async def test_late_transcribe_for_the_other_transport_is_reported_and_kept_for_the_next_stream(
        self, enhanced_handler, mock_info, mock_clients, caplog
    ):
        """A transcribe that cannot move the recording to Realtime is logged and selects the next stream."""
        from openai.types.audio.transcription import Transcription

        stt_client, _ = mock_clients
        mock_info.asr[0].models.append(_asr_model("gpt-realtime-whisper", ["en"]))
        enhanced_handler._stt_realtime_models = {"gpt-realtime-whisper"}
        stt_client.audio.transcriptions.create = AsyncMock(return_value=Transcription(text="Hello"))
        connection = _FakeRealtimeConnection(
            [_FakeRealtimeServerEvent("conversation.item.input_audio_transcription.completed", transcript="Hi")]
        )
        stt_client.realtime.connect = Mock(return_value=_FakeRealtimeConnectionManager(connection))

        with caplog.at_level(logging.WARNING, logger="wyoming_openai.handler"):
            await _stream_audio(enhanced_handler, transcribe={"name": "gpt-realtime-whisper"})

        assert stt_client.audio.transcriptions.create.call_args.kwargs["model"] == "whisper-1"
        assert "keeps model whisper-1" in caplog.text
        stt_client.realtime.connect.assert_not_called()

        await _stream_audio(enhanced_handler, rate=24000)

        session = connection.session.update.call_args.kwargs["session"]
        assert session["audio"]["input"]["transcription"]["model"] == "gpt-realtime-whisper"
        stt_client.audio.transcriptions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_late_transcribe_language_for_a_realtime_session_is_reported(
        self, enhanced_handler, mock_info, mock_clients, caplog
    ):
        """An open Realtime session keeps its language, and that is logged rather than silent."""
        stt_client, _ = mock_clients
        mock_info.asr[0].models[0].name = "gpt-realtime-whisper"
        enhanced_handler._stt_realtime_models = {"gpt-realtime-whisper"}
        connection = _FakeRealtimeConnection(
            [_FakeRealtimeServerEvent("conversation.item.input_audio_transcription.completed", transcript="Salut")]
        )
        stt_client.realtime.connect = Mock(return_value=_FakeRealtimeConnectionManager(connection))

        assert await enhanced_handler.handle_event(
            Event(type="transcribe", data={"name": "gpt-realtime-whisper", "language": "en"})
        )
        with caplog.at_level(logging.WARNING, logger="wyoming_openai.handler"):
            await _stream_audio(
                enhanced_handler, rate=24000, transcribe={"name": "gpt-realtime-whisper", "language": "fr"}
            )

        assert "language fr" in caplog.text and "language en" in caplog.text
        session = connection.session.update.call_args.kwargs["session"]
        assert session["audio"]["input"]["transcription"]["language"] == "en"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("realtime", [False, True])
    @pytest.mark.parametrize(
        ("advertised", "requested", "sent"),
        [(["en"], "en-US", "en"), (["en"], "EN_us", "en"), (["en-US"], "en", "en-US"), (["en"], "en", "en")],
    )
    async def test_transcribe_language_is_sent_as_advertised(
        self, enhanced_handler, mock_info, mock_clients, realtime, advertised, requested, sent
    ):
        """A tag that only matched WYOMING_LANGUAGES regionally reaches the backend in the advertised spelling."""
        stt_client, _ = mock_clients
        mock_info.asr[0].models[0].languages = advertised
        model_name = mock_info.asr[0].models[0].name
        connection = None
        if realtime:
            model_name = "gpt-realtime-whisper"
            _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients, name=model_name)
            connection = _FakeRealtimeConnection(
                [_FakeRealtimeServerEvent("conversation.item.input_audio_transcription.completed", transcript="Hi")]
            )
            stt_client.realtime.connect = Mock(return_value=_FakeRealtimeConnectionManager(connection))

        assert await enhanced_handler.handle_event(
            Event(type="transcribe", data={"name": model_name, "language": requested})
        )
        await _stream_audio(enhanced_handler, rate=24000 if realtime else 16000)

        if connection is not None:
            session = connection.session.update.call_args.kwargs["session"]
            assert session["audio"]["input"]["transcription"]["language"] == sent
        else:
            assert stt_client.audio.transcriptions.create.call_args.kwargs["language"] == sent
        await enhanced_handler._drain_background_tasks()

    @pytest.mark.asyncio
    async def test_repeated_transcribe_during_a_realtime_session_is_not_reported(
        self, enhanced_handler, mock_info, mock_clients, caplog
    ):
        """A transcribe that asks for what the session already uses is not a warning."""
        stt_client, _ = mock_clients
        mock_info.asr[0].models[0].name = "gpt-realtime-whisper"
        enhanced_handler._stt_realtime_models = {"gpt-realtime-whisper"}
        connection = _FakeRealtimeConnection(
            [_FakeRealtimeServerEvent("conversation.item.input_audio_transcription.completed", transcript="Hi")]
        )
        stt_client.realtime.connect = Mock(return_value=_FakeRealtimeConnectionManager(connection))

        with caplog.at_level(logging.WARNING, logger="wyoming_openai.handler"):
            await _stream_audio(enhanced_handler, rate=24000, transcribe={"name": "gpt-realtime-whisper"})

        assert "arrived during" not in caplog.text

    @pytest.mark.asyncio
    @pytest.mark.parametrize("transcribe", [{"name": "unknown-model"}, {"name": "whisper-1", "language": "zh"}])
    async def test_late_transcribe_is_refused_like_one_before_the_recording(
        self, enhanced_handler, mock_clients, transcribe, caplog
    ):
        """A transcribe refused during a recording leaves nothing to transcribe, as one refused before it does."""
        stt_client, _ = mock_clients
        stt_client.audio.transcriptions.create = AsyncMock()
        chunk = Event(type="audio-chunk", data={"rate": 16000, "width": 2, "channels": 1}, payload=b"\x00\x01" * 10)

        await enhanced_handler.handle_event(Event(type="audio-start", data={"rate": 16000, "width": 2, "channels": 1}))
        await enhanced_handler.handle_event(chunk)
        with caplog.at_level(logging.WARNING, logger="wyoming_openai.handler"):
            assert await enhanced_handler.handle_event(Event(type="transcribe", data=transcribe)) is False
        assert "Dropping an audio stream without transcribing it" in caplog.text
        # A live connection is closed by the refusal; a client that kept sending would still get no transcript
        await enhanced_handler.handle_event(chunk)
        await enhanced_handler.handle_event(Event(type="audio-stop"))

        stt_client.audio.transcriptions.create.assert_not_called()
        assert [call.args[0].type for call in enhanced_handler.write_event.call_args_list] == ["error"]

    @pytest.mark.asyncio
    async def test_refused_transcribe_closes_the_realtime_session_it_interrupts(
        self, enhanced_handler, mock_info, mock_clients
    ):
        """A refusal during a Realtime stream closes its session and its transcript, then reports the failure."""
        stt_client, _ = mock_clients
        mock_info.asr[0].models[0].name = "gpt-realtime-whisper"
        enhanced_handler._stt_realtime_models = {"gpt-realtime-whisper"}
        manager = _FakeRealtimeConnectionManager(_FakeRealtimeConnection([]))
        stt_client.realtime.connect = Mock(return_value=manager)

        await enhanced_handler.handle_event(Event(type="audio-start", data={"rate": 24000, "width": 2, "channels": 1}))
        assert await enhanced_handler.handle_event(Event(type="transcribe", data={"name": "unknown-model"})) is False
        await enhanced_handler._drain_background_tasks()

        assert manager.exited is True
        assert enhanced_handler._audio_stream is None
        assert [call.args[0].type for call in enhanced_handler.write_event.call_args_list] == [
            "transcript-start",
            "transcript-stop",
            "error",
        ]

    @pytest.mark.asyncio
    async def test_disconnect_drops_the_recording_in_progress(self, enhanced_handler, mock_clients):
        """A client that disconnects during a recording leaves no recording behind and is sent nothing."""
        stt_client, _ = mock_clients
        stt_client.audio.transcriptions.create = AsyncMock()

        await enhanced_handler.handle_event(Event(type="audio-start", data={"rate": 16000, "width": 2, "channels": 1}))
        await enhanced_handler.handle_event(
            Event(type="audio-chunk", data={"rate": 16000, "width": 2, "channels": 1}, payload=b"\x00\x01" * 10)
        )
        await enhanced_handler.disconnect()

        assert enhanced_handler._audio_stream is None
        assert enhanced_handler._wav_buffer is None
        assert enhanced_handler._is_recording is False
        stt_client.audio.transcriptions.create.assert_not_called()
        enhanced_handler.write_event.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("teardown", ["disconnect", "stop"])
    async def test_teardown_closes_the_realtime_session_without_writing(
        self, enhanced_handler, mock_info, mock_clients, teardown
    ):
        """A connection that ends during a Realtime stream closes the session; only a server stop tells the client."""
        stt_client, _ = mock_clients
        mock_info.asr[0].models[0].name = "gpt-realtime-whisper"
        enhanced_handler._stt_realtime_models = {"gpt-realtime-whisper"}
        manager = _FakeRealtimeConnectionManager(_FakeRealtimeConnection([]))
        stt_client.realtime.connect = Mock(return_value=manager)

        await enhanced_handler.handle_event(Event(type="audio-start", data={"rate": 24000, "width": 2, "channels": 1}))
        enhanced_handler.write_event.reset_mock()
        await getattr(enhanced_handler, teardown)()

        assert manager.exited is True
        written = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
        assert [Error.from_event(event).text for event in written] == ([SHUTDOWN_ERROR] if teardown == "stop" else [])

    @pytest.mark.asyncio
    async def test_stop_cancels_an_upload_in_progress_before_closing_its_recording(
        self, enhanced_handler, mock_clients
    ):
        """A server stopping while a recording is being sent cancels the upload, then closes the file, not before."""
        stt_client, _ = mock_clients
        uploading = asyncio.Event()
        upload = {}

        async def slow_upload(**kwargs):
            upload["file"] = kwargs["file"]
            uploading.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                upload["closed_when_cancelled"] = kwargs["file"].closed
                raise

        stt_client.audio.transcriptions.create = AsyncMock(side_effect=slow_upload)

        request = asyncio.create_task(_stream_audio(enhanced_handler))
        await asyncio.wait_for(uploading.wait(), timeout=1)
        await enhanced_handler.stop()
        await asyncio.wait_for(request, timeout=1)

        assert upload["closed_when_cancelled"] is False
        assert upload["file"].closed
        # The client was told before the writer closed
        written = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
        assert [Error.from_event(event).text for event in written] == [SHUTDOWN_ERROR]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "abandon",
        [
            pytest.param(Event(type="audio-start", data={"rate": 24000, "width": 2, "channels": 1}), id="audio-start"),
            pytest.param(Event(type="transcribe", data={"name": "unknown-model"}), id="refused-transcribe"),
        ],
    )
    async def test_abandoning_a_realtime_stream_that_never_opened_closes_no_transcript(
        self, enhanced_handler, mock_info, mock_clients, abandon
    ):
        """A Realtime session that failed to open sent no transcript-start, so abandoning its stream closes none."""
        stt_client, _ = mock_clients
        mock_info.asr[0].models[0].name = "gpt-realtime-whisper"
        enhanced_handler._stt_realtime_models = {"gpt-realtime-whisper"}
        stt_client.realtime.connect = Mock(side_effect=RuntimeError("invalid api key"))

        await enhanced_handler.handle_event(Event(type="audio-start", data={"rate": 24000, "width": 2, "channels": 1}))
        await enhanced_handler.handle_event(abandon)

        assert stt_client.realtime.connect.call_count == (2 if abandon.type == "audio-start" else 1)
        # Each session that failed to open was reported when it failed, and nothing else was sent
        assert [call.args[0].type for call in enhanced_handler.write_event.call_args_list] == [
            "error"
        ] * stt_client.realtime.connect.call_count

    @pytest.mark.asyncio
    async def test_audio_stop_without_audio_start_sends_nothing(self, enhanced_handler, mock_clients):
        """An audio stop outside a stream neither transcribes nor answers."""
        stt_client, _ = mock_clients
        stt_client.audio.transcriptions.create = AsyncMock()

        await enhanced_handler.handle_event(Event(type="audio-stop"))

        stt_client.audio.transcriptions.create.assert_not_called()
        enhanced_handler.write_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_audio_start_closes_a_realtime_session_that_was_never_stopped(
        self, enhanced_handler, mock_info, mock_clients, caplog
    ):
        """A new stream on the other transport closes the old session and records its own audio."""
        from openai.types.audio.transcription import Transcription

        stt_client, _ = mock_clients
        mock_info.asr[0].models.append(_asr_model("gpt-realtime-whisper", ["en"]))
        enhanced_handler._stt_realtime_models = {"gpt-realtime-whisper"}
        manager = _FakeRealtimeConnectionManager(_FakeRealtimeConnection([]))
        stt_client.realtime.connect = Mock(return_value=manager)
        stt_client.audio.transcriptions.create = AsyncMock(return_value=Transcription(text="Hello"))

        assert await enhanced_handler.handle_event(Event(type="transcribe", data={"name": "gpt-realtime-whisper"}))
        await enhanced_handler.handle_event(Event(type="audio-start", data={"rate": 24000, "width": 2, "channels": 1}))
        # The client gives up on that stream without an audio stop and selects an HTTP model
        assert await enhanced_handler.handle_event(Event(type="transcribe", data={"name": "whisper-1"}))
        with caplog.at_level(logging.WARNING, logger="wyoming_openai.handler"):
            await _stream_audio(enhanced_handler)

        assert "Dropping an audio stream without transcribing it: audio-start arrived before" in caplog.text
        await enhanced_handler._drain_background_tasks()
        assert manager.exited is True
        assert manager.connection.input_audio_buffer.appended_audio == []
        assert stt_client.audio.transcriptions.create.call_args.kwargs["model"] == "whisper-1"
        # Closed without a Transcript, which the client would take as the result of the new stream
        assert [call.args[0].type for call in enhanced_handler.write_event.call_args_list] == [
            "transcript-start",
            "transcript-stop",
            "transcript-start",
            "transcript",
            "transcript-stop",
        ]

    @pytest.mark.asyncio
    async def test_audio_start_survives_a_stale_session_that_fails_to_close(
        self, enhanced_handler, mock_info, mock_clients, caplog
    ):
        """A Realtime session that errors while closing does not fail the stream that replaces it."""
        from openai.types.audio.transcription import Transcription

        stt_client, _ = mock_clients
        mock_info.asr[0].models.append(_asr_model("gpt-realtime-whisper", ["en"]))
        enhanced_handler._stt_realtime_models = {"gpt-realtime-whisper"}
        class _FailingCloseManager(_FakeRealtimeConnectionManager):
            # On the class, so the failure holds however the handler exits the manager
            async def __aexit__(self, exc_type, exc, tb):
                raise ConnectionResetError("reset while closing")

        manager = _FailingCloseManager(_FakeRealtimeConnection([]))
        stt_client.realtime.connect = Mock(return_value=manager)
        stt_client.audio.transcriptions.create = AsyncMock(return_value=Transcription(text="Hello"))

        assert await enhanced_handler.handle_event(Event(type="transcribe", data={"name": "gpt-realtime-whisper"}))
        await enhanced_handler.handle_event(Event(type="audio-start", data={"rate": 24000, "width": 2, "channels": 1}))
        assert await enhanced_handler.handle_event(Event(type="transcribe", data={"name": "whisper-1"}))
        with caplog.at_level(logging.DEBUG, logger="wyoming_openai.handler"):
            await _stream_audio(enhanced_handler)
            await enhanced_handler._drain_background_tasks()

        # The session is closed in the background, where the failure is logged without touching the stream
        assert "Background task openai_realtime_stt_close failed: reset while closing" in caplog.text
        assert stt_client.audio.transcriptions.create.call_args.kwargs["model"] == "whisper-1"

    @pytest.mark.asyncio
    async def test_chunks_follow_the_stream_whose_realtime_session_failed_to_open(
        self, enhanced_handler, mock_info, mock_clients, caplog
    ):
        """An unstopped recording is dropped, and the next stream's chunks are not written into it."""
        stt_client, _ = mock_clients
        mock_info.asr[0].models.append(_asr_model("gpt-realtime-whisper", ["en"]))
        enhanced_handler._stt_realtime_models = {"gpt-realtime-whisper"}
        stt_client.realtime.connect = Mock(side_effect=RuntimeError("invalid api key"))
        stt_client.audio.transcriptions.create = AsyncMock()

        await enhanced_handler.handle_event(Event(type="audio-start", data={"rate": 16000, "width": 2, "channels": 1}))
        await enhanced_handler.handle_event(
            Event(type="audio-chunk", data={"rate": 16000, "width": 2, "channels": 1}, payload=b"\x00\x01" * 10)
        )
        # The client abandons that recording and starts a Realtime stream whose session cannot be opened
        assert await enhanced_handler.handle_event(Event(type="transcribe", data={"name": "gpt-realtime-whisper"}))
        await enhanced_handler.handle_event(Event(type="audio-start", data={"rate": 24000, "width": 2, "channels": 1}))
        assert enhanced_handler._wav_buffer is None
        assert enhanced_handler._is_recording is False

        with caplog.at_level(logging.WARNING, logger="wyoming_openai.handler"):
            await enhanced_handler.handle_event(
                Event(type="audio-chunk", data={"rate": 24000, "width": 2, "channels": 1}, payload=b"\x00\x01" * 10)
            )
            assert await enhanced_handler.handle_event(Event(type="audio-stop")) is False

        # Reported once when the session failed to open, so its chunks are dropped without a warning each
        assert "audio chunk" not in caplog.text
        assert enhanced_handler._wav_buffer is None
        stt_client.audio.transcriptions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_realtime_transcription_flow(self, enhanced_handler, mock_info, mock_clients):
        """Realtime STT emits snapshots and an authoritative final transcript, then cleans up."""
        stt_client, _ = mock_clients
        mock_info.asr[0].models[0].name = "gpt-realtime-whisper"
        enhanced_handler._stt_realtime_models = {"gpt-realtime-whisper"}

        connection = _FakeRealtimeConnection(
            [
                _FakeRealtimeServerEvent("conversation.item.input_audio_transcription.delta", delta="Hello"),
                _FakeRealtimeServerEvent("conversation.item.input_audio_transcription.delta", delta=""),
                _FakeRealtimeServerEvent("conversation.item.input_audio_transcription.delta", delta=" word"),
                _FakeRealtimeServerEvent(
                    "conversation.item.input_audio_transcription.completed", transcript="Hello world"
                ),
            ]
        )
        manager = _FakeRealtimeConnectionManager(connection)
        stt_client.realtime.connect = Mock(return_value=manager)
        stt_client.audio.transcriptions.create = AsyncMock()

        transcribe_event = Event(type="transcribe", data={"language": "en", "name": "gpt-realtime-whisper"})
        assert await enhanced_handler.handle_event(transcribe_event) is True

        audio_data = b"\x00\x01" * 100
        await enhanced_handler.handle_event(Event(type="audio-start", data={"rate": 24000, "width": 2, "channels": 1}))
        await enhanced_handler.handle_event(
            Event(type="audio-chunk", data={"rate": 24000, "width": 2, "channels": 1}, payload=audio_data)
        )
        await enhanced_handler.handle_event(Event(type="audio-stop"))

        stt_client.realtime.connect.assert_called_once_with(extra_query={"intent": "transcription"})
        stt_client.audio.transcriptions.create.assert_not_called()
        connection.session.update.assert_called_once()
        session = connection.session.update.call_args.kwargs["session"]
        assert session["type"] == "transcription"
        assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
        transcription = session["audio"]["input"]["transcription"]
        assert transcription["model"] == "gpt-realtime-whisper"
        assert transcription["language"] == "en"
        assert session["audio"]["input"]["turn_detection"] is None
        assert base64.b64decode(connection.input_audio_buffer.appended_audio[0]) == audio_data
        assert connection.input_audio_buffer.committed is True

        event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
        assert event_types == [
            "transcript-start",
            "transcript-chunk",
            "transcript-chunk",
            "transcript",
            "transcript-stop",
        ]
        chunks = [
            TranscriptChunk.from_event(call.args[0]).text
            for call in enhanced_handler.write_event.call_args_list
            if call.args[0].type == "transcript-chunk"
        ]
        assert chunks == ["Hello", "Hello word"]
        transcript_event = next(
            call.args[0] for call in enhanced_handler.write_event.call_args_list if call.args[0].type == "transcript"
        )
        assert Transcript.from_event(transcript_event).text == "Hello world"
        # The closing handshake runs in the background so it cannot hold up the Wyoming client
        await enhanced_handler._drain_background_tasks()
        assert manager.exited is True
        assert connection.closed is True
        assert enhanced_handler._realtime_receive_task is None

        # A second request on the same Wyoming connection starts with an empty snapshot.
        connection = _FakeRealtimeConnection(
            [
                _FakeRealtimeServerEvent("conversation.item.input_audio_transcription.delta", delta="Next"),
                _FakeRealtimeServerEvent("conversation.item.input_audio_transcription.completed", transcript="Next!"),
            ]
        )
        stt_client.realtime.connect.return_value = _FakeRealtimeConnectionManager(connection)
        enhanced_handler.write_event.reset_mock()
        assert await self._record_and_stop(enhanced_handler, "gpt-realtime-whisper")
        chunks = [
            TranscriptChunk.from_event(call.args[0]).text
            for call in enhanced_handler.write_event.call_args_list
            if call.args[0].type == "transcript-chunk"
        ]
        assert chunks == ["Next"]
        await enhanced_handler._drain_background_tasks()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("deltas", "completed", "expected_chunks", "expected_final"),
        [
            ([""], "Only final text", [], "Only final text"),
            (["Hel", "lo"], "", ["Hel", "Hello"], "Hello"),
            (["Hel", "lo"], "Hello.", ["Hel", "Hello"], "Hello."),
        ],
    )
    async def test_realtime_transcription_final_text(
        self, enhanced_handler, mock_info, mock_clients, deltas, completed, expected_chunks, expected_final
    ):
        """A completion carries the final text, but one without any leaves the streamed snapshots standing."""
        stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
        connection = _FakeRealtimeConnection(
            [
                *[
                    _FakeRealtimeServerEvent("conversation.item.input_audio_transcription.delta", delta=delta)
                    for delta in deltas
                ],
                _FakeRealtimeServerEvent(
                    "conversation.item.input_audio_transcription.completed", transcript=completed
                ),
            ]
        )
        stt_client.realtime.connect = Mock(return_value=_FakeRealtimeConnectionManager(connection))

        assert await self._record_and_stop(enhanced_handler, "gpt-realtime-whisper")

        events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
        assert [TranscriptChunk.from_event(e).text for e in events if e.type == "transcript-chunk"] == expected_chunks
        assert events[-2].type == "transcript"
        assert Transcript.from_event(events[-2]).text == expected_final
        assert events[-1].type == "transcript-stop"
        await enhanced_handler._drain_background_tasks()

    @pytest.mark.asyncio
    async def test_handle_realtime_transcription_error_cleans_up(self, enhanced_handler, mock_info, mock_clients):
        """Test realtime STT errors are reported with an error event, close the connection and clean up."""
        stt_client, _ = mock_clients
        mock_info.asr[0].models[0].name = "gpt-realtime-whisper"
        enhanced_handler._stt_realtime_models = {"gpt-realtime-whisper"}

        connection = _FakeRealtimeConnection(
            [
                _FakeRealtimeServerEvent(
                    "conversation.item.input_audio_transcription.failed", error={"message": "bad audio"}
                )
            ]
        )
        manager = _FakeRealtimeConnectionManager(connection)
        stt_client.realtime.connect = Mock(return_value=manager)
        stt_client.audio.transcriptions.create = AsyncMock()

        assert await enhanced_handler.handle_event(
            Event(type="transcribe", data={"language": "en", "name": "gpt-realtime-whisper"})
        )
        await enhanced_handler.handle_event(Event(type="audio-start", data={"rate": 24000, "width": 2, "channels": 1}))
        await enhanced_handler.handle_event(
            Event(type="audio-chunk", data={"rate": 24000, "width": 2, "channels": 1}, payload=b"\x00\x01" * 10)
        )
        assert await enhanced_handler.handle_event(Event(type="audio-stop")) is False

        stt_client.realtime.connect.assert_called_once_with(extra_query={"intent": "transcription"})
        stt_client.audio.transcriptions.create.assert_not_called()
        event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
        assert event_types == ["transcript-start", "error"]
        # The closing handshake runs in the background so it cannot hold up the Wyoming client
        await enhanced_handler._drain_background_tasks()
        assert manager.exited is True
        assert connection.closed is True
        assert enhanced_handler._realtime_receive_task is None

    @pytest.mark.asyncio
    async def test_handle_realtime_audio_stop_missing_future_reports_the_failure(self, enhanced_handler):
        """Test realtime STT stop reports a failure when future state is missing."""
        connection = Mock()
        connection.input_audio_buffer.commit = AsyncMock()
        connection.close = AsyncMock()
        enhanced_handler._is_recording = True
        enhanced_handler._realtime_connection = connection
        enhanced_handler._realtime_transcript_future = None
        enhanced_handler.write_event = AsyncMock()
        audio_stream = _realtime_stream()

        assert await enhanced_handler._handle_realtime_audio_stop(audio_stream) is False

        event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
        assert event_types == ["error"]
        connection.input_audio_buffer.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_realtime_audio_stop_keeps_its_connection_when_cleanup_runs_during_flush(
        self, enhanced_handler
    ):
        """A stop() during the awaited flush skips the commit instead of using a closing session."""
        connection = Mock()
        append_started = asyncio.Event()
        release_append = asyncio.Event()
        append_count = 0

        async def append(*, audio):
            nonlocal append_count
            append_count += 1
            if append_count == 2:
                append_started.set()
                await release_append.wait()

        connection.input_audio_buffer.append = AsyncMock(side_effect=append)
        connection.input_audio_buffer.commit = AsyncMock()
        connection.close = AsyncMock()
        audio_stream = _realtime_stream(16000)
        enhanced_handler._audio_stream = audio_stream
        enhanced_handler._is_recording = True
        enhanced_handler._realtime_connection = connection
        enhanced_handler._realtime_transcript_future = asyncio.get_running_loop().create_future()
        await enhanced_handler._handle_realtime_audio_chunk(
            audio_stream, AudioChunk(audio=b"\x00\x00" * 10, rate=16000, width=2, channels=1)
        )

        stop_task = asyncio.create_task(enhanced_handler._handle_realtime_audio_stop(audio_stream))
        try:
            await asyncio.wait_for(append_started.wait(), 1)
            await enhanced_handler._cleanup_realtime_transcription()
            release_append.set()

            assert await stop_task is False
            connection.input_audio_buffer.commit.assert_not_called()
        finally:
            release_append.set()
            if not stop_task.done():
                stop_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stop_task
            await enhanced_handler._drain_background_tasks()

    @pytest.mark.asyncio
    async def test_realtime_audio_start_rejects_a_tiny_sample_rate(
        self, enhanced_handler, mock_info, mock_clients, caplog
    ):
        """A rate below the speech minimum must not make the resampler expand a sample into seconds of output."""
        stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
        stt_client.realtime.connect = Mock()

        assert await enhanced_handler.handle_event(Event(type="transcribe", data={"name": "gpt-realtime-whisper"}))
        with caplog.at_level(logging.WARNING, logger="wyoming_openai.handler"):
            await enhanced_handler.handle_event(
                Event(type="audio-start", data={"rate": 1, "width": 2, "channels": 1})
            )

        assert "Rejecting audio-start: realtime transcription requires" in caplog.text
        assert [call.args[0].type for call in enhanced_handler.write_event.call_args_list] == ["error"]
        assert enhanced_handler._is_recording is False
        stt_client.realtime.connect.assert_not_called()

    @pytest.mark.asyncio
    async def test_realtime_audio_start_coerces_whole_float_format(
        self, enhanced_handler, mock_info, mock_clients
    ):
        """A whole float rate reaches the resampler as an integer, so chunk conversion does not raise."""
        stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
        connection = _FakeRealtimeConnection([])
        stt_client.realtime.connect = Mock(return_value=_FakeRealtimeConnectionManager(connection))

        assert await enhanced_handler.handle_event(Event(type="transcribe", data={"name": "gpt-realtime-whisper"}))
        try:
            await enhanced_handler.handle_event(
                Event(type="audio-start", data={"rate": 16000.0, "width": 2.0, "channels": 1.0})
            )
            await enhanced_handler.handle_event(
                Event(
                    type="audio-chunk",
                    data={"rate": 16000.0, "width": 2.0, "channels": 1.0},
                    payload=b"\x00\x01" * 10,
                )
            )

            assert enhanced_handler._is_recording is True
            assert len(connection.input_audio_buffer.appended_audio) == 1
        finally:
            await enhanced_handler._cleanup_realtime_transcription()
            await enhanced_handler._drain_background_tasks()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("realtime", [False, True], ids=["http", "realtime"])
    @pytest.mark.parametrize(
        "audio_format",
        [
            pytest.param({"rate": 0, "width": 2, "channels": 1}, id="zero-rate"),
            pytest.param({"rate": 16000.5, "width": 2, "channels": 1}, id="fractional-rate"),
            pytest.param({"rate": "16000", "width": 2, "channels": 1}, id="string-rate"),
            pytest.param({"rate": True, "width": 2, "channels": 1}, id="boolean-rate"),
            pytest.param({"rate": 16000, "width": 5, "channels": 1}, id="unsupported-width"),
            pytest.param({"rate": 16000, "width": 2, "channels": 0}, id="zero-channels"),
            pytest.param({"rate": 16000, "width": 4, "channels": 32768}, id="oversized-frame"),
            pytest.param({"rate": 4294967295, "width": 2, "channels": 1}, id="byte-rate-overflow"),
        ],
    )
    async def test_invalid_audio_start_reports_the_failure_instead_of_raising(
        self, enhanced_handler, mock_info, mock_clients, audio_format, realtime
    ):
        """Both transports reject an unusable audio format with an error event before wave.open() can raise."""
        stt_client, _ = mock_clients
        if realtime:
            _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)

        await enhanced_handler.handle_event(Event(type="audio-start", data=audio_format))
        await enhanced_handler.handle_event(
            Event(type="audio-chunk", data={"rate": 16000, "width": 2, "channels": 1}, payload=b"\x00\x01")
        )

        assert enhanced_handler._is_recording is False
        assert enhanced_handler._wav_buffer is None
        assert enhanced_handler._audio_stream is not None
        assert enhanced_handler._audio_stream.realtime is realtime
        assert enhanced_handler._audio_stream.failed is True
        assert [call.args[0].type for call in enhanced_handler.write_event.call_args_list] == ["error"]
        assert await enhanced_handler.handle_event(Event(type="audio-stop")) is False
        if realtime:
            stt_client.realtime.connect.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failing_step", ["connect", "session.update send"])
    async def test_failed_realtime_session_start_reports_the_error_at_once(
        self, enhanced_handler, mock_info, mock_clients, failing_step, caplog
    ):
        """A Realtime session that cannot be opened is reported right away, its audio is dropped, and its audio
        stop closes the connection for clients that do not know the error event."""
        stt_client, _ = mock_clients
        mock_info.asr[0].models[0].name = "gpt-realtime-whisper"
        enhanced_handler._stt_realtime_models = {"gpt-realtime-whisper"}
        if failing_step == "connect":
            stt_client.realtime.connect = Mock(side_effect=RuntimeError("invalid api key"))
        else:
            connection = _FakeRealtimeConnection([])
            connection.session.update = AsyncMock(side_effect=RuntimeError("invalid api key"))
            stt_client.realtime.connect = Mock(return_value=_FakeRealtimeConnectionManager(connection))

        assert await enhanced_handler.handle_event(
            Event(type="transcribe", data={"language": "en", "name": "gpt-realtime-whisper"})
        )
        await enhanced_handler.handle_event(Event(type="audio-start", data={"rate": 24000, "width": 2, "channels": 1}))

        # Before the audio stop, so a client that reads while it streams audio can end the request
        event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
        assert event_types == ["error"]
        error = Error.from_event(enhanced_handler.write_event.call_args_list[0].args[0])
        assert error.text == STT_FAILED_ERROR
        # The cause stays in the log
        assert "invalid api key" not in error.text

        with caplog.at_level(logging.WARNING, logger="wyoming_openai.handler"):
            for _ in range(3):
                await enhanced_handler.handle_event(
                    Event(type="audio-chunk", data={"rate": 24000, "width": 2, "channels": 1}, payload=b"\x00\x01" * 10)
                )
            assert await enhanced_handler.handle_event(Event(type="audio-stop")) is False

        # Nothing follows the error, and the chunks were dropped without a warning each
        assert [call.args[0].type for call in enhanced_handler.write_event.call_args_list] == ["error"]
        assert "audio chunk" not in caplog.text

        # A second audio stop without a new audio stream is not answered again
        enhanced_handler.write_event.reset_mock()
        assert await enhanced_handler.handle_event(Event(type="audio-stop")) is True
        enhanced_handler.write_event.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "server_message",
        [
            pytest.param("The current model does not support realtime transcription.", id="server-error"),
            pytest.param(None, id="transport-error-only"),
        ],
    )
    async def test_realtime_startup_close_logs_the_captured_server_error(
        self, enhanced_handler, mock_info, mock_clients, caplog, server_message
    ):
        """An error received before session.update fails must survive cleanup, with the send traceback intact."""
        stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
        connection = _FakeRealtimeConnection([])
        manager = _FakeRealtimeConnectionManager(connection)
        stt_client.realtime.connect = Mock(return_value=manager)
        send_error = ConnectionClosedOK(Close(1000, ""), Close(1000, ""), True)

        async def update_then_closed(**kwargs):
            if server_message is not None:
                await connection.events.put(_FakeRealtimeServerEvent("error", error={"message": server_message}))
                # Deterministic ordering: the receiver has captured the rejection when sending fails.
                await asyncio.wait_for(enhanced_handler._realtime_receive_task, timeout=1)
            raise send_error

        connection.session.update = AsyncMock(side_effect=update_then_closed)

        with caplog.at_level(logging.ERROR, logger="wyoming_openai.handler"):
            await enhanced_handler.handle_event(
                Event(type="audio-start", data={"rate": 24000, "width": 2, "channels": 1})
            )
            for _ in range(3):
                await enhanced_handler.handle_event(
                    Event(type="audio-chunk", data={"rate": 24000, "width": 2, "channels": 1}, payload=b"\x00\x01")
                )
            assert await enhanced_handler.handle_event(Event(type="audio-stop")) is False

        records = [record for record in caplog.records if record.name == "wyoming_openai.handler"]
        assert len(records) == 1
        cause = server_message if server_message is not None else str(send_error)
        assert records[0].getMessage() == f"Error starting realtime transcription: {cause}"
        assert records[0].exc_info is not None
        assert records[0].exc_info[1] is send_error
        assert str(send_error) in caplog.text

        events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
        assert [event.type for event in events] == ["error"]
        assert Error.from_event(events[0]).text == STT_FAILED_ERROR
        assert enhanced_handler._realtime_transcript_future is None
        assert enhanced_handler._realtime_receive_task is None
        await enhanced_handler._drain_background_tasks()
        assert manager.exited is True
        assert connection.closed is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "server_event",
        [
            pytest.param(
                _FakeRealtimeServerEvent(
                    "error", error={"message": "The 'keywords' parameter is not supported for this model."}
                ),
                id="settings-rejected",
            ),
            pytest.param(None, id="socket-closed"),
        ],
    )
    async def test_realtime_session_that_fails_during_the_audio_is_reported_with_the_next_chunk(
        self, enhanced_handler, mock_info, mock_clients, server_event, caplog
    ):
        """session.update only sends the settings, so a rejection or a closed socket is reported once it is read."""
        stt_client, _ = mock_clients
        mock_info.asr[0].models[0].name = "gpt-realtime-whisper"
        enhanced_handler._stt_realtime_models = {"gpt-realtime-whisper"}
        connection = _FakeRealtimeConnection([])
        manager = _FakeRealtimeConnectionManager(connection)
        stt_client.realtime.connect = Mock(return_value=manager)
        chunk = Event(type="audio-chunk", data={"rate": 24000, "width": 2, "channels": 1}, payload=b"\x00\x01" * 10)

        await enhanced_handler.handle_event(Event(type="audio-start", data={"rate": 24000, "width": 2, "channels": 1}))
        await enhanced_handler.handle_event(chunk)
        # The server answers the settings, or the socket closes, while the speaker is still talking
        await connection.events.put(server_event)
        for _ in range(3):
            await asyncio.sleep(0)
        with caplog.at_level(logging.WARNING, logger="wyoming_openai.handler"):
            for _ in range(3):
                await enhanced_handler.handle_event(chunk)
            assert await enhanced_handler.handle_event(Event(type="audio-stop")) is False

        assert [call.args[0].type for call in enhanced_handler.write_event.call_args_list] == [
            "transcript-start",
            "error",
        ]
        # Only the audio that arrived before the failure was sent, and the failure is logged once
        assert len(connection.input_audio_buffer.appended_audio) == 1
        assert connection.input_audio_buffer.committed is False
        assert caplog.text.count("Realtime transcription failed while the audio was arriving") == 1
        await enhanced_handler._drain_background_tasks()
        assert manager.exited is True

    @pytest.mark.asyncio
    async def test_realtime_audio_that_cannot_be_sent_is_reported_once(
        self, enhanced_handler, mock_info, mock_clients, caplog
    ):
        """A socket that drops during the audio fails the stream at once, rather than once for every chunk."""
        stt_client, _ = mock_clients
        mock_info.asr[0].models[0].name = "gpt-realtime-whisper"
        enhanced_handler._stt_realtime_models = {"gpt-realtime-whisper"}
        connection = _FakeRealtimeConnection([])
        connection.input_audio_buffer.append = AsyncMock(side_effect=ConnectionResetError("websocket closed"))
        stt_client.realtime.connect = Mock(return_value=_FakeRealtimeConnectionManager(connection))
        chunk = Event(type="audio-chunk", data={"rate": 24000, "width": 2, "channels": 1}, payload=b"\x00\x01" * 10)

        await enhanced_handler.handle_event(Event(type="audio-start", data={"rate": 24000, "width": 2, "channels": 1}))
        with caplog.at_level(logging.WARNING, logger="wyoming_openai.handler"):
            for _ in range(5):
                await enhanced_handler.handle_event(chunk)
            assert await enhanced_handler.handle_event(Event(type="audio-stop")) is False

        assert connection.input_audio_buffer.append.await_count == 1
        assert caplog.text.count("Error sending realtime audio chunk") == 1
        assert [call.args[0].type for call in enhanced_handler.write_event.call_args_list] == [
            "transcript-start",
            "error",
        ]
        await enhanced_handler._drain_background_tasks()

    @pytest.mark.asyncio
    async def test_dropping_a_failed_realtime_stream_leaves_no_unread_error(
        self, enhanced_handler, mock_info, mock_clients, caplog
    ):
        """A session that failed before anyone waited for its transcript does not log its error as never read."""
        import gc

        stt_client, _ = mock_clients
        mock_info.asr[0].models[0].name = "gpt-realtime-whisper"
        enhanced_handler._stt_realtime_models = {"gpt-realtime-whisper"}
        connection = _FakeRealtimeConnection([])
        stt_client.realtime.connect = Mock(return_value=_FakeRealtimeConnectionManager(connection))

        await enhanced_handler.handle_event(Event(type="audio-start", data={"rate": 24000, "width": 2, "channels": 1}))
        # The socket closes, which fails the transcript, and the client goes away before sending more audio
        await connection.events.put(None)
        for _ in range(3):
            await asyncio.sleep(0)
        with caplog.at_level(logging.ERROR, logger="asyncio"):
            await enhanced_handler.disconnect()
            gc.collect()

        assert "exception was never retrieved" not in caplog.text

    @pytest.mark.asyncio
    async def test_stop_during_a_realtime_audio_stop_cancels_the_transcription(
        self, enhanced_handler, mock_info, mock_clients, caplog
    ):
        """A server shutdown while the transcript is awaited cancels that audio stop, which closes the connection."""
        stt_client, _ = mock_clients
        mock_info.asr[0].models[0].name = "gpt-realtime-whisper"
        enhanced_handler._stt_realtime_models = {"gpt-realtime-whisper"}
        connection = _FakeRealtimeConnection([])
        stt_client.realtime.connect = Mock(return_value=_FakeRealtimeConnectionManager(connection))

        written = []

        async def write(event):
            written.append(Error.from_event(event).text if event.type == "error" else event.type)

        enhanced_handler.write_event = AsyncMock(side_effect=write)
        enhanced_handler.writer.close = Mock(side_effect=lambda: written.append("<closed>"))

        await enhanced_handler.handle_event(Event(type="audio-start", data={"rate": 24000, "width": 2, "channels": 1}))
        with caplog.at_level(logging.INFO, logger="wyoming_openai.handler"):
            stopping = asyncio.create_task(enhanced_handler.handle_event(Event(type="audio-stop")))
            for _ in range(3):
                await asyncio.sleep(0)
            assert connection.input_audio_buffer.committed is True
            await enhanced_handler.stop()
            # Ends once stop() has answered the client and closed the connection
            assert await stopping is False

        assert "Cancelling the STT request in flight" in caplog.text
        assert not [record for record in caplog.records if record.levelno >= logging.ERROR]
        # Told before the writer closes, and only once: the cancelled audio stop writes nothing
        assert written == ["transcript-start", SHUTDOWN_ERROR, "<closed>"]
        await enhanced_handler._drain_background_tasks()

    @pytest.mark.asyncio
    async def test_realtime_audio_stop_without_audio_start_emits_nothing(self, enhanced_handler, mock_info):
        """A stray Realtime audio stop does not produce a transcript."""
        mock_info.asr[0].models[0].name = "gpt-realtime-whisper"
        enhanced_handler._stt_realtime_models = {"gpt-realtime-whisper"}

        assert await enhanced_handler.handle_event(
            Event(type="transcribe", data={"language": "en", "name": "gpt-realtime-whisper"})
        )
        await enhanced_handler.handle_event(Event(type="audio-stop"))

        enhanced_handler.write_event.assert_not_called()

    async def _record_and_stop(self, handler, model_name="whisper-1"):
        """Send a transcribe request and a short audio stream; returns whether the connection stays open."""
        assert await handler.handle_event(Event(type="transcribe", data={"language": "en", "name": model_name}))
        await handler.handle_event(Event(type="audio-start", data={"rate": 16000, "width": 2, "channels": 1}))
        await handler.handle_event(
            Event(type="audio-chunk", data={"rate": 16000, "width": 2, "channels": 1}, payload=b"\x00\x01" * 100)
        )
        return await handler.handle_event(Event(type="audio-stop"))

    @pytest.mark.asyncio
    async def test_transcription_request_error_reports_the_failure(self, enhanced_handler, mock_clients):
        """A failed /v1/audio/transcriptions request is reported with an error event, and the connection is closed."""
        stt_client, _ = mock_clients
        stt_client.audio.transcriptions.create = AsyncMock(side_effect=RuntimeError("400 unknown parameter"))

        assert await self._record_and_stop(enhanced_handler) is False

        event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
        assert event_types == ["error"]
        assert Error.from_event(enhanced_handler.write_event.call_args_list[0].args[0]).text == STT_FAILED_ERROR
        assert enhanced_handler._wav_buffer is None

    @pytest.mark.asyncio
    async def test_transcription_failure_on_a_closed_connection_does_not_raise(self, enhanced_handler, mock_clients):
        """A client that is gone before the error event is written does not make the handler raise."""
        stt_client, _ = mock_clients
        stt_client.audio.transcriptions.create = AsyncMock(side_effect=RuntimeError("503 service unavailable"))
        written: list[str] = []

        async def write(event):
            written.append(event.type)
            raise ConnectionResetError("connection closed by the client")

        enhanced_handler.write_event = AsyncMock(side_effect=write)

        assert await self._record_and_stop(enhanced_handler) is False

        assert written == ["error"]
        assert enhanced_handler._wav_buffer is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("deltas", "done_text", "expected_chunks", "expected_final"),
        [
            (["Hello", "", " word"], "Hello world!", ["Hello", "Hello word"], "Hello world!"),
            (["Hello", " world"], None, ["Hello", "Hello world"], "Hello world"),
            (["Noise"], "", ["Noise"], "Noise"),  # Speaches 0.9 sends the deltas and an empty final text
            ([], "Only final text", [], "Only final text"),
            ([""], None, [], ""),
        ],
    )
    async def test_transcription_stream_snapshots(
        self, enhanced_handler, mock_clients, deltas, done_text, expected_chunks, expected_final
    ):
        """SSE deltas become replaceable snapshots; final text can correct them or arrive without deltas."""
        stt_client, _ = mock_clients

        class _Stream(_FakeTranscriptionStream):
            async def __aiter__(self):
                for delta in deltas:
                    yield Mock(type="transcript.text.delta", delta=delta)
                if done_text is not None:
                    yield Mock(type="transcript.text.done", text=done_text)

        stt_client.audio.transcriptions.create = AsyncMock(return_value=_Stream())
        with patch("wyoming_openai.handler.AsyncStream", _Stream):
            # Reuse the handler to catch accidental accumulation across requests.
            for _ in range(2):
                enhanced_handler.write_event.reset_mock()
                assert await self._record_and_stop(enhanced_handler)
                events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
                assert [event.type for event in events] == [
                    "transcript-start",
                    *(["transcript-chunk"] * len(expected_chunks)),
                    "transcript",
                    "transcript-stop",
                ]
                assert [TranscriptChunk.from_event(e).text for e in events if e.type == "transcript-chunk"] == (
                    expected_chunks
                )
                assert Transcript.from_event(events[-2]).text == expected_final

    @pytest.mark.asyncio
    async def test_transcription_stream_error_reports_the_failure(self, enhanced_handler, mock_clients):
        """A transcription stream that fails midway ends with an error event and a closed connection."""
        stt_client, _ = mock_clients

        class _FailingStream(_FakeTranscriptionStream):
            def __aiter__(self):
                return self._events()

            async def _events(self):
                yield Mock(type="transcript.text.delta", delta="Hel")
                raise RuntimeError("stream reset")

        stream = _FailingStream()
        stt_client.audio.transcriptions.create = AsyncMock(return_value=stream)

        with patch("wyoming_openai.handler.AsyncStream", _FailingStream):
            assert await self._record_and_stop(enhanced_handler) is False

        event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
        assert event_types == ["transcript-start", "transcript-chunk", "error"]
        assert stream.closed

    @pytest.mark.asyncio
    async def test_transcription_stream_is_closed_when_the_client_goes_away(self, enhanced_handler, mock_clients):
        """A client that disconnects mid-stream does not leave the response open."""
        stt_client, _ = mock_clients

        class _Stream(_FakeTranscriptionStream):
            async def __aiter__(self):
                yield Mock(type="transcript.text.delta", delta="Hel")
                yield Mock(type="transcript.text.delta", delta="lo")

        stream = _Stream()
        stt_client.audio.transcriptions.create = AsyncMock(return_value=stream)

        async def write_event(event):
            if event.type == "transcript-chunk":
                raise ConnectionResetError("client went away")

        enhanced_handler.write_event = AsyncMock(side_effect=write_event)

        with patch("wyoming_openai.handler.AsyncStream", _Stream):
            assert await self._record_and_stop(enhanced_handler) is False

        assert stream.closed

    @pytest.mark.asyncio
    async def test_transcription_without_asr_model_reports_the_failure(self, enhanced_handler, mock_info, mock_clients):
        """Audio that no model can transcribe is reported as a failed transcription."""
        stt_client, _ = mock_clients
        stt_client.audio.transcriptions.create = AsyncMock()
        mock_info.asr[0].models.clear()

        await enhanced_handler.handle_event(Event(type="audio-start", data={"rate": 16000, "width": 2, "channels": 1}))
        assert await enhanced_handler.handle_event(Event(type="audio-stop")) is False

        stt_client.audio.transcriptions.create.assert_not_called()
        event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
        assert event_types == ["error"]

    @pytest.mark.asyncio
    async def test_unexpected_transcription_response_reports_the_failure(self, enhanced_handler, mock_clients):
        """A response of an unknown type is reported to the client as a failed transcription."""
        stt_client, _ = mock_clients
        stt_client.audio.transcriptions.create = AsyncMock(return_value=object())

        assert await self._record_and_stop(enhanced_handler) is False

        event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
        # No transcript is opened for a response that cannot be read
        assert event_types == ["error"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("width,channels", [(1, 1), (2, 1), (3, 1), (4, 1), (1, 2), (2, 2), (3, 2), (4, 2)])
    async def test_realtime_pcm_keeps_partial_frames_between_chunks(self, enhanced_handler, width, channels):
        """Chunk boundaries must not shift PCM samples or discard stereo frames."""
        audio = bytes(range(width * channels * 5))
        connection = Mock()
        connection.input_audio_buffer.append = AsyncMock()
        enhanced_handler._realtime_connection = connection
        enhanced_handler._audio_stream = _realtime_stream(width=width, channels=channels)
        enhanced_handler._is_recording = True
        for part in (audio[:1], audio[1:width * channels + 1], audio[width * channels + 1:]):
            await enhanced_handler.handle_event(
                Event(type="audio-chunk", data={"rate": 24000, "width": width, "channels": channels}, payload=part)
            )
        sent = b"".join(
            base64.b64decode(call.kwargs["audio"]) for call in connection.input_audio_buffer.append.call_args_list
        )
        expected = downmix_pcm_samples(decode_pcm_samples(audio, width), channels)
        assert sent == encode_pcm16_samples(expected)
        assert all(
            len(base64.b64decode(call.kwargs["audio"])) % 2 == 0
            for call in connection.input_audio_buffer.append.call_args_list
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("rate", [8000, 16000, 44100, 48000])
    @pytest.mark.parametrize("chunk_frames", [160, 161])
    @pytest.mark.parametrize("channels", [1, 2])
    async def test_realtime_resampling_matches_a_whole_buffer_reference(
        self, enhanced_handler, mock_info, mock_clients, rate, chunk_frames, channels
    ):
        """Resampling carries its phase across chunks, so a stream matches a whole-buffer reference."""
        stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
        connection = _FakeRealtimeConnection(
            [
                _FakeRealtimeServerEvent(
                    "conversation.item.input_audio_transcription.completed", transcript="Hello"
                )
            ]
        )
        stt_client.realtime.connect = Mock(return_value=_FakeRealtimeConnectionManager(connection))

        mono_samples = sine_samples(rate, 0.25)
        audio = interleave_pcm(mono_samples, channels)
        assert await enhanced_handler.handle_event(Event(type="transcribe", data={"name": "gpt-realtime-whisper"}))
        try:
            await enhanced_handler.handle_event(
                Event(type="audio-start", data={"rate": rate, "width": 2, "channels": channels})
            )
            chunk_size = chunk_frames * channels * 2
            for start in range(0, len(audio), chunk_size):
                await enhanced_handler.handle_event(
                    Event(
                        type="audio-chunk",
                        data={"rate": rate, "width": 2, "channels": channels},
                        payload=audio[start : start + chunk_size],
                    )
                )
            assert await enhanced_handler.handle_event(Event(type="audio-stop")) is True

            sent = b"".join(base64.b64decode(payload) for payload in connection.input_audio_buffer.appended_audio)
            assert_samples_close(decode_pcm_samples(sent, 2), reference_resample(mono_samples, rate))
            assert connection.input_audio_buffer.committed is True
        finally:
            await enhanced_handler._cleanup_realtime_transcription()
            await enhanced_handler._drain_background_tasks()

    @pytest.mark.asyncio
    async def test_realtime_pcm_partial_frame_does_not_carry_to_the_next_stream(
        self, enhanced_handler, mock_info, mock_clients
    ):
        """A new audio-start gets a new converter, so the previous stream's partial frame cannot leak into it."""
        stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
        connections = [_FakeRealtimeConnection([]), _FakeRealtimeConnection([])]
        connections[1].commit_events.append(
            _FakeRealtimeServerEvent(
                "conversation.item.input_audio_transcription.completed", transcript="Hello"
            )
        )
        stt_client.realtime.connect = Mock(
            side_effect=[_FakeRealtimeConnectionManager(connection) for connection in connections]
        )

        assert await enhanced_handler.handle_event(Event(type="transcribe", data={"name": "gpt-realtime-whisper"}))
        try:
            await enhanced_handler.handle_event(
                Event(type="audio-start", data={"rate": 24000, "width": 2, "channels": 1})
            )
            await enhanced_handler.handle_event(
                Event(type="audio-chunk", data={"rate": 24000, "width": 2, "channels": 1}, payload=b"\xff")
            )
            # The client starts a new stream without stopping the first
            await enhanced_handler.handle_event(
                Event(type="audio-start", data={"rate": 24000, "width": 2, "channels": 1})
            )
            await enhanced_handler.handle_event(
                Event(type="audio-chunk", data={"rate": 24000, "width": 2, "channels": 1}, payload=b"\x01\x02")
            )
            assert await enhanced_handler.handle_event(Event(type="audio-stop")) is True

            assert connections[0].input_audio_buffer.appended_audio == []
            sent = b"".join(base64.b64decode(audio) for audio in connections[1].input_audio_buffer.appended_audio)
            assert sent == b"\x01\x02"
        finally:
            await enhanced_handler._cleanup_realtime_transcription()
            await enhanced_handler._drain_background_tasks()

    @pytest.mark.asyncio
    async def test_realtime_transcription_session_includes_configured_prompt(self, enhanced_handler, mock_info):
        """Test realtime STT preserves configured prompt in the session payload."""
        mock_info.asr[0].models[0].name = "gpt-4o-transcribe"
        enhanced_handler._stt_realtime_models = {"gpt-4o-transcribe"}

        assert await enhanced_handler.handle_event(
            Event(type="transcribe", data={"language": "en", "name": "gpt-4o-transcribe"})
        )

        session = enhanced_handler._get_realtime_transcription_session(
            _AudioStreamState(
                model=enhanced_handler._current_asr_model, language="en", realtime=True, streaming=False
            )
        )

        assert session["audio"]["input"]["transcription"] == {
            "model": "gpt-4o-transcribe",
            "language": "en",
            "prompt": "Test prompt",
        }

    @pytest.mark.asyncio
    async def test_handle_transcribe_preserves_configured_speaches_vad_filter(self, enhanced_handler, mock_clients):
        """Test STT requests preserve an explicit Speaches vad_filter override."""
        stt_client, _ = mock_clients
        stt_client.backend = OpenAIBackend.SPEACHES
        enhanced_handler._stt_extra_body = {"foo": "bar", "vad_filter": True}

        mock_transcription = Mock()
        mock_transcription.text = "Test transcription"
        stt_client.audio.transcriptions.create = AsyncMock(return_value=mock_transcription)

        transcribe_event = Event(type="transcribe", data={"language": "en", "name": "whisper-1"})
        assert await enhanced_handler.handle_event(transcribe_event) is True

        await enhanced_handler.handle_event(Event(type="audio-start", data={"rate": 16000, "width": 2, "channels": 1}))
        await enhanced_handler.handle_event(
            Event(type="audio-chunk", data={"rate": 16000, "width": 2, "channels": 1}, payload=b"\x00\x01" * 1000)
        )

        with patch("wyoming_openai.handler.isinstance") as mock_isinstance:

            def isinstance_side_effect(obj, class_or_tuple):
                if obj is mock_transcription:
                    from openai.types.audio.transcription_create_response import TranscriptionCreateResponse

                    return class_or_tuple is TranscriptionCreateResponse
                return builtins.isinstance(obj, class_or_tuple)

            mock_isinstance.side_effect = isinstance_side_effect
            await enhanced_handler.handle_event(Event(type="audio-stop"))

        call_args = stt_client.audio.transcriptions.create.call_args.kwargs
        assert call_args["extra_body"] == {"foo": "bar", "vad_filter": True}

    @pytest.mark.asyncio
    async def test_handle_transcribe_adds_default_speaches_vad_filter_when_missing(
        self, enhanced_handler, mock_clients
    ):
        """Test STT requests still inject the historical Speaches vad_filter default."""
        stt_client, _ = mock_clients
        stt_client.backend = OpenAIBackend.SPEACHES
        enhanced_handler._stt_extra_body = {"foo": "bar"}

        mock_transcription = Mock()
        mock_transcription.text = "Test transcription"
        stt_client.audio.transcriptions.create = AsyncMock(return_value=mock_transcription)

        transcribe_event = Event(type="transcribe", data={"language": "en", "name": "whisper-1"})
        assert await enhanced_handler.handle_event(transcribe_event) is True

        await enhanced_handler.handle_event(Event(type="audio-start", data={"rate": 16000, "width": 2, "channels": 1}))
        await enhanced_handler.handle_event(
            Event(type="audio-chunk", data={"rate": 16000, "width": 2, "channels": 1}, payload=b"\x00\x01" * 1000)
        )

        with patch("wyoming_openai.handler.isinstance") as mock_isinstance:

            def isinstance_side_effect(obj, class_or_tuple):
                if obj is mock_transcription:
                    from openai.types.audio.transcription_create_response import TranscriptionCreateResponse

                    return class_or_tuple is TranscriptionCreateResponse
                return builtins.isinstance(obj, class_or_tuple)

            mock_isinstance.side_effect = isinstance_side_effect
            await enhanced_handler.handle_event(Event(type="audio-stop"))

        call_args = stt_client.audio.transcriptions.create.call_args.kwargs
        assert call_args["extra_body"] == {"foo": "bar", "vad_filter": False}

    @pytest.mark.asyncio
    async def test_handle_transcribe_enables_streaming_when_extra_body_overrides_default(
        self, enhanced_handler, mock_clients
    ):
        """Test STT stream overrides update the client-side parser selection."""
        stt_client, _ = mock_clients
        enhanced_handler._stt_extra_body = {"stream": True}
        stt_client.audio.transcriptions.create = AsyncMock(side_effect=Exception("Streaming test - expected"))

        transcribe_event = Event(type="transcribe", data={"language": "en", "name": "whisper-1"})
        assert await enhanced_handler.handle_event(transcribe_event) is True

        await enhanced_handler.handle_event(Event(type="audio-start", data={"rate": 16000, "width": 2, "channels": 1}))
        await enhanced_handler.handle_event(
            Event(type="audio-chunk", data={"rate": 16000, "width": 2, "channels": 1}, payload=b"\x00\x01" * 100)
        )
        await enhanced_handler.handle_event(Event(type="audio-stop"))

        call_args = stt_client.audio.transcriptions.create.call_args.kwargs
        assert call_args["stream"] is True
        assert call_args["extra_body"]["stream"] is True

    @pytest.mark.asyncio
    async def test_handle_transcribe_disables_streaming_when_extra_body_overrides_default(
        self, enhanced_handler, mock_clients, mock_info
    ):
        """Test STT stream overrides can force non-streaming parsing."""
        stt_client, _ = mock_clients
        mock_info.asr[0].supports_transcript_streaming = True
        enhanced_handler._stt_extra_body = {"stream": False}
        stt_client.audio.transcriptions.create = AsyncMock(side_effect=Exception("Non-streaming test - expected"))

        transcribe_event = Event(type="transcribe", data={"language": "en", "name": "whisper-1"})
        assert await enhanced_handler.handle_event(transcribe_event) is True

        await enhanced_handler.handle_event(Event(type="audio-start", data={"rate": 16000, "width": 2, "channels": 1}))
        await enhanced_handler.handle_event(
            Event(type="audio-chunk", data={"rate": 16000, "width": 2, "channels": 1}, payload=b"\x00\x01" * 100)
        )
        await enhanced_handler.handle_event(Event(type="audio-stop"))

        call_args = stt_client.audio.transcriptions.create.call_args.kwargs
        assert call_args["stream"] is omit
        assert call_args["extra_body"]["stream"] is False

    @pytest.mark.asyncio
    async def test_handle_synthesize_event(self, enhanced_handler, mock_clients):
        """Test handling of Synthesize event."""
        _, tts_client = mock_clients

        # Create proper WAV data with header
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(b"\x00\x01" * 1000)
        wav_buffer.seek(0)
        mock_audio_data = wav_buffer.read()

        # Mock the streaming response with async iteration
        class MockAsyncIterator:
            def __init__(self, data):
                self.data = data
                self.chunks = [data[i : i + 2048] for i in range(0, len(data), 2048)]
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.chunks):
                    raise StopAsyncIteration
                chunk = self.chunks[self.index]
                self.index += 1
                return chunk

        mock_response = Mock()
        mock_response.iter_bytes = Mock(return_value=MockAsyncIterator(mock_audio_data))

        # Mock the with_streaming_response context manager
        mock_stream_response = AsyncMock()
        mock_stream_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_response.__aexit__ = AsyncMock(return_value=None)

        tts_client.audio.speech.with_streaming_response.create = Mock(return_value=mock_stream_response)

        event = Event(
            type="synthesize", data={"text": "Hello world", "voice": {"name": "alloy"}, "raw_text": "Hello world"}
        )

        # Clear previous write_event calls
        enhanced_handler.write_event.reset_mock()

        result = await enhanced_handler.handle_event(event)

        assert result is True

        # Verify TTS client was called
        tts_client.audio.speech.with_streaming_response.create.assert_called_once()

        # Verify audio events were written
        assert enhanced_handler.write_event.call_count >= 2  # At least AudioStart and AudioStop

        # Check that AudioStart and AudioStop were written
        event_types = [call[0][0].type for call in enhanced_handler.write_event.call_args_list]
        assert "audio-start" in event_types
        assert "audio-stop" in event_types

    @pytest.mark.asyncio
    async def test_handle_synthesize_event_includes_configured_tts_extra_body(self, enhanced_handler, mock_clients):
        """Test buffered TTS requests include configured extra_body."""
        _, tts_client = mock_clients
        enhanced_handler._tts_extra_body = {"response_format": "pcm"}

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(b"\x00\x01" * 1000)
        wav_buffer.seek(0)
        mock_audio_data = wav_buffer.read()

        class MockAsyncIterator:
            def __init__(self, data):
                self.data = data
                self.chunks = [data[i : i + 2048] for i in range(0, len(data), 2048)]
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.chunks):
                    raise StopAsyncIteration
                chunk = self.chunks[self.index]
                self.index += 1
                return chunk

        mock_response = Mock()
        mock_response.iter_bytes = Mock(return_value=MockAsyncIterator(mock_audio_data))

        mock_stream_response = AsyncMock()
        mock_stream_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_response.__aexit__ = AsyncMock(return_value=None)

        tts_client.audio.speech.with_streaming_response.create = Mock(return_value=mock_stream_response)

        event = Event(
            type="synthesize", data={"text": "Hello world", "voice": {"name": "alloy"}, "raw_text": "Hello world"}
        )
        assert await enhanced_handler.handle_event(event) is True

        call_args = tts_client.audio.speech.with_streaming_response.create.call_args.kwargs
        assert call_args["extra_body"] == {"response_format": "pcm"}

    @pytest.mark.asyncio
    async def test_stream_audio_to_wyoming_uses_frame_count_for_stereo_audio(self, enhanced_handler):
        """Test stereo audio timestamps are based on PCM frames, not per-channel samples."""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(b"\x00\x01\x02\x03" * 24000)
        wav_buffer.seek(0)

        timestamp = await _play_buffered_sentence(
            enhanced_handler,
            wav_buffer.read(),
            is_first_chunk=True,
            start_timestamp=0,
            audio_format=WAV_AUDIO_FORMAT,
        )

        assert timestamp == pytest.approx(1000.0)

    @pytest.mark.asyncio
    async def test_stream_audio_to_wyoming_excludes_trailing_riff_chunks(self, enhanced_handler):
        """Test buffered WAV conversion stops at the declared PCM data boundary."""
        pcm = b"\x00\x01" * 240
        wav_data = _pcm_wav_with_trailing_metadata(pcm)

        timestamp = await _play_buffered_sentence(
            enhanced_handler,
            wav_data,
            is_first_chunk=True,
            start_timestamp=0,
            audio_format=WAV_AUDIO_FORMAT,
        )

        audio_chunk_events = [
            call.args[0] for call in enhanced_handler.write_event.call_args_list if call.args[0].type == "audio-chunk"
        ]
        emitted_audio = b"".join(event.payload for event in audio_chunk_events)
        assert emitted_audio == pcm
        assert b"LIST" not in emitted_audio
        assert b"C2PA" not in emitted_audio
        assert timestamp == pytest.approx(10.0)

    @pytest.mark.asyncio
    async def test_stream_audio_to_wyoming_warns_for_truncated_wav(self, enhanced_handler, caplog):
        """Test buffered WAV conversion reports an incomplete declared PCM payload."""
        pcm = b"\x00\x01" * 240
        wav_data = _pcm_wav_with_trailing_metadata(pcm)
        wav_params = parse_wav_header(wav_data)
        assert wav_params is not None
        data_offset = wav_params[3]
        truncated_wav = wav_data[: data_offset + len(pcm) - 10]

        timestamp = await _play_buffered_sentence(
            enhanced_handler,
            truncated_wav,
            is_first_chunk=True,
            start_timestamp=0,
            audio_format=WAV_AUDIO_FORMAT,
        )

        audio_chunk_events = [
            call.args[0] for call in enhanced_handler.write_event.call_args_list if call.args[0].type == "audio-chunk"
        ]
        assert b"".join(event.payload for event in audio_chunk_events) == pcm[:-10]
        assert timestamp == pytest.approx(470 / 2 / 24000 * 1000)
        assert "ended with 10 declared audio bytes missing" in caplog.text

    @pytest.mark.asyncio
    async def test_stream_audio_to_wyoming_accepts_unbounded_wav_data(self, enhanced_handler, caplog):
        """Test buffered WAV conversion accepts an unbounded data chunk size."""
        pcm = b"\x00\x01" * 240
        wav_data = _pcm_wav_with_unbounded_sizes(pcm)

        wav_params = parse_wav_header(wav_data)
        assert wav_params == WavHeader(24000, 1, 2, 44, None, WAVE_FORMAT_PCM)

        timestamp = await _play_buffered_sentence(
            enhanced_handler,
            wav_data,
            is_first_chunk=True,
            start_timestamp=0,
            audio_format=WAV_AUDIO_FORMAT,
        )

        audio_start_events = [
            call.args[0] for call in enhanced_handler.write_event.call_args_list if call.args[0].type == "audio-start"
        ]
        audio_chunk_events = [
            call.args[0] for call in enhanced_handler.write_event.call_args_list if call.args[0].type == "audio-chunk"
        ]
        assert len(audio_start_events) == 1
        assert audio_start_events[0].data == {"rate": 24000, "width": 2, "channels": 1, "timestamp": None}
        assert b"".join(event.payload for event in audio_chunk_events) == pcm
        assert audio_chunk_events[0].data["timestamp"] == 0
        assert timestamp == pytest.approx(10.0)
        assert "declared audio bytes missing" not in caplog.text
        assert "ended after" not in caplog.text

    @pytest.mark.asyncio
    async def test_stream_tts_audio_uses_frame_count_for_stereo_audio(self, enhanced_handler, mock_clients):
        """Test direct TTS streaming preserves correct stereo timing."""
        _, tts_client = mock_clients

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(b"\x00\x01\x02\x03" * 24000)
        wav_buffer.seek(0)
        mock_audio_data = wav_buffer.read()

        class MockAsyncIterator:
            def __init__(self, data):
                self.data = data
                self.chunks = [data]
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.chunks):
                    raise StopAsyncIteration
                chunk = self.chunks[self.index]
                self.index += 1
                return chunk

        mock_response = Mock()
        mock_response.iter_bytes = Mock(return_value=MockAsyncIterator(mock_audio_data))

        mock_stream_response = AsyncMock()
        mock_stream_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_response.__aexit__ = AsyncMock(return_value=None)

        tts_client.audio.speech.with_streaming_response.create = Mock(return_value=mock_stream_response)

        voice = enhanced_handler._get_voice("alloy")
        assert voice is not None

        timestamp = await _stream_whole_request(enhanced_handler, voice, "Hello world", send_audio_start=True)

        assert timestamp == pytest.approx(1000.0)

    @pytest.mark.asyncio
    async def test_stream_tts_audio_buffers_fragmented_wav_header(self, enhanced_handler, mock_clients):
        """Test fragmented WAV headers are buffered without breaking stereo timestamps."""
        _, tts_client = mock_clients

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(b"\x00\x01\x02\x03" * 1000)
        wav_buffer.seek(0)
        wav_data = wav_buffer.read()

        fragmented_chunks = [wav_data[:20], wav_data[20:60], wav_data[60:]]

        class MockAsyncIterator:
            def __init__(self, chunks):
                self.chunks = chunks
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.chunks):
                    raise StopAsyncIteration
                chunk = self.chunks[self.index]
                self.index += 1
                return chunk

        mock_response = Mock()
        mock_response.iter_bytes = Mock(return_value=MockAsyncIterator(fragmented_chunks))

        mock_stream_response = AsyncMock()
        mock_stream_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_response.__aexit__ = AsyncMock(return_value=None)

        tts_client.audio.speech.with_streaming_response.create = Mock(return_value=mock_stream_response)

        voice = enhanced_handler._get_voice("alloy")
        assert voice is not None

        enhanced_handler.write_event.reset_mock()
        timestamp = await _stream_whole_request(enhanced_handler, voice, "Hello world", send_audio_start=True)

        expected_frames = (len(wav_data) - 44) // (2 * 2)
        assert timestamp == pytest.approx(expected_frames / 24000 * 1000)

        audio_chunk_events = [
            call.args[0] for call in enhanced_handler.write_event.call_args_list if call.args[0].type == "audio-chunk"
        ]
        assert audio_chunk_events
        assert audio_chunk_events[0].payload == wav_data[44:60]
        assert b"".join(event.payload for event in audio_chunk_events) == wav_data[44:]

    @pytest.mark.asyncio
    async def test_stream_tts_audio_excludes_trailing_riff_chunks_across_http_boundaries(
        self, enhanced_handler, mock_clients
    ):
        """Test incremental WAV conversion stops at the data boundary for varied HTTP chunks."""
        _, tts_client = mock_clients
        pcm = b"\x00\x01" * 240
        wav_data = _pcm_wav_with_trailing_metadata(pcm)
        wav_params = parse_wav_header(wav_data)
        assert wav_params is not None
        data_offset = wav_params[3]
        data_end = data_offset + len(pcm)
        chunk_layouts = [
            [wav_data],
            [wav_data[: data_end - 8], wav_data[data_end - 8 : data_end + 4], wav_data[data_end + 4 :]],
            [wav_data[:data_end], wav_data[data_end:]],
        ]

        class MockAsyncIterator:
            def __init__(self, chunks):
                self.chunks = chunks
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.chunks):
                    raise StopAsyncIteration
                chunk = self.chunks[self.index]
                self.index += 1
                return chunk

        voice = enhanced_handler._get_voice("alloy")
        assert voice is not None

        for chunks in chunk_layouts:
            mock_response = Mock()
            mock_response.iter_bytes = Mock(return_value=MockAsyncIterator(chunks))
            mock_stream_response = AsyncMock()
            mock_stream_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream_response.__aexit__ = AsyncMock(return_value=None)
            tts_client.audio.speech.with_streaming_response.create = Mock(return_value=mock_stream_response)
            enhanced_handler.write_event.reset_mock()

            timestamp = await _stream_whole_request(enhanced_handler, voice, "Hello world", send_audio_start=True)

            audio_chunk_events = [
                call.args[0]
                for call in enhanced_handler.write_event.call_args_list
                if call.args[0].type == "audio-chunk"
            ]
            emitted_audio = b"".join(event.payload for event in audio_chunk_events)
            assert emitted_audio == pcm
            assert b"LIST" not in emitted_audio
            assert b"C2PA" not in emitted_audio
            assert timestamp == pytest.approx(10.0)

    @pytest.mark.asyncio
    async def test_stream_tts_audio_warns_for_truncated_wav(self, enhanced_handler, mock_clients, caplog):
        """Test incremental WAV conversion reports missing declared audio bytes."""
        _, tts_client = mock_clients
        pcm = b"\x00\x01" * 240
        wav_data = _pcm_wav_with_trailing_metadata(pcm)
        wav_params = parse_wav_header(wav_data)
        assert wav_params is not None
        data_end = wav_params[3] + len(pcm)
        truncated_wav = wav_data[: data_end - 10]

        class MockAsyncIterator:
            def __init__(self, data):
                self.data = data
                self.done = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.done:
                    raise StopAsyncIteration
                self.done = True
                return self.data

        mock_response = Mock()
        mock_response.iter_bytes = Mock(return_value=MockAsyncIterator(truncated_wav))
        mock_stream_response = AsyncMock()
        mock_stream_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_response.__aexit__ = AsyncMock(return_value=None)
        tts_client.audio.speech.with_streaming_response.create = Mock(return_value=mock_stream_response)
        voice = enhanced_handler._get_voice("alloy")
        assert voice is not None

        timestamp = await _stream_whole_request(enhanced_handler, voice, "Hello world", send_audio_start=True)

        audio_chunk_events = [
            call.args[0] for call in enhanced_handler.write_event.call_args_list if call.args[0].type == "audio-chunk"
        ]
        assert b"".join(event.payload for event in audio_chunk_events) == pcm[:-10]
        assert timestamp == pytest.approx(470 / 2 / 24000 * 1000)
        assert "ended with 10 declared audio bytes missing" in caplog.text

    @pytest.mark.asyncio
    async def test_stream_tts_audio_accepts_unbounded_wav_data(self, enhanced_handler, mock_clients, caplog):
        """Test incremental WAV conversion accepts an unbounded data chunk size."""
        _, tts_client = mock_clients
        pcm = b"\x00\x01" * 240
        wav_data = _pcm_wav_with_unbounded_sizes(pcm)

        class MockAsyncIterator:
            def __init__(self, chunks):
                self.chunks = chunks
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.chunks):
                    raise StopAsyncIteration
                chunk = self.chunks[self.index]
                self.index += 1
                return chunk

        mock_response = Mock()
        mock_response.iter_bytes = Mock(return_value=MockAsyncIterator([wav_data[:60], wav_data[60:]]))
        mock_stream_response = AsyncMock()
        mock_stream_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_response.__aexit__ = AsyncMock(return_value=None)
        tts_client.audio.speech.with_streaming_response.create = Mock(return_value=mock_stream_response)
        voice = enhanced_handler._get_voice("alloy")
        assert voice is not None

        timestamp = await _stream_whole_request(enhanced_handler, voice, "Hello world", send_audio_start=True)

        audio_start_events = [
            call.args[0] for call in enhanced_handler.write_event.call_args_list if call.args[0].type == "audio-start"
        ]
        audio_chunk_events = [
            call.args[0] for call in enhanced_handler.write_event.call_args_list if call.args[0].type == "audio-chunk"
        ]
        assert len(audio_start_events) == 1
        assert audio_start_events[0].data == {"rate": 24000, "width": 2, "channels": 1, "timestamp": None}
        assert b"".join(event.payload for event in audio_chunk_events) == pcm
        assert [event.data["timestamp"] for event in audio_chunk_events] == [0, 0]
        assert timestamp == pytest.approx(10.0)
        assert "declared audio bytes missing" not in caplog.text

    @pytest.mark.asyncio
    async def test_handle_streaming_synthesis(self, enhanced_handler, mock_clients):
        """Test handling of streaming synthesis events."""
        _, tts_client = mock_clients

        # Create proper WAV data with header
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(b"\x00\x01" * 1000)
        wav_buffer.seek(0)
        mock_audio_data = wav_buffer.read()

        # Mock the streaming response
        class MockAsyncIterator:
            def __init__(self, data):
                self.data = data
                self.chunks = [data[i : i + 2048] for i in range(0, len(data), 2048)]
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.chunks):
                    raise StopAsyncIteration
                chunk = self.chunks[self.index]
                self.index += 1
                return chunk

        mock_response = Mock()
        mock_response.iter_bytes = Mock(return_value=MockAsyncIterator(mock_audio_data))

        mock_stream_response = AsyncMock()
        mock_stream_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_response.__aexit__ = AsyncMock(return_value=None)

        tts_client.audio.speech.with_streaming_response.create = Mock(return_value=mock_stream_response)

        # Start synthesis
        start_event = Event(type="synthesize-start", data={"voice": {"name": "alloy"}})
        result = await enhanced_handler.handle_event(start_event)
        assert result is True

        # Send text chunks
        chunk1_event = Event(type="synthesize-chunk", data={"text": "Hello "})
        result = await enhanced_handler.handle_event(chunk1_event)
        assert result is True

        chunk2_event = Event(type="synthesize-chunk", data={"text": "world"})
        result = await enhanced_handler.handle_event(chunk2_event)
        assert result is True

        # Clear previous write_event calls
        enhanced_handler.write_event.reset_mock()

        # Stop synthesis - just confirms completion
        stop_event = Event(type="synthesize-stop")
        result = await enhanced_handler.handle_event(stop_event)
        assert result is True

        # For non-streaming TTS voices (default mock behavior), the TTS client should be called
        # to synthesize the accumulated text using our non-streaming fallback
        tts_client.audio.speech.with_streaming_response.create.assert_called_once_with(
            model="tts-1",
            voice="alloy",
            input="Hello world",
            response_format="wav",
            speed=1.0,
            instructions="Test instructions",
        )

        # Verify completion and audio events were written
        event_types = [call[0][0].type for call in enhanced_handler.write_event.call_args_list]
        assert "synthesize-stopped" in event_types  # Confirms streaming synthesis completion
        assert "audio-start" in event_types  # Audio should be generated
        assert "audio-stop" in event_types

    @pytest.mark.asyncio
    async def test_streaming_synthesis_preserves_whitespace_in_retained_sentence(self, handler):
        """Keep token whitespace when a partial sentence spans synthesis chunks."""
        handler._wyoming_info.tts[0].supports_synthesize_streaming = True
        handler._process_ready_sentences = AsyncMock(return_value=True)

        assert await handler.handle_event(Event(type="synthesize-start", data={"voice": {"name": "voice1"}})) is True
        assert await handler.handle_event(Event(type="synthesize-chunk", data={"text": "First sentence. "})) is True
        assert await handler.handle_event(Event(type="synthesize-chunk", data={"text": "Todo "})) is True
        # yasbd keeps the inter-sentence space but attributes it to the retained tail
        assert handler._synthesis.text_accumulator == " Todo "

        assert await handler.handle_event(Event(type="synthesize-chunk", data={"text": "listo."})) is True
        assert handler._synthesis.text_accumulator == " Todo listo."
        handler._process_ready_sentences.assert_awaited_once_with(["First sentence."])

    @pytest.mark.asyncio
    async def test_directly_streamed_sentence_includes_configured_tts_extra_body(
        self, enhanced_handler, mock_clients
    ):
        """Test the request of a sentence that is streamed as it arrives includes configured extra_body."""
        _, tts_client = mock_clients
        enhanced_handler._tts_extra_body = {"response_format": "pcm"}

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(b"\x00\x01" * 1000)
        wav_buffer.seek(0)
        mock_audio_data = wav_buffer.read()

        class MockAsyncIterator:
            def __init__(self, data):
                self.data = data
                self.chunks = [data[i : i + 2048] for i in range(0, len(data), 2048)]
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.chunks):
                    raise StopAsyncIteration
                chunk = self.chunks[self.index]
                self.index += 1
                return chunk

        mock_response = Mock()
        mock_response.iter_bytes = Mock(return_value=MockAsyncIterator(mock_audio_data))

        mock_stream_response = AsyncMock()
        mock_stream_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_response.__aexit__ = AsyncMock(return_value=None)

        tts_client.audio.speech.with_streaming_response.create = Mock(return_value=mock_stream_response)

        voice = enhanced_handler._get_voice("alloy")
        assert voice is not None

        enhanced_handler._allow_streaming_task_id = "sentence_0"
        result = await enhanced_handler._get_tts_audio_stream("Hello world", voice, task_id="sentence_0")
        assert result.streamed

        call_args = tts_client.audio.speech.with_streaming_response.create.call_args.kwargs
        assert call_args["extra_body"] == {"response_format": "pcm"}

    @pytest.mark.asyncio
    async def test_handle_synthesize_uses_backend_voice_name_for_conflicts(
        self, enhanced_handler, mock_clients, mock_info
    ):
        """Test synthesis uses the backend voice token when public names are model-specific."""
        _, tts_client = mock_clients

        conflicting_voice = Mock()
        conflicting_voice.name = "alloy (tts-1)"
        conflicting_voice.backend_voice_name = "alloy"
        conflicting_voice.description = "alloy (tts-1)"
        conflicting_voice.languages = ["en"]
        conflicting_voice.model_name = "tts-1"
        mock_info.tts[0].voices = [conflicting_voice]

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(b"\x00\x01" * 1000)
        wav_buffer.seek(0)
        mock_audio_data = wav_buffer.read()

        class MockAsyncIterator:
            def __init__(self, data):
                self.data = data
                self.chunks = [data[i : i + 2048] for i in range(0, len(data), 2048)]
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.chunks):
                    raise StopAsyncIteration
                chunk = self.chunks[self.index]
                self.index += 1
                return chunk

        mock_response = Mock()
        mock_response.iter_bytes = Mock(return_value=MockAsyncIterator(mock_audio_data))

        mock_stream_response = AsyncMock()
        mock_stream_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_response.__aexit__ = AsyncMock(return_value=None)

        tts_client.audio.speech.with_streaming_response.create = Mock(return_value=mock_stream_response)

        event = Event(
            type="synthesize",
            data={"text": "Hello world", "voice": {"name": "alloy (tts-1)"}, "raw_text": "Hello world"},
        )

        assert await enhanced_handler.handle_event(event) is True

        call_args = tts_client.audio.speech.with_streaming_response.create.call_args.kwargs
        assert call_args["voice"] == "alloy"

    @pytest.mark.asyncio
    async def test_handle_synthesize_falls_back_to_voice_name_when_backend_voice_name_missing(
        self, enhanced_handler, mock_clients, mock_info
    ):
        """Test synthesis remains compatible with plain Wyoming TtsVoice objects."""
        _, tts_client = mock_clients

        legacy_voice = TtsVoice(
            name="legacy-voice",
            description="Legacy voice",
            attribution=Attribution(name="Test", url="https://example.com"),
            installed=True,
            languages=["en"],
            version=None,
        )
        legacy_voice.model_name = "tts-1"  # type: ignore[reportAttributeAccessIssue]
        mock_info.tts[0].voices = [legacy_voice]

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(b"\x00\x01" * 1000)
        wav_buffer.seek(0)
        mock_audio_data = wav_buffer.read()

        class MockAsyncIterator:
            def __init__(self, data):
                self.data = data
                self.chunks = [data[i : i + 2048] for i in range(0, len(data), 2048)]
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.chunks):
                    raise StopAsyncIteration
                chunk = self.chunks[self.index]
                self.index += 1
                return chunk

        mock_response = Mock()
        mock_response.iter_bytes = Mock(return_value=MockAsyncIterator(mock_audio_data))

        mock_stream_response = AsyncMock()
        mock_stream_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_response.__aexit__ = AsyncMock(return_value=None)

        tts_client.audio.speech.with_streaming_response.create = Mock(return_value=mock_stream_response)

        event = Event(
            type="synthesize",
            data={"text": "Hello world", "voice": {"name": "legacy-voice"}, "raw_text": "Hello world"},
        )

        assert await enhanced_handler.handle_event(event) is True

        call_args = tts_client.audio.speech.with_streaming_response.create.call_args.kwargs
        assert call_args["voice"] == "legacy-voice"

    def test_validate_tts_voice_and_language_falls_back_for_legacy_backend_voice(self, enhanced_handler, mock_info):
        """Test ambiguous raw backend voice names fall back to the first configured choice."""
        voice_a = Mock()
        voice_a.name = "glados (model-a)"
        voice_a.backend_voice_name = "glados"
        voice_a.languages = ["en"]
        voice_a.model_name = "model-a"

        voice_b = Mock()
        voice_b.name = "glados (model-b)"
        voice_b.backend_voice_name = "glados"
        voice_b.languages = ["en"]
        voice_b.model_name = "model-b"

        mock_info.tts[0].voices = [voice_a, voice_b]

        selected_voice = enhanced_handler._validate_tts_voice_and_language("glados", None)

        assert selected_voice is voice_a

    def test_validate_tts_voice_and_language_prefers_language_compatible_legacy_voice(
        self, enhanced_handler, mock_info
    ):
        """Test ambiguous raw backend voice names prefer a language-compatible match before falling back."""
        voice_a = Mock()
        voice_a.name = "glados (model-a)"
        voice_a.backend_voice_name = "glados"
        voice_a.languages = ["en"]
        voice_a.model_name = "model-a"

        voice_b = Mock()
        voice_b.name = "glados (model-b)"
        voice_b.backend_voice_name = "glados"
        voice_b.languages = ["fr"]
        voice_b.model_name = "model-b"

        mock_info.tts[0].voices = [voice_a, voice_b]

        selected_voice = enhanced_handler._validate_tts_voice_and_language("glados", "fr")

        assert selected_voice is voice_b

    @pytest.mark.asyncio
    async def test_handle_transcribe_with_streaming(self, enhanced_handler, mock_clients, mock_info):
        """Test handling of Transcribe event with streaming model."""
        stt_client, _ = mock_clients

        # Make model support streaming
        mock_info.asr[0].supports_transcript_streaming = True

        # For this test, just verify that the streaming path is attempted
        # by checking that create is called with stream=True
        stt_client.audio.transcriptions.create = AsyncMock(side_effect=Exception("Streaming test - expected"))

        # First send the transcribe event to set the model
        transcribe_event = Event(type="transcribe", data={"language": "en", "name": "whisper-1"})
        result = await enhanced_handler.handle_event(transcribe_event)
        assert result is True

        # Start recording
        start_event = Event(type="audio-start", data={"rate": 16000, "width": 2, "channels": 1})
        await enhanced_handler.handle_event(start_event)

        # Add some audio
        audio_data = b"\x00\x01" * 100
        chunk_event = Event(type="audio-chunk", data={"rate": 16000, "width": 2, "channels": 1}, payload=audio_data)
        await enhanced_handler.handle_event(chunk_event)

        # Stop recording - this triggers streaming transcription
        stop_event = Event(type="audio-stop")
        await enhanced_handler.handle_event(stop_event)

        # Verify that streaming transcription was attempted
        stt_client.audio.transcriptions.create.assert_called_once()
        call_args = stt_client.audio.transcriptions.create.call_args[1]
        assert call_args["stream"] is True  # Verify streaming was enabled

    @pytest.mark.asyncio
    async def test_handle_invalid_model(self, enhanced_handler):
        """Test handling of Transcribe event with invalid model."""
        event = Event(type="transcribe", data={"language": "en", "name": "invalid-model"})

        result = await enhanced_handler.handle_event(event)

        assert result is False

    @pytest.mark.asyncio
    async def test_handle_unsupported_language(self, enhanced_handler):
        """Test handling of Transcribe event with unsupported language."""
        event = Event(
            type="transcribe",
            data={
                "language": "zh",  # Not in supported languages
                "name": "whisper-1",
            },
        )

        result = await enhanced_handler.handle_event(event)

        assert result is False
        assert enhanced_handler._current_asr_model is None
        assert enhanced_handler._current_language is None

    @pytest.mark.asyncio
    async def test_unsupported_language_does_not_call_transcription_create(self, enhanced_handler, mock_clients):
        """Test that rejected transcription requests do not reach the STT backend."""
        stt_client, _ = mock_clients
        stt_client.audio.transcriptions.create = AsyncMock()

        result = await enhanced_handler.handle_event(
            Event(type="transcribe", data={"language": "zh", "name": "whisper-1"})
        )

        assert result is False

        await enhanced_handler.handle_event(Event(type="audio-start", data={"rate": 16000, "width": 2, "channels": 1}))
        await enhanced_handler.handle_event(
            Event(type="audio-chunk", data={"rate": 16000, "width": 2, "channels": 1}, payload=b"\x00\x01" * 100)
        )
        await enhanced_handler.handle_event(Event(type="audio-stop"))

        stt_client.audio.transcriptions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalid_transcribe_clears_previous_request_state(self, enhanced_handler, mock_clients):
        """Test that a failed Transcribe request clears any previously accepted STT request."""
        stt_client, _ = mock_clients
        stt_client.audio.transcriptions.create = AsyncMock()

        valid_result = await enhanced_handler.handle_event(
            Event(type="transcribe", data={"language": "en", "name": "whisper-1"})
        )
        invalid_result = await enhanced_handler.handle_event(
            Event(type="transcribe", data={"language": "zh", "name": "whisper-1"})
        )

        assert valid_result is True
        assert invalid_result is False
        assert enhanced_handler._current_asr_model is None
        assert enhanced_handler._current_language is None

        await enhanced_handler.handle_event(Event(type="audio-start", data={"rate": 16000, "width": 2, "channels": 1}))
        await enhanced_handler.handle_event(
            Event(type="audio-chunk", data={"rate": 16000, "width": 2, "channels": 1}, payload=b"\x00\x01" * 100)
        )
        await enhanced_handler.handle_event(Event(type="audio-stop"))

        stt_client.audio.transcriptions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_audio_recording_workflow(self, enhanced_handler, mock_clients):
        """Test complete audio recording workflow."""
        stt_client, _ = mock_clients

        # Mock transcription response
        mock_transcription = Mock()
        mock_transcription.text = "Recorded audio transcription"
        stt_client.audio.transcriptions.create = AsyncMock(return_value=mock_transcription)

        # First set up transcription model
        transcribe_event = Event(type="transcribe", data={"language": "en", "name": "whisper-1"})
        await enhanced_handler.handle_event(transcribe_event)

        # Start recording
        start_event = Event(type="audio-start", data={"rate": 16000, "width": 2, "channels": 1})
        await enhanced_handler.handle_event(start_event)

        assert enhanced_handler._is_recording is True
        assert enhanced_handler._wav_buffer is not None

        # Send multiple audio chunks
        for i in range(5):
            chunk_data = bytes([i % 256] * 200)
            chunk_event = Event(type="audio-chunk", data={"rate": 16000, "width": 2, "channels": 1}, payload=chunk_data)
            await enhanced_handler.handle_event(chunk_event)

        # Stop recording - this triggers transcription
        with patch("wyoming_openai.handler.isinstance") as mock_isinstance:

            def isinstance_side_effect(obj, class_or_tuple):
                if obj is mock_transcription:
                    from openai.types.audio.transcription_create_response import TranscriptionCreateResponse

                    return class_or_tuple is TranscriptionCreateResponse
                return builtins.isinstance(obj, class_or_tuple)

            mock_isinstance.side_effect = isinstance_side_effect

            stop_event = Event(type="audio-stop")
            await enhanced_handler.handle_event(stop_event)

        # Verify final state
        assert enhanced_handler._is_recording is False
        stt_client.audio.transcriptions.create.assert_called_once()

    def test_helper_methods(self, enhanced_handler):
        """Test various helper methods."""
        # Test _get_asr_model
        model = enhanced_handler._get_asr_model("whisper-1")
        assert model is not None
        assert model.name == "whisper-1"

        # Test invalid model
        invalid_model = enhanced_handler._get_asr_model("invalid")
        assert invalid_model is None

        # Test _get_voice
        voice = enhanced_handler._get_voice("alloy")
        assert voice is not None
        assert voice.name == "alloy"
        assert voice.backend_voice_name == "alloy"

        # Test invalid voice
        invalid_voice = enhanced_handler._get_voice("invalid")
        assert invalid_voice is None

        # Test _is_asr_model_streaming
        assert enhanced_handler._is_asr_model_streaming("whisper-1") is False

        # Test language support
        model = enhanced_handler._get_asr_model("whisper-1")
        assert enhanced_handler._is_asr_language_supported("en", model) is True
        assert enhanced_handler._is_asr_language_supported("zh", model) is False

        voice = enhanced_handler._get_voice("alloy")
        assert enhanced_handler._validate_tts_language("en", voice) is True
        assert enhanced_handler._validate_tts_language("fr", voice) is False


@pytest.fixture
def multi_program_info():
    """Real Info built by the compatibility factories, with two programs per domain."""
    asr_programs = create_asr_programs(["whisper-1"], ["gpt-4o-transcribe"], "http://stt.test", ["en"])
    voices = create_tts_voices(["tts-1"], ["gpt-4o-mini-tts"], ["alloy"], "http://tts.test", ["en"])
    tts_programs = create_tts_programs(voices, ["gpt-4o-mini-tts"])
    return create_info(asr_programs, tts_programs)


@pytest.fixture
def multi_program_handler(multi_program_info, dummy_clients, dummy_reader_writer):
    stt_client, tts_client = dummy_clients
    reader, writer = dummy_reader_writer
    handler = OpenAIEventHandler(
        reader,
        writer,
        info=multi_program_info,
        stt_client=stt_client,
        tts_client=tts_client,
    )
    handler.write_event = AsyncMock()
    return handler


@pytest.fixture
def distinct_names_handler(multi_program_info, dummy_clients, dummy_reader_writer):
    """Handler whose program names are unique within a domain (asr-*/tts-*).

    Exercises cross-domain selection, where a name can never match both
    domains in one event.
    """
    asr_programs = list(multi_program_info.asr)
    tts_programs = list(multi_program_info.tts)
    for program in asr_programs:
        program.name = f"asr-{program.name}"
    for program in tts_programs:
        program.name = f"tts-{program.name}"
    stt_client, tts_client = dummy_clients
    reader, writer = dummy_reader_writer
    handler = OpenAIEventHandler(
        reader,
        writer,
        info=create_info(asr_programs, tts_programs),
        stt_client=stt_client,
        tts_client=tts_client,
    )
    handler.write_event = AsyncMock()
    return handler


@pytest.mark.asyncio
async def test_default_model_follows_a_program_selected_between_streams(multi_program_handler):
    """The default model of a stream without transcribe is resolved per stream, not kept from the first one."""
    from openai.types.audio.transcription import Transcription

    stt_client = multi_program_handler._stt_client
    stt_client.audio.transcriptions.create = AsyncMock(return_value=Transcription(text="Hello"))

    await _stream_audio(multi_program_handler)
    assert await multi_program_handler.handle_event(Event(type="select-program", data={"name": "openai"}))
    await _stream_audio(multi_program_handler)

    models = [call.kwargs["model"] for call in stt_client.audio.transcriptions.create.call_args_list]
    assert models == ["gpt-4o-transcribe", "whisper-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transcribed_model", "expected_model", "expected_language"),
    [("gpt-4o-transcribe", "whisper-1", omit), ("whisper-1", "whisper-1", "en")],
)
async def test_select_program_drops_a_transcribe_selection_it_does_not_offer(
    multi_program_handler, transcribed_model, expected_model, expected_language, caplog
):
    """A model selected by an earlier transcribe, with its language, only outlives a program that offers it."""
    from openai.types.audio.transcription import Transcription

    stt_client = multi_program_handler._stt_client
    stt_client.audio.transcriptions.create = AsyncMock(return_value=Transcription(text="Hello"))

    assert await multi_program_handler.handle_event(
        Event(type="transcribe", data={"name": transcribed_model, "language": "en"})
    )
    with caplog.at_level(logging.INFO, logger="wyoming_openai.handler"):
        assert await multi_program_handler.handle_event(Event(type="select-program", data={"name": "openai"}))
    # The language the client asked for is dropped with the model, which is logged where it can be seen
    assert ("Dropping model gpt-4o-transcribe and language en" in caplog.text) == (expected_language is omit)
    await _stream_audio(multi_program_handler)

    kwargs = stt_client.audio.transcriptions.create.call_args.kwargs
    assert (kwargs["model"], kwargs["language"]) == (expected_model, expected_language)
    # whisper-1 is not a streaming model, so its response is not requested as a stream
    assert kwargs["stream"] is omit


@pytest.mark.asyncio
async def test_program_selected_during_a_recording_keeps_its_streaming_response(multi_program_handler):
    """A program selected during a recording does not change how its response is requested."""
    from openai.types.audio.transcription import Transcription

    stt_client = multi_program_handler._stt_client
    stt_client.audio.transcriptions.create = AsyncMock(return_value=Transcription(text="Hello"))

    assert await multi_program_handler.handle_event(Event(type="transcribe", data={"name": "gpt-4o-transcribe"}))
    await multi_program_handler.handle_event(Event(type="audio-start", data={"rate": 16000, "width": 2, "channels": 1}))
    assert await multi_program_handler.handle_event(Event(type="select-program", data={"name": "openai"}))
    await multi_program_handler.handle_event(Event(type="audio-stop"))

    kwargs = stt_client.audio.transcriptions.create.call_args.kwargs
    assert (kwargs["model"], kwargs["stream"]) == ("gpt-4o-transcribe", True)


@pytest.mark.asyncio
async def test_late_transcribe_for_a_streaming_model_streams_the_response(multi_program_handler):
    """A recording switched to a streaming model during it requests a streamed response."""
    from openai.types.audio.transcription import Transcription

    stt_client = multi_program_handler._stt_client
    stt_client.audio.transcriptions.create = AsyncMock(return_value=Transcription(text="Hello"))

    # The recording starts with the default model, whisper-1, whose program does not stream
    await _stream_audio(multi_program_handler, transcribe={"name": "gpt-4o-transcribe"})

    kwargs = stt_client.audio.transcriptions.create.call_args.kwargs
    assert (kwargs["model"], kwargs["stream"]) == ("gpt-4o-transcribe", True)


@pytest.mark.asyncio
async def test_select_program_picks_asr_program_by_name(multi_program_handler):
    result = await multi_program_handler.handle_event(Event(type="select-program", data={"name": "openai"}))

    assert result is True
    assert multi_program_handler._selected_asr_program is not None
    assert multi_program_handler._selected_asr_program.name == "openai"
    # "openai" is a program name in both domains, so one event selects both
    assert multi_program_handler._selected_tts_program is not None
    assert multi_program_handler._selected_tts_program.name == "openai"


@pytest.mark.asyncio
async def test_select_program_picks_tts_program_by_name(multi_program_handler):
    result = await multi_program_handler.handle_event(
        Event(type="select-program", data={"name": "openai-streaming"})
    )

    assert result is True
    assert multi_program_handler._selected_tts_program is not None
    assert multi_program_handler._selected_tts_program.name == "openai-streaming"
    assert multi_program_handler._get_default_voice().name == "alloy (gpt-4o-mini-tts)"


@pytest.mark.asyncio
async def test_select_program_unknown_name_is_dropped(multi_program_handler):
    result = await multi_program_handler.handle_event(
        Event(type="select-program", data={"name": "does-not-exist"})
    )

    assert result is True
    assert multi_program_handler._selected_asr_program is None
    assert multi_program_handler._selected_tts_program is None
    # Resolution still spans all programs
    assert multi_program_handler._get_asr_model("whisper-1") is not None
    assert multi_program_handler._get_voice("alloy (tts-1)") is not None


@pytest.mark.asyncio
async def test_select_program_selections_persist_independently_per_domain(distinct_names_handler):
    """select-program only affects matching domains: a later TTS-only event must
    not clear a prior ASR selection (or vice versa). Each selection stays active
    for the lifetime of the connection and constrains its own domain."""
    handler = distinct_names_handler

    # Select an ASR-only program, then a TTS-only program
    result = await handler.handle_event(Event(type="select-program", data={"name": "asr-openai"}))
    assert result is True
    assert handler._selected_asr_program is not None
    assert handler._selected_asr_program.name == "asr-openai"
    assert handler._selected_tts_program is None

    result = await handler.handle_event(Event(type="select-program", data={"name": "tts-openai"}))
    assert result is True
    assert handler._selected_asr_program is not None
    assert handler._selected_asr_program.name == "asr-openai"
    assert handler._selected_tts_program is not None
    assert handler._selected_tts_program.name == "tts-openai"

    # Each selection still constrains its own domain
    result = await handler.handle_event(
        Event(type="transcribe", data={"name": "whisper-1", "language": "en"})
    )
    assert result is True
    assert handler._current_asr_model.name == "whisper-1"

    assert handler._get_voice("alloy (gpt-4o-mini-tts)") is None


@pytest.fixture
def streaming_collision_info():
    """Info where a model/voice name exists in both a streaming and a non-streaming program.

    The streaming program is listed first, so a name-only scan returns the
    streaming flag even when the non-streaming program is selected.
    """
    asr_programs = create_asr_programs(["shared"], ["shared"], "http://stt.test", ["en"]) + create_asr_programs(
        ["shared"], [], "http://stt.test", ["en"]
    )
    streaming_voices = create_tts_voices(["tts-1"], ["tts-1"], ["alloy"], "http://tts.test", ["en"])
    non_streaming_voices = create_tts_voices(["tts-1"], [], ["alloy"], "http://tts.test", ["en"])
    tts_programs = create_tts_programs(streaming_voices, ["tts-1"]) + create_tts_programs(non_streaming_voices)
    return create_info(asr_programs, tts_programs)


@pytest.fixture
def streaming_collision_handler(streaming_collision_info, dummy_clients, dummy_reader_writer):
    stt_client, tts_client = dummy_clients
    reader, writer = dummy_reader_writer
    handler = OpenAIEventHandler(
        reader,
        writer,
        info=streaming_collision_info,
        stt_client=stt_client,
        tts_client=tts_client,
    )
    handler.write_event = AsyncMock()
    return handler


@pytest.mark.asyncio
async def test_streaming_flag_follows_selected_asr_program(streaming_collision_handler):
    handler = streaming_collision_handler

    # Without selection the first name match across programs wins (pre-existing)
    assert handler._is_asr_model_streaming("shared") is True

    # Selecting the non-streaming program must not pick up the streaming program's flag
    await handler.handle_event(Event(type="select-program", data={"name": "openai"}))
    assert handler._is_asr_model_streaming("shared") is False

    await handler.handle_event(Event(type="select-program", data={"name": "openai-streaming"}))
    assert handler._is_asr_model_streaming("shared") is True


@pytest.mark.asyncio
async def test_streaming_flag_follows_selected_tts_program(streaming_collision_handler):
    handler = streaming_collision_handler

    assert handler._is_tts_voice_streaming("alloy") is True

    # Selecting the non-streaming program must not pick up the streaming program's flag
    await handler.handle_event(Event(type="select-program", data={"name": "openai"}))
    assert handler._is_tts_voice_streaming("alloy") is False

    await handler.handle_event(Event(type="select-program", data={"name": "openai-streaming"}))
    assert handler._is_tts_voice_streaming("alloy") is True


@pytest.mark.asyncio
async def test_select_program_then_transcribe_resolves_within_selected_program(multi_program_handler):
    await multi_program_handler.handle_event(Event(type="select-program", data={"name": "openai"}))

    # Model from the non-selected "openai-streaming" program is rejected
    result = await multi_program_handler.handle_event(
        Event(type="transcribe", data={"name": "gpt-4o-transcribe", "language": "en"})
    )
    assert result is False

    # Nameless transcribe resolves to the selected program's first model
    result = await multi_program_handler.handle_event(Event(type="transcribe", data={"language": "en"}))
    assert result is True
    assert multi_program_handler._current_asr_model.name == "whisper-1"


@pytest.mark.asyncio
async def test_no_select_program_defaults_to_first_program(multi_program_handler):
    result = await multi_program_handler.handle_event(Event(type="transcribe", data={"language": "en"}))

    assert result is True
    assert multi_program_handler._current_asr_model.name == "gpt-4o-transcribe"
    assert multi_program_handler._get_default_voice().name == "alloy (gpt-4o-mini-tts)"


@pytest.mark.asyncio
async def test_transcribe_vad_sensitivity_logged_and_ignored(enhanced_handler, caplog):
    caplog.set_level(logging.DEBUG, logger="wyoming_openai.handler")

    result = await enhanced_handler.handle_event(
        Event(type="transcribe", data={"name": "whisper-1", "language": "en", "vad_sensitivity": "aggressive"})
    )

    assert result is True
    assert enhanced_handler._current_asr_model is not None
    assert "vad_sensitivity" in caplog.text


@pytest.mark.asyncio
async def test_transcribe_transcript_names_logged_and_ignored(enhanced_handler, caplog):
    caplog.set_level(logging.DEBUG, logger="wyoming_openai.handler")

    result = await enhanced_handler.handle_event(
        Event(
            type="transcribe",
            data={
                "name": "whisper-1",
                "language": "en",
                "transcript_names": ["Alice"],
                "transcript_terms": ["Kubernetes"],
            },
        )
    )

    assert result is True
    assert "transcript_names" in caplog.text
    assert "transcript_terms" in caplog.text


@pytest.mark.asyncio
async def test_synthesize_ssml_strips_tags_before_sending(enhanced_handler):
    enhanced_handler._stream_tts_audio = AsyncMock(return_value=123.0)

    result = await enhanced_handler.handle_event(
        Event(
            type="synthesize",
            data={
                "text": '<speak>Hello <break time="1s"/>world</speak>',
                "text_format": "ssml",
                "voice": {"name": "alloy"},
            },
        )
    )

    assert result is True
    enhanced_handler._stream_tts_audio.assert_awaited_once()
    assert enhanced_handler._stream_tts_audio.await_args_list[0].args[1] == "Hello world"


@pytest.mark.asyncio
async def test_synthesize_text_format_text_passes_through(enhanced_handler):
    enhanced_handler._stream_tts_audio = AsyncMock(return_value=123.0)
    text = "Hello <break/>world"

    result = await enhanced_handler.handle_event(
        Event(type="synthesize", data={"text": text, "text_format": "text", "voice": {"name": "alloy"}})
    )

    assert result is True
    assert enhanced_handler._stream_tts_audio.await_args_list[0].args[1] == text


@pytest.mark.asyncio
async def test_synthesize_text_format_none_passes_through(enhanced_handler):
    enhanced_handler._stream_tts_audio = AsyncMock(return_value=123.0)
    text = "Hello <break/>world"

    result = await enhanced_handler.handle_event(
        Event(type="synthesize", data={"text": text, "voice": {"name": "alloy"}})
    )

    assert result is True
    assert enhanced_handler._stream_tts_audio.await_args_list[0].args[1] == text


@pytest.mark.asyncio
async def test_synthesize_start_ssml_strips_chunk_tags(enhanced_handler):
    result = await enhanced_handler.handle_event(
        Event(
            type="synthesize-start",
            data={"text_format": "ssml", "voice": {"name": "alloy", "language": "en"}},
        )
    )
    assert result is True
    assert enhanced_handler._synthesis.ssml_transformer is not None

    result = await enhanced_handler.handle_event(
        Event(type="synthesize-chunk", data={"text": "Hello <break/>world"})
    )

    assert result is True
    assert enhanced_handler._synthesis.text_accumulator == "Hello world"
    assert enhanced_handler._synthesis.buffer == ["Hello world"]


@pytest.mark.asyncio
async def test_synthesize_chunk_ssml_split_speak_and_break(enhanced_handler):
    await enhanced_handler.handle_event(
        Event(
            type="synthesize-start",
            data={"text_format": "ssml", "voice": {"name": "alloy", "language": "en"}},
        )
    )

    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": "<speak>Hello"}))
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": "<break/>world</speak>"}))

    assert "".join(enhanced_handler._synthesis.buffer) == "Hello world"
    assert enhanced_handler._synthesis.text_accumulator == "Hello world"


@pytest.mark.asyncio
async def test_synthesize_chunk_ssml_tag_split_across_chunks(enhanced_handler):
    await enhanced_handler.handle_event(
        Event(
            type="synthesize-start",
            data={"text_format": "ssml", "voice": {"name": "alloy", "language": "en"}},
        )
    )

    # Tag split mid-name across chunk boundaries must not leak markup
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": "<speak>Hello <em"}))
    await enhanced_handler.handle_event(
        Event(type="synthesize-chunk", data={"text": "phasis>world</emphasis></speak>"})
    )

    assert enhanced_handler._synthesis.text_accumulator == "Hello world"
    assert "".join(enhanced_handler._synthesis.buffer) == "Hello world"


@pytest.mark.asyncio
async def test_synthesize_chunk_ssml_tag_split_across_three_chunks(enhanced_handler):
    await enhanced_handler.handle_event(
        Event(
            type="synthesize-start",
            data={"text_format": "ssml", "voice": {"name": "alloy", "language": "en"}},
        )
    )

    for chunk in ("Hello<br", "ea", "k/>world"):
        await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": chunk}))

    assert enhanced_handler._synthesis.text_accumulator == "Hello world"
    assert "".join(enhanced_handler._synthesis.buffer) == "Hello world"


@pytest.mark.asyncio
async def test_synthesize_chunk_ssml_entity_split_across_chunks(enhanced_handler):
    await enhanced_handler.handle_event(
        Event(
            type="synthesize-start",
            data={"text_format": "ssml", "voice": {"name": "alloy", "language": "en"}},
        )
    )

    # An entity split mid-name across chunk boundaries must be reassembled and
    # decoded, not left as a literal fragment
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": "Tom &am"}))
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": "p; Jerry"}))

    assert enhanced_handler._synthesis.text_accumulator == "Tom & Jerry"
    assert "".join(enhanced_handler._synthesis.buffer) == "Tom & Jerry"


@pytest.mark.asyncio
async def test_synthesize_chunk_ssml_bare_ampersand_at_boundary(enhanced_handler):
    await enhanced_handler.handle_event(
        Event(
            type="synthesize-start",
            data={"text_format": "ssml", "voice": {"name": "alloy", "language": "en"}},
        )
    )

    # A literal ampersand ending a chunk is held as a pending candidate; when
    # the next chunk cannot extend it to a valid entity, it must be stripped
    # separately so html.unescape cannot decode across the boundary
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": "Rules &"}))
    assert enhanced_handler._synthesis.text_accumulator == "Rules"

    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": "regulations"}))

    assert enhanced_handler._synthesis.text_accumulator == "Rules &regulations"
    assert "".join(enhanced_handler._synthesis.buffer) == "Rules &regulations"


@pytest.mark.asyncio
async def test_synthesize_chunk_ssml_entity_prefix_then_plain_text(enhanced_handler):
    await enhanced_handler.handle_event(
        Event(
            type="synthesize-start",
            data={"text_format": "ssml", "voice": {"name": "alloy", "language": "en"}},
        )
    )

    # An entity-looking pending token rejected by the next chunk must keep every
    # character: html.unescape may shorten the prefix, so the result must not
    # be sliced by a previous chunk length
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": "Foo &amp"}))

    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": "regulations"}))

    assert enhanced_handler._synthesis.text_accumulator == "Foo &ampregulations"
    assert "".join(enhanced_handler._synthesis.buffer) == "Foo &ampregulations"


@pytest.mark.asyncio
async def test_synthesize_chunk_ssml_entity_split_after_ampersand(enhanced_handler):
    await enhanced_handler.handle_event(
        Event(
            type="synthesize-start",
            data={"text_format": "ssml", "voice": {"name": "alloy", "language": "en"}},
        )
    )

    # An entity split immediately after "&" must still be reassembled and decoded
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": "Tom &"}))
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": "amp; Jerry"}))

    assert enhanced_handler._synthesis.text_accumulator == "Tom & Jerry"
    assert "".join(enhanced_handler._synthesis.buffer) == "Tom & Jerry"


@pytest.mark.asyncio
async def test_synthesize_chunk_ssml_bare_ampersand_preserves_tag_boundary(enhanced_handler):
    await enhanced_handler.handle_event(
        Event(
            type="synthesize-start",
            data={"text_format": "ssml", "voice": {"name": "alloy", "language": "en"}},
        )
    )

    # A rejected "&" token followed by an unknown tag must keep the tag's word
    # boundary: the malformed fallback would produce "Foo& bar" for the
    # combined text, so the split path must not join the words
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": "Foo&"}))

    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": "<vendor/>bar"}))

    assert enhanced_handler._synthesis.text_accumulator == "Foo& bar"
    assert "".join(enhanced_handler._synthesis.buffer) == "Foo& bar"


@pytest.mark.asyncio
async def test_synthesize_chunk_ssml_completed_entity_preserves_tag_boundary(enhanced_handler):
    await enhanced_handler.handle_event(
        Event(
            type="synthesize-start",
            data={"text_format": "ssml", "voice": {"name": "alloy", "language": "en"}},
        )
    )

    # A pending entity completed by the next chunk followed by an unknown tag
    # must keep the tag's word boundary: the malformed fallback would produce
    # "Foo & bar" for the combined text, so the seam must not join the words
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": "Foo &amp"}))

    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": ";<vendor>bar"}))

    assert enhanced_handler._synthesis.text_accumulator == "Foo & bar"
    assert "".join(enhanced_handler._synthesis.buffer) == "Foo & bar"


@pytest.mark.asyncio
async def test_synthesize_chunk_ssml_entity_then_split_tag_preserves_boundary(enhanced_handler):
    await enhanced_handler.handle_event(
        Event(
            type="synthesize-start",
            data={"text_format": "ssml", "voice": {"name": "alloy", "language": "en"}},
        )
    )

    # A pending entity completed by the next chunk that then starts a tag split
    # across the following chunk must keep the tag's word boundary: the
    # malformed fallback would produce "Foo & bar" for the combined text, so
    # the seams must not join the words
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": "Foo &amp"}))
    assert enhanced_handler._synthesis.text_accumulator == "Foo"

    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": ";<vendor"}))
    assert enhanced_handler._synthesis.text_accumulator == "Foo &"

    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": ">bar"}))

    assert enhanced_handler._synthesis.text_accumulator == "Foo & bar"
    assert "".join(enhanced_handler._synthesis.buffer) == "Foo & bar"


@pytest.mark.asyncio
async def test_synthesize_chunk_ssml_normalizes_spaces_across_chunks(enhanced_handler):
    await enhanced_handler.handle_event(
        Event(
            type="synthesize-start",
            data={"text_format": "ssml", "voice": {"name": "alloy", "language": "en"}},
        )
    )

    # The stripped stream must match one-shot stripping regardless of where the
    # client split the text around a pause tag
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": "Hello "}))
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": "<break/>world"}))

    assert enhanced_handler._synthesis.text_accumulator == "Hello world"
    assert "".join(enhanced_handler._synthesis.buffer) == "Hello world"


@pytest.mark.asyncio
async def test_synthesize_chunk_ssml_split_pause_tag_normalizes_spaces(enhanced_handler):
    await enhanced_handler.handle_event(
        Event(
            type="synthesize-start",
            data={"text_format": "ssml", "voice": {"name": "alloy", "language": "en"}},
        )
    )

    # A pause tag split mid-name still contributes exactly one separator
    for chunk in ("Hello ", "<bre", "ak/> world"):
        await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": chunk}))

    assert enhanced_handler._synthesis.text_accumulator == "Hello world"
    assert "".join(enhanced_handler._synthesis.buffer) == "Hello world"


@pytest.mark.asyncio
async def test_synthesize_chunk_ssml_normalizes_spaces_before_split_tag(enhanced_handler):
    await enhanced_handler.handle_event(
        Event(
            type="synthesize-start",
            data={"text_format": "ssml", "voice": {"name": "alloy", "language": "en"}},
        )
    )

    # The leading space must be normalized even while the trailing tag is held.
    for chunk in ("Hello ", " <voice", ">world"):
        await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": chunk}))

    assert enhanced_handler._synthesis.text_accumulator == "Hello world"
    assert "".join(enhanced_handler._synthesis.buffer) == "Hello world"


@pytest.mark.asyncio
async def test_synthesize_chunk_plain_ssml_spaces_normalize_across_chunks(enhanced_handler):
    await enhanced_handler.handle_event(
        Event(
            type="synthesize-start",
            data={"text_format": "ssml", "voice": {"name": "alloy", "language": "en"}},
        )
    )

    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": "Hello "}))
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": " world"}))

    assert enhanced_handler._synthesis.text_accumulator == "Hello world"
    assert "".join(enhanced_handler._synthesis.buffer) == "Hello world"


@pytest.mark.asyncio
async def test_synthesize_chunk_ssml_normalizes_spaces_after_sentence_flush(enhanced_handler):
    await enhanced_handler.handle_event(
        Event(
            type="synthesize-start",
            data={"text_format": "ssml", "voice": {"name": "alloy", "language": "en"}},
        )
    )

    # Sentence detection retains only the final segment, which drops the
    # trailing space still present in the emitted synthesis buffer.
    segmenter = MagicMock()
    segmenter.segment.side_effect = [["First.", "Second."], ["Second. Next"]]
    enhanced_handler._process_ready_sentences = AsyncMock(return_value=True)

    with patch("wyoming_openai.handler._get_boundary_detector", return_value=segmenter):
        await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": "First. Second. "}))
        assert enhanced_handler._synthesis.text_accumulator == "Second."

        await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": " Next"}))

    assert "".join(enhanced_handler._synthesis.buffer) == "First. Second. Next"
    assert segmenter.segment.call_args_list[1].args == ("Second. Next",)
    assert enhanced_handler._synthesis.text_accumulator == "Second. Next"


@pytest.mark.asyncio
async def test_synthesize_chunk_ssml_long_tag_split_across_chunks(enhanced_handler):
    await enhanced_handler.handle_event(
        Event(
            type="synthesize-start",
            data={"text_format": "ssml", "voice": {"name": "alloy", "language": "en"}},
        )
    )

    # A valid tag longer than the old carry limit must still be
    # reassembled and stripped; one-shot stripping keeps the literal space
    tag = '<audio src="https://example.com/' + ("a" * 50) + '"/>'
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": "Hello " + tag[:-2]}))
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": tag[-2:] + "world"}))

    assert enhanced_handler._synthesis.text_accumulator == "Hello world"
    assert "".join(enhanced_handler._synthesis.buffer) == "Hello world"


@pytest.mark.asyncio
async def test_synthesize_chunk_ssml_long_tag_growing_past_previous_limit(enhanced_handler):
    await enhanced_handler.handle_event(
        Event(
            type="synthesize-start",
            data={"text_format": "ssml", "voice": {"name": "alloy", "language": "en"}},
        )
    )

    # A valid fragment longer than the old carry limit must still be completed
    # by the next chunks is still reassembled as one tag
    tag = '<audio src="https://example.com/' + ("a" * 50) + '"/>'
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": tag[:64]}))
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": tag[64:65]}))
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": tag[65:] + "world"}))

    assert enhanced_handler._synthesis.text_accumulator == "world"
    assert "".join(enhanced_handler._synthesis.buffer) == "world"


@pytest.mark.asyncio
async def test_synthesize_chunk_ssml_quoted_tag_close_waits_for_real_tag_end(enhanced_handler):
    await enhanced_handler.handle_event(
        Event(
            type="synthesize-start",
            data={"text_format": "ssml", "voice": {"name": "alloy", "language": "en"}},
        )
    )

    # ">" inside a quoted attribute is not the tag end
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": '<voice name="a>'}))
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": 'b"/>spoken'}))

    assert enhanced_handler._synthesis.text_accumulator == "spoken"
    assert "".join(enhanced_handler._synthesis.buffer) == "spoken"


@pytest.mark.asyncio
async def test_synthesize_chunk_ssml_unclosed_tag_is_bounded(enhanced_handler):
    await enhanced_handler.handle_event(
        Event(
            type="synthesize-start",
            data={"text_format": "ssml", "voice": {"name": "alloy", "language": "en"}},
        )
    )

    # A client sending an unclosed "<" followed by endless text must not grow
    # the pending token (and its per-chunk rescan/copy) without bound: after
    # the cap it is released and subsequent chunks stream normally
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": "<"}))

    flooded = "x" * 5000
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": flooded}))
    assert "<" + flooded in "".join(enhanced_handler._synthesis.buffer)

    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": "more text"}))


@pytest.mark.asyncio
async def test_synthesize_chunk_ssml_unterminated_numeric_entity_is_bounded(enhanced_handler):
    await enhanced_handler.handle_event(
        Event(
            type="synthesize-start",
            data={"text_format": "ssml", "voice": {"name": "alloy", "language": "en"}},
        )
    )

    # A numeric entity prefix never terminated by ";" must not grow unbounded
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": "&#123"}))

    flooded = "4" * 5000
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": flooded}))
    assert "&#123" + flooded in "".join(enhanced_handler._synthesis.buffer)


@pytest.mark.asyncio
async def test_synthesize_chunk_ssml_oversized_token_flushes_literally_at_stop(enhanced_handler):
    enhanced_handler.write_event = AsyncMock()
    await enhanced_handler.handle_event(
        Event(
            type="synthesize-start",
            data={"text_format": "ssml", "voice": {"name": "alloy", "language": "en"}},
        )
    )

    # A fragment held at the cap and released at stop is literal text, not decoded
    fragment = "<" + "x" * 5000
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": fragment}))

    enhanced_handler._process_ready_sentences = AsyncMock(return_value=True)
    enhanced_handler._synthesis.audio_started = True  # take the incremental early-exit path
    await enhanced_handler.handle_event(Event(type="synthesize-stop", data={}))
    flushed_text = enhanced_handler._process_ready_sentences.await_args_list[0].args[0]
    assert flushed_text == [fragment]


@pytest.mark.asyncio
async def test_synthesize_chunk_ssml_complete_entity_decodes_despite_oversized_split_tag(enhanced_handler):
    await enhanced_handler.handle_event(
        Event(
            type="synthesize-start",
            data={"text_format": "ssml", "voice": {"name": "alloy", "language": "en"}},
        )
    )

    # A pending "&amp" completed by ";" stays decoded: an oversized trailing
    # tag split must not re-mark the already-complete entity as literal text
    # ("&amp;<..." is emitted, not "<...>"), even though the merged fragment
    # crossed the token bound
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": "&amp"}))

    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": ";<"}))

    flood = "x" * 4096
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": flood}))

    assert enhanced_handler._synthesis.text_accumulator == "&" + "<" + flood
    assert "".join(enhanced_handler._synthesis.buffer) == "&" + "<" + flood


@pytest.mark.asyncio
async def test_synthesize_stop_flushes_pending_ssml_token(enhanced_handler):
    enhanced_handler.write_event = AsyncMock()
    await enhanced_handler.handle_event(
        Event(
            type="synthesize-start",
            data={"text_format": "ssml", "voice": {"name": "alloy", "language": "en"}},
        )
    )
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": "Hi <"}))

    # A trailing "<" never completed into a tag is literal text, not markup
    enhanced_handler._process_ready_sentences = AsyncMock(return_value=True)
    enhanced_handler._synthesis.audio_started = True  # take the incremental early-exit path
    await enhanced_handler.handle_event(Event(type="synthesize-stop", data={}))

    flushed_text = enhanced_handler._process_ready_sentences.await_args_list[0].args[0]
    assert flushed_text == ["Hi <"]


@pytest.mark.asyncio
async def test_synthesize_stop_flushes_ssml_incomplete_entity(enhanced_handler):
    enhanced_handler.write_event = AsyncMock()
    await enhanced_handler.handle_event(
        Event(
            type="synthesize-start",
            data={"text_format": "ssml", "voice": {"name": "alloy", "language": "en"}},
        )
    )
    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": "Foo &amp"}))
    assert enhanced_handler._synthesis.text_accumulator == "Foo"

    # A trailing entity-like fragment never completed into an entity is literal
    # text; flushing must not decode it ("&amp" must not become "&")
    enhanced_handler._process_ready_sentences = AsyncMock(return_value=True)
    enhanced_handler._synthesis.audio_started = True  # take the incremental early-exit path
    await enhanced_handler.handle_event(Event(type="synthesize-stop", data={}))

    flushed_text = enhanced_handler._process_ready_sentences.await_args_list[0].args[0]
    assert flushed_text == ["Foo &amp"]


@pytest.mark.asyncio
async def test_no_select_program_explicit_cross_program_names_resolve(multi_program_handler):
    """Without select-program, every advertised model/voice stays addressable by
    explicit name across programs (deliberate: pre-1.10 clients cannot select)."""
    result = await multi_program_handler.handle_event(
        Event(type="transcribe", data={"name": "whisper-1", "language": "en"})
    )
    assert result is True
    assert multi_program_handler._current_asr_model.name == "whisper-1"

    assert multi_program_handler._get_voice("alloy (tts-1)") is not None


async def _transcribe_over_http(handler, stt_client, model_name):
    """Run a transcribe request through the HTTP path and return the request kwargs."""
    stt_client.audio.transcriptions.create = AsyncMock(side_effect=Exception("Request capture - expected"))

    assert await handler.handle_event(Event(type="transcribe", data={"language": "en", "name": model_name}))
    await handler.handle_event(Event(type="audio-start", data={"rate": 16000, "width": 2, "channels": 1}))
    await handler.handle_event(
        Event(type="audio-chunk", data={"rate": 16000, "width": 2, "channels": 1}, payload=b"\x00\x01" * 100)
    )
    await handler.handle_event(Event(type="audio-stop"))

    return stt_client.audio.transcriptions.create.call_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["gpt-transcribe", "gpt-transcribe-2026-07-28"])
async def test_transcribe_sends_plural_languages_for_new_openai_models(
    enhanced_handler, mock_info, mock_clients, model_name
):
    """Test gpt-transcribe requests use `languages` instead of `language`."""
    stt_client, _ = mock_clients
    stt_client.backend = OpenAIBackend.OPENAI
    mock_info.asr[0].models[0].name = model_name

    call_args = await _transcribe_over_http(enhanced_handler, stt_client, model_name)

    assert call_args["language"] is omit
    assert call_args["languages"] == ["en"]
    assert call_args["temperature"] == 0.5
    assert "extra_body" not in call_args
    assert call_args["prompt"] == "Test prompt"


@pytest.mark.asyncio
async def test_transcribe_plural_languages_extra_body_override_wins(enhanced_handler, mock_info, mock_clients):
    """Test a configured `languages` extra_body field is not replaced by the request language."""
    stt_client, _ = mock_clients
    stt_client.backend = OpenAIBackend.OPENAI
    mock_info.asr[0].models[0].name = "gpt-transcribe"
    enhanced_handler._stt_extra_body = {"languages": ["en", "fr"], "keywords": ["Wyoming"]}

    call_args = await _transcribe_over_http(enhanced_handler, stt_client, "gpt-transcribe")

    assert call_args["language"] is omit
    assert call_args["languages"] is omit
    assert call_args["extra_body"] == {"languages": ["en", "fr"], "keywords": ["Wyoming"]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "model_name"),
    [(OpenAIBackend.OPENAI, "whisper-1"), (OpenAIBackend.SPEACHES, "gpt-transcribe")],
)
async def test_transcribe_keeps_singular_language_for_other_models_and_backends(
    enhanced_handler, mock_info, mock_clients, backend, model_name
):
    """Test older OpenAI models and non-OpenAI backends keep `language` and `temperature`."""
    stt_client, _ = mock_clients
    stt_client.backend = backend
    mock_info.asr[0].models[0].name = model_name

    call_args = await _transcribe_over_http(enhanced_handler, stt_client, model_name)

    assert call_args["language"] == "en"
    assert call_args["temperature"] == 0.5
    assert call_args["languages"] is omit
    assert "languages" not in call_args.get("extra_body", {})


@pytest.mark.asyncio
async def test_realtime_transcription_session_uses_plural_languages(enhanced_handler, mock_info, mock_clients):
    """Test gpt-live-transcribe Realtime sessions use `languages` instead of `language`."""
    stt_client, _ = mock_clients
    stt_client.backend = OpenAIBackend.OPENAI
    mock_info.asr[0].models[0].name = "gpt-live-transcribe"
    enhanced_handler._stt_realtime_models = {"gpt-live-transcribe"}

    assert await enhanced_handler.handle_event(
        Event(type="transcribe", data={"language": "en", "name": "gpt-live-transcribe"})
    )

    session = enhanced_handler._get_realtime_transcription_session(_selected_realtime_stream(enhanced_handler))

    assert session["audio"]["input"]["transcription"] == {
        "model": "gpt-live-transcribe",
        "languages": ["en"],
        "prompt": "Test prompt",
    }


REALTIME_TTS_MODEL = "gpt-realtime-2.1-mini"
WAV_AUDIO_FORMAT = TtsAudioFormat(headerless=False, rate=24000)
REALTIME_TTS_PCM = b"\x00\x01" * 240


def _realtime_tts_events(spoken_text="Hello world", status="completed"):
    """Build the server events for one Realtime TTS response."""
    encoded = base64.b64encode(REALTIME_TTS_PCM).decode()
    return [
        _FakeRealtimeServerEvent("response.output_audio.delta", delta=encoded),
        _FakeRealtimeServerEvent("response.output_audio_transcript.done", transcript=spoken_text),
        _FakeRealtimeServerEvent("response.done", response=Mock(status=status)),
    ]


@pytest_asyncio.fixture
async def realtime_tts(enhanced_handler, mock_info, mock_clients):
    """Configure the only voice to synthesize over Realtime and yield a connection factory.

    On teardown every websocket a test opened must have been closed.
    """
    _, tts_client = mock_clients
    mock_info.tts[0].voices[0].model_name = REALTIME_TTS_MODEL
    enhanced_handler._tts_realtime_models = {REALTIME_TTS_MODEL}
    tts_client.audio.speech.with_streaming_response.create = Mock()

    managers = []

    def connect(events):
        connection = _FakeRealtimeConnection(events)
        manager = _FakeRealtimeConnectionManager(connection)
        managers.append(manager)
        tts_client.realtime.connect = Mock(return_value=manager)
        return connection, manager

    def connect_each(events_per_connection):
        """Give every websocket its own connection, scripted in the order they are opened."""
        pending = list(events_per_connection)

        def open_connection(**kwargs):
            manager = _FakeRealtimeConnectionManager(_FakeRealtimeConnection(pending.pop(0)))
            managers.append(manager)
            return manager

        tts_client.realtime.connect = Mock(side_effect=open_connection)
        return managers

    connect.each = connect_each  # type: ignore[attr-defined]

    yield connect

    await enhanced_handler._drain_background_tasks()
    assert all(manager.exited for manager in managers if manager.entered)


@pytest.mark.asyncio
async def test_realtime_tts_synthesize_flow(enhanced_handler, mock_clients, realtime_tts):
    """Test Realtime TTS sends an out-of-band response and forwards PCM without the speech API."""
    _, tts_client = mock_clients
    connection, manager = realtime_tts(_realtime_tts_events())

    result = await enhanced_handler.handle_event(
        Event(type="synthesize", data={"text": "Hello world", "voice": {"name": "alloy"}})
    )

    assert result is True
    # The receive queue is unbounded so playback at the client's pace never stops the socket being read
    tts_client.realtime.connect.assert_called_once_with(
        model=REALTIME_TTS_MODEL, websocket_connection_options={"max_queue": None}
    )
    tts_client.audio.speech.with_streaming_response.create.assert_not_called()

    session = connection.session.update.call_args.kwargs["session"]
    assert session["type"] == "realtime"
    assert session["output_modalities"] == ["audio"]
    assert session["tool_choice"] == "none"
    assert session["instructions"].endswith("Delivery style: Test instructions")
    assert session["audio"]["output"] == {
        "format": {"type": "audio/pcm", "rate": 24000},
        "voice": "alloy",
        "speed": 1.0,
    }
    assert connection.response.created == [
        {
            "conversation": "none",
            "output_modalities": ["audio"],
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello world"}]}
            ],
        }
    ]

    events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
    assert [event.type for event in events] == ["audio-start", "audio-chunk", "audio-stop"]
    assert events[0].data["rate"] == 24000
    assert events[1].payload == REALTIME_TTS_PCM
    # The closing handshake runs in the background so it cannot delay the end of the audio stream
    await enhanced_handler._drain_background_tasks()
    assert manager.exited is True


@pytest.mark.asyncio
async def test_realtime_tts_buffered_stream_returns_pcm(enhanced_handler, mock_info, realtime_tts):
    """Test the buffered sentence path collects Realtime PCM for later playback."""
    _, manager = realtime_tts(_realtime_tts_events())

    result = await enhanced_handler._get_tts_audio_stream("Hello world", mock_info.tts[0].voices[0])

    assert result.streamed is False
    assert result.audio == REALTIME_TTS_PCM
    await enhanced_handler.disconnect()
    assert manager.exited is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("events", "expected_event_types"),
    [
        # A failure before any audio is only reported
        ([_FakeRealtimeServerEvent("error", error={"message": "invalid voice"})], ["error"]),
        # Audio that was already started gets no audio stop, which would end it as a complete synthesis
        (_realtime_tts_events(status="failed"), ["audio-start", "audio-chunk", "error"]),
        # A completed response that never produced audio is a failure, not an empty success
        ([_FakeRealtimeServerEvent("response.done", response=Mock(status="completed"))], ["error"]),
    ],
)
async def test_realtime_tts_failure_closes_connection(enhanced_handler, realtime_tts, events, expected_event_types):
    """Test Realtime TTS errors fail the synthesis, report it and still close the websocket."""
    _, manager = realtime_tts(events)

    result = await enhanced_handler.handle_event(
        Event(type="synthesize", data={"text": "Hello world", "voice": {"name": "alloy"}})
    )

    assert result is False
    event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
    assert event_types == expected_event_types
    await enhanced_handler._drain_background_tasks()
    assert manager.exited is True


@pytest.mark.asyncio
async def test_incremental_realtime_tts_failure_stops_audio_once(enhanced_handler, realtime_tts):
    """Test a Realtime response failing after audio reports it and closes the stream exactly once when streaming."""
    realtime_tts(_realtime_tts_events(status="failed"))
    voice = SynthesizeVoice(name="alloy")

    await enhanced_handler.handle_event(SynthesizeStart(voice=voice).event())
    result = await enhanced_handler.handle_event(SynthesizeChunk(text="First sentence. Second one.").event())

    assert result is False
    event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
    assert event_types == ["audio-start", "audio-chunk", "error", "audio-stop", "synthesize-stopped"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stt_extra_body", "expected_languages"),
    [
        ({"language": "de"}, ["de"]),
        ({"language": "de", "languages": ["fr"]}, ["fr"]),
        # Without a usable language the field is left out and the model detects it
        ({"language": ""}, omit),
        ({"language": ["de", "fr"]}, omit),
    ],
)
async def test_transcribe_translates_singular_language_extra_body_for_plural_models(
    enhanced_handler, mock_info, mock_clients, stt_extra_body, expected_languages
):
    """Test a singular `language` extra_body override never reaches gpt-transcribe next to `languages`."""
    stt_client, _ = mock_clients
    stt_client.backend = OpenAIBackend.OPENAI
    mock_info.asr[0].models[0].name = "gpt-transcribe"
    enhanced_handler._stt_extra_body = stt_extra_body

    call_args = await _transcribe_over_http(enhanced_handler, stt_client, "gpt-transcribe")

    assert call_args["language"] is omit
    extra_body = call_args.get("extra_body", {})
    assert "language" not in extra_body
    # An extra_body `languages` list is merged over the request field by the SDK
    sent_languages = extra_body["languages"] if "languages" in extra_body else call_args["languages"]
    assert sent_languages == expected_languages


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("spoken_text", "expect_warning"),
    [("Hello, world!", False), ("Sure, I can help with that.", True)],
)
async def test_realtime_tts_warns_when_spoken_text_differs(
    enhanced_handler, realtime_tts, caplog, spoken_text, expect_warning
):
    """Test a Realtime model answering instead of reading is surfaced in the logs."""
    realtime_tts(_realtime_tts_events(spoken_text=spoken_text))

    with caplog.at_level(logging.WARNING, logger="wyoming_openai.handler"):
        await enhanced_handler.handle_event(
            Event(type="synthesize", data={"text": "Hello world", "voice": {"name": "alloy"}})
        )

    assert ("differs from the request" in caplog.text) is expect_warning


def test_realtime_tts_session_clamps_speed_and_merges_realtime_extra_body(enhanced_handler, mock_info):
    """Test the Realtime session caps speed and takes only the Realtime extra_body."""
    enhanced_handler._tts_speed = 3.0
    enhanced_handler._tts_instructions = None
    enhanced_handler._tts_extra_body = {"response_format": "pcm", "lang_code": "en"}
    # Validation rejects `instructions`; the session builder still never lets it through
    enhanced_handler._tts_realtime_extra_body = {"reasoning": {"effort": "low"}, "instructions": "Answer the user"}

    session = enhanced_handler._get_realtime_tts_session(mock_info.tts[0].voices[0])

    assert session["audio"]["output"]["speed"] == 1.5
    assert session["reasoning"] == {"effort": "low"}
    # /v1/audio/speech fields never reach the session, and the read-aloud instructions are not replaceable
    assert "response_format" not in session
    assert "lang_code" not in session
    assert session["instructions"].startswith("You are a text-to-speech engine.")
    assert "Delivery style" not in session["instructions"]


def test_realtime_tts_session_merges_audio_overrides_without_changing_format(enhanced_handler, mock_info):
    """Test audio overrides are merged into the session while the PCM output format is kept."""
    enhanced_handler._tts_realtime_extra_body = {
        "audio": {"output": {"speed": 0.5, "voice": "cedar", "format": {"type": "audio/pcmu"}}},
    }

    session = enhanced_handler._get_realtime_tts_session(mock_info.tts[0].voices[0])

    # The voice the Wyoming client selected is kept as well
    assert session["audio"] == {
        "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "alloy", "speed": 0.5}
    }


@pytest.mark.parametrize("extra_body", [{"audio": None}, {"audio": {"output": None}}])
def test_realtime_tts_session_treats_null_audio_as_unset(enhanced_handler, mock_info, extra_body):
    """Test a null audio or audio.output in the Realtime extra body leaves the session audio as without one."""
    enhanced_handler._tts_speed = None
    enhanced_handler._tts_realtime_extra_body = extra_body

    session = enhanced_handler._get_realtime_tts_session(mock_info.tts[0].voices[0])

    assert session["audio"] == {"output": {"voice": "alloy", "format": {"type": "audio/pcm", "rate": 24000}}}


def test_handler_rejects_incompatible_realtime_audio_format(mock_info, mock_clients, dummy_reader_writer):
    """Test a Realtime output format Wyoming cannot play is rejected at construction."""
    stt_client, tts_client = mock_clients
    reader, writer = dummy_reader_writer

    with pytest.raises(ValueError, match=r"audio\.output\.format"):
        OpenAIEventHandler(
            reader,
            writer,
            info=mock_info,
            stt_client=stt_client,
            tts_client=tts_client,
            tts_realtime_models=["gpt-realtime-2.1-mini"],
            tts_realtime_extra_body={"audio": {"output": {"format": {"type": "audio/pcmu"}}}},
        )


@pytest.mark.asyncio
async def test_incremental_realtime_tts_without_audio_never_starts_audio(enhanced_handler, mock_info, realtime_tts):
    """Test a completed Realtime response with no audio aborts instead of streaming chunks without audio-start."""
    mock_info.tts[0].supports_synthesize_streaming = True
    realtime_tts([_FakeRealtimeServerEvent("response.done", response=Mock(status="completed"))])

    await enhanced_handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="alloy")).event())
    result = await enhanced_handler.handle_event(SynthesizeChunk(text="First sentence. Second one.").event())

    assert result is False
    event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
    assert event_types == ["error", "synthesize-stopped"]


@pytest.mark.asyncio
async def test_realtime_tts_times_out_when_server_stalls(enhanced_handler, realtime_tts, monkeypatch):
    """Test a Realtime server that stops sending events fails the synthesis instead of hanging."""
    monkeypatch.setattr("wyoming_openai.handler.REALTIME_TTS_EVENT_TIMEOUT", 0.01)
    _, manager = realtime_tts(_realtime_tts_events()[:1])

    result = await enhanced_handler.handle_event(
        Event(type="synthesize", data={"text": "Hello world", "voice": {"name": "alloy"}})
    )

    assert result is False
    event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
    assert event_types == ["audio-start", "audio-chunk", "error"]
    await enhanced_handler._drain_background_tasks()
    assert manager.exited is True


@pytest.mark.asyncio
async def test_realtime_tts_closes_websocket_when_client_write_fails(enhanced_handler, realtime_tts):
    """Test the websocket is closed by the time the handler disconnects when the Wyoming client goes away."""
    _, manager = realtime_tts(_realtime_tts_events())
    enhanced_handler.write_event = AsyncMock(side_effect=[None, ConnectionResetError("client gone"), None])

    result = await enhanced_handler.handle_event(
        Event(type="synthesize", data={"text": "Hello world", "voice": {"name": "alloy"}})
    )

    assert result is False
    # A failed synthesis closes in the background too, so an abort never waits on the closing handshake
    await enhanced_handler.disconnect()
    assert manager.exited is True


@pytest.mark.asyncio
async def test_realtime_tts_abort_does_not_wait_for_websocket_close(enhanced_handler, realtime_tts):
    """Test a failed synthesis reports back while the websocket closing handshake is still pending."""
    _, manager = realtime_tts([_FakeRealtimeServerEvent("error", error={"message": "invalid voice"})])
    release_close = asyncio.Event()
    close_connection = manager.connection.close

    async def slow_close():
        await release_close.wait()
        await close_connection()

    manager.connection.close = slow_close

    result = await asyncio.wait_for(
        enhanced_handler.handle_event(
            Event(type="synthesize", data={"text": "Hello world", "voice": {"name": "alloy"}})
        ),
        timeout=1,
    )

    assert result is False
    assert manager.connection.closed is False

    # The Wyoming client is released before the closing handshake finishes
    disconnect = asyncio.create_task(enhanced_handler.disconnect())
    await asyncio.sleep(0.01)
    enhanced_handler.writer.close.assert_called_once()
    assert not disconnect.done()

    release_close.set()
    await disconnect
    assert manager.connection.closed is True


@pytest.mark.asyncio
async def test_realtime_tts_fidelity_joins_transcript_parts(enhanced_handler, mock_info, realtime_tts, caplog):
    """Test a response spoken in several transcript parts is compared as a whole."""
    mock_info.tts[0].supports_synthesize_streaming = False
    encoded = base64.b64encode(REALTIME_TTS_PCM).decode()
    realtime_tts(
        [
            _FakeRealtimeServerEvent("response.output_audio.delta", delta=encoded),
            _FakeRealtimeServerEvent("response.output_audio_transcript.delta", delta="Hello "),
            _FakeRealtimeServerEvent("response.output_audio_transcript.done", transcript="Hello world."),
            _FakeRealtimeServerEvent("response.output_audio_transcript.done", transcript="How are you?"),
            _FakeRealtimeServerEvent("response.done", response=Mock(status="completed")),
        ]
    )

    with caplog.at_level(logging.WARNING, logger="wyoming_openai.handler"):
        result = await enhanced_handler.handle_event(
            Event(type="synthesize", data={"text": "Hello world. How are you?", "voice": {"name": "alloy"}})
        )

    assert result is True
    assert "differs from the request" not in caplog.text


@pytest.mark.asyncio
async def test_buffered_realtime_tts_is_forwarded_as_raw_pcm(enhanced_handler, mock_info, realtime_tts):
    """Test buffered Realtime audio is known to be headerless PCM and sent without WAV sniffing."""
    realtime_tts(_realtime_tts_events())
    voice = mock_info.tts[0].voices[0]

    result = await enhanced_handler._get_tts_audio_stream("Hello world", voice)
    audio_format = enhanced_handler._get_tts_audio_format(voice)
    assert audio_format == TtsAudioFormat(headerless=True, rate=24000)

    with patch("wyoming_openai.wav._reject_foreign_audio") as sniff:
        await _play_buffered_sentence(enhanced_handler, result.audio, True, 0, audio_format=audio_format)

    sniff.assert_not_called()
    events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
    assert [event.type for event in events] == ["audio-start", "audio-chunk"]
    assert events[0].data["rate"] == 24000


@pytest.mark.asyncio
async def test_realtime_transcription_session_applies_stt_extra_body(enhanced_handler, mock_info, mock_clients):
    """Test only the Realtime STT extra_body reaches transcription sessions, never the HTTP one."""
    stt_client, _ = mock_clients
    stt_client.backend = OpenAIBackend.OPENAI
    mock_info.asr[0].models[0].name = "gpt-live-transcribe"
    enhanced_handler._stt_realtime_models = {"gpt-live-transcribe"}
    enhanced_handler._stt_extra_body = {"temperature": 0.2, "stream": True, "response_format": "json"}
    enhanced_handler._stt_realtime_extra_body = {"language": "de", "keywords": ["Wyoming"]}

    assert await enhanced_handler.handle_event(
        Event(type="transcribe", data={"language": "en", "name": "gpt-live-transcribe"})
    )

    session = enhanced_handler._get_realtime_transcription_session(_selected_realtime_stream(enhanced_handler))

    assert session["audio"]["input"]["transcription"] == {
        "model": "gpt-live-transcribe",
        "languages": ["de"],
        "prompt": "Test prompt",
        "keywords": ["Wyoming"],
    }


def _mock_speech_response(
    tts_client, chunks: list[bytes] | Callable[[str], list[bytes] | AsyncIterator[bytes]]
) -> None:
    """Make /v1/audio/speech return byte chunks.

    `chunks` is a list of chunks, or a callable that takes the request input and returns the chunks for it
    as a list or as an async iterator, which may stall or raise midway.
    """

    def create(**kwargs):
        source = chunks if isinstance(chunks, list) else chunks(kwargs["input"])

        async def iter_bytes(chunk_size=None):
            if isinstance(source, list):
                for chunk in source:
                    yield chunk
            else:
                async for chunk in source:
                    yield chunk

        response = Mock()
        response.iter_bytes = iter_bytes
        stream_response = AsyncMock()
        stream_response.__aenter__ = AsyncMock(return_value=response)
        stream_response.__aexit__ = AsyncMock(return_value=None)
        return stream_response

    tts_client.audio.speech.with_streaming_response.create = Mock(side_effect=create)


def _header_only_wav():
    """Build a valid WAV file that declares no PCM samples."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(b"")
    return wav_buffer.getvalue()


def _unbounded_header_only_wav():
    """Build a header-only WAV whose data size is the unknown-length sentinel streaming servers write."""
    return _header_only_wav()[:-4] + struct.pack("<I", 0xFFFFFFFF)


def _empty_wav_with_trailer(trailer: bytes) -> bytes:
    """Build a WAV file without samples whose RIFF container goes on with metadata after the data chunk."""
    wav = _header_only_wav() + trailer
    return wav[:4] + struct.pack("<I", len(wav) - 8) + wav[8:]


@pytest.mark.asyncio
@pytest.mark.parametrize("header_only_wav", [_header_only_wav, _unbounded_header_only_wav])
async def test_http_tts_header_only_wav_for_speakable_text_fails(
    enhanced_handler, mock_info, mock_clients, header_only_wav
):
    """Test a WAV without samples fails for text with something to say, whatever length it declares."""
    _, tts_client = mock_clients
    voice = mock_info.tts[0].voices[0]
    _mock_speech_response(tts_client, [header_only_wav()])

    assert not await enhanced_handler.handle_event(
        Event(type="synthesize", data={"text": "Hello", "voice": {"name": "alloy"}})
    )
    event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
    assert event_types == ["audio-start", "error"]

    assert await _stream_whole_request(enhanced_handler, voice, "Hello", send_audio_start=False) is None

    # A buffered response is always a sentence, which is skipped and counted instead
    result = await enhanced_handler._get_tts_audio_stream("Hello", voice)
    assert await _play_buffered_sentence(enhanced_handler, result.audio, False, 0, WAV_AUDIO_FORMAT, "Hello") == 0
    assert enhanced_handler._synthesis.skipped_speakable_sentence


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_sentence", ["First", "Second"])
async def test_incremental_tts_sentence_without_audio_is_skipped_with_warning(
    enhanced_handler, mock_info, mock_clients, caplog, empty_sentence
):
    """Test an empty direct or buffered sentence warns, and the remaining sentences play."""
    _, tts_client = mock_clients
    mock_info.tts[0].supports_synthesize_streaming = True
    spoken = _pcm_wav_with_unbounded_sizes(b"\x00\x01" * 240)
    _mock_speech_response(tts_client, lambda text: [_unbounded_header_only_wav() if empty_sentence in text else spoken])

    assert await enhanced_handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="alloy")).event())
    assert await enhanced_handler._process_ready_sentences(["First one.", "Second one.", "Third one."])
    assert enhanced_handler._synthesis.skipped_speakable_sentence
    assert await enhanced_handler.handle_event(Event(type="synthesize-stop"))

    event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
    assert event_types == ["audio-start", "audio-chunk", "audio-chunk", "audio-stop", "synthesize-stopped"]
    assert f"Skipping sentence without audio: {empty_sentence} one." in caplog.text
    assert not enhanced_handler._synthesis.skipped_speakable_sentence
    assert tts_client.audio.speech.with_streaming_response.create.call_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("header_only_wav", [_header_only_wav, _unbounded_header_only_wav])
async def test_http_tts_header_only_wav_is_an_empty_success(enhanced_handler, mock_info, mock_clients, header_only_wav):
    """Test a WAV without samples for text with nothing to say is not a failure on any of the three paths."""
    _, tts_client = mock_clients
    voice = mock_info.tts[0].voices[0]

    _mock_speech_response(tts_client, [header_only_wav()])
    assert await enhanced_handler.handle_event(
        Event(type="synthesize", data={"text": "...", "voice": {"name": "alloy"}})
    )
    event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
    assert event_types == ["audio-start", "audio-stop"]

    _mock_speech_response(tts_client, [header_only_wav()])
    assert await _stream_whole_request(enhanced_handler, voice, "...", send_audio_start=False, start_timestamp=5) == 5

    _mock_speech_response(tts_client, [header_only_wav()])
    result = await enhanced_handler._get_tts_audio_stream("...", voice)
    assert await _play_buffered_sentence(enhanced_handler, result.audio, False, 5, WAV_AUDIO_FORMAT) == 5


@pytest.mark.asyncio
async def test_buffered_http_pcm_is_forwarded_without_wav_parsing(enhanced_handler, mock_info, mock_clients, caplog):
    """Test the buffered path reads the format the direct path uses, so raw PCM answering `pcm` is played as asked."""
    _, tts_client = mock_clients
    voice = mock_info.tts[0].voices[0]
    wav_format = TtsAudioFormat(headerless=False, rate=24000, may_be_wav=True)
    assert enhanced_handler._get_tts_audio_format(voice) == wav_format

    enhanced_handler._tts_extra_body = {"response_format": "pcm"}
    audio_format = enhanced_handler._get_tts_audio_format(voice)
    assert audio_format == TtsAudioFormat(headerless=True, rate=24000, may_be_wav=True)

    pcm = b"\x00\x01" * 240
    _mock_speech_response(tts_client, [pcm])
    result = await enhanced_handler._get_tts_audio_stream("Hello", voice)
    with caplog.at_level(logging.WARNING, logger="wyoming_openai.handler"):
        assert await _play_buffered_sentence(enhanced_handler, result.audio, True, 0, audio_format=audio_format) == 10

    # Read as a WAV request, the same bytes would be reported as a missing header
    assert "is not a WAV file" not in caplog.text
    assert enhanced_handler.write_event.call_args.args[0].payload == pcm


@pytest.mark.asyncio
async def test_http_tts_zero_size_wav_header_followed_by_audio_is_played(enhanced_handler, mock_info, mock_clients):
    """Test audio after a header that declares no samples is played, as servers streaming unknown lengths send it."""
    _, tts_client = mock_clients
    voice = mock_info.tts[0].voices[0]
    pcm = b"\x00\x01" * 240

    _mock_speech_response(tts_client, [_header_only_wav(), pcm])
    assert await _stream_whole_request(enhanced_handler, voice, "Hello", send_audio_start=True) == 10
    events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
    assert [event.type for event in events] == ["audio-start", "audio-chunk"]
    assert events[1].payload == pcm

    enhanced_handler.write_event.reset_mock()
    wav = _header_only_wav() + pcm
    assert await _play_buffered_sentence(enhanced_handler, wav, False, 0, WAV_AUDIO_FORMAT) == 10
    assert enhanced_handler.write_event.call_args.args[0].payload == pcm


@pytest.mark.asyncio
async def test_http_tts_empty_body_fails_without_starting_audio(enhanced_handler, mock_info, mock_clients):
    """Test a speech response without any bytes fails on the standalone, direct and buffered paths."""
    _, tts_client = mock_clients
    voice = mock_info.tts[0].voices[0]

    _mock_speech_response(tts_client, [])
    result = await enhanced_handler.handle_event(
        Event(type="synthesize", data={"text": "Hello", "voice": {"name": "alloy"}})
    )
    assert result is False
    assert _event_types(enhanced_handler) == ["error"]
    enhanced_handler.write_event.reset_mock()

    _mock_speech_response(tts_client, [])
    assert await _stream_whole_request(enhanced_handler, voice, "Hello", send_audio_start=False) is None

    # A buffered response is always a sentence, which is skipped and counted without opening the stream
    _mock_speech_response(tts_client, [])
    result = await enhanced_handler._get_tts_audio_stream("Hello", voice)
    assert await _play_buffered_sentence(enhanced_handler, result.audio, True, 0, WAV_AUDIO_FORMAT, "Hello") == 0
    assert enhanced_handler._synthesis.skipped_speakable_sentence
    enhanced_handler.write_event.assert_not_called()


@pytest.mark.asyncio
async def test_realtime_tts_skips_text_with_nothing_to_speak(enhanced_handler, mock_clients, realtime_tts):
    """Test punctuation-only text never reaches a conversational Realtime model."""
    _, tts_client = mock_clients
    realtime_tts(_realtime_tts_events())

    result = await enhanced_handler.handle_event(
        Event(type="synthesize", data={"text": "... \U0001f642", "voice": {"name": "alloy"}})
    )

    assert result is True
    tts_client.realtime.connect.assert_not_called()
    event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
    assert event_types == ["audio-start", "audio-stop"]


@pytest.mark.parametrize(
    ("text", "speakable"),
    [
        ("Hello", True),
        ("42", True),
        # Symbols with a spoken name are read aloud
        ("+", True),
        ("=", True),
        ("&", True),
        ("\u20ac", True),
        # Punctuation and emoji are not
        ("...", False),
        ("\u2014 !?", False),
        ("\U0001f642", False),
    ],
)
def test_speakable_text(text, speakable):
    """Test only punctuation and emoji count as text with nothing to read aloud."""
    assert _has_speakable_content(text) is speakable


@pytest.mark.asyncio
async def test_incremental_realtime_tts_skips_unspeakable_sentences(enhanced_handler, mock_info, realtime_tts):
    """Test unspeakable sentences are dropped from a streaming batch instead of aborting it."""
    mock_info.tts[0].supports_synthesize_streaming = True
    realtime_tts(_realtime_tts_events(spoken_text="Hello world."))
    assert await enhanced_handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="alloy")).event())

    assert await enhanced_handler._process_ready_sentences(["...", "Hello world."])

    event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
    assert event_types == ["audio-start", "audio-chunk"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup", ["_abort_synthesis", "disconnect", "stop"])
async def test_pending_synthesis_tasks_are_cancelled(enhanced_handler, cleanup):
    """Test sentence tasks do not outlive an aborted session or a disconnected client."""
    task = enhanced_handler._create_synthesis_task(asyncio.sleep(60), name="pending_sentence")
    await asyncio.sleep(0)

    with patch("wyoming.server.AsyncEventHandler.stop", new=AsyncMock()):
        await getattr(enhanced_handler, cleanup)()

    assert task.cancelled()
    assert not enhanced_handler._synthesis_tasks


def test_suffixed_voice_name_resolves_after_models_change(enhanced_handler, mock_info, caplog):
    """Test a "voice (model)" name from a multi-model setup still resolves once the suffix is gone."""
    with caplog.at_level(logging.WARNING, logger="wyoming_openai.handler"):
        voice = enhanced_handler._validate_tts_voice_and_language("alloy (gpt-4o-mini-tts)", None)

    assert voice is mock_info.tts[0].voices[0]
    assert "no longer advertised" in caplog.text
    assert enhanced_handler._validate_tts_voice_and_language("unknown (gpt-4o-mini-tts)", None) is None


def _add_realtime_voice_first(mock_info, enhanced_handler):
    """Advertise a Realtime voice ahead of the speech API one and return both."""
    speech_voice = mock_info.tts[0].voices[0]
    realtime_voice = TtsVoiceModel(
        name="alloy (gpt-realtime-2.1-mini)",
        model_name=REALTIME_TTS_MODEL,
        backend_voice_name=speech_voice.backend_voice_name,
        description="alloy",
        attribution=speech_voice.attribution,
        installed=True,
        languages=speech_voice.languages,
        version=None,
    )
    mock_info.tts[0].voices.insert(0, realtime_voice)
    enhanced_handler._tts_realtime_models = {REALTIME_TTS_MODEL}
    return speech_voice, realtime_voice


def test_suffixed_voice_name_of_a_removed_model_prefers_the_speech_api(enhanced_handler, mock_info):
    """Test a "voice (model)" name whose model is gone does not move the client onto a Realtime model."""
    speech_voice, _ = _add_realtime_voice_first(mock_info, enhanced_handler)

    assert enhanced_handler._validate_tts_voice_and_language("alloy (tts-1-hd)", None) is speech_voice
    # A plain name shared by both transports follows the same rule
    speech_voice.name = "alloy (gpt-4o-mini-tts)"
    assert enhanced_handler._validate_tts_voice_and_language("alloy", None) is speech_voice


@pytest.mark.asyncio
async def test_realtime_tts_failure_reports_status_details(enhanced_handler, realtime_tts, caplog):
    """Test the reason a Realtime response did not complete is part of the logged error."""
    status_details = Mock(reason="content_filter", error=Mock(code="policy", type=None))
    realtime_tts(
        [_FakeRealtimeServerEvent("response.done", response=Mock(status="incomplete", status_details=status_details))]
    )

    with caplog.at_level(logging.ERROR, logger="wyoming_openai.handler"):
        result = await enhanced_handler.handle_event(
            Event(type="synthesize", data={"text": "Hello world", "voice": {"name": "alloy"}})
        )

    assert result is False
    assert "status incomplete (content_filter, policy)" in caplog.text


@pytest.mark.asyncio
async def test_http_tts_truncated_wav_without_samples_fails(enhanced_handler, mock_info, mock_clients):
    """Test a WAV that declares PCM data but delivers only its header fails without ending the opened stream."""
    _, tts_client = mock_clients
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(b"\x00\x01" * 240)
    header_only = wav_buffer.getvalue()[:-480]

    _mock_speech_response(tts_client, [header_only])
    result = await enhanced_handler.handle_event(
        Event(type="synthesize", data={"text": "Hello", "voice": {"name": "alloy"}})
    )

    assert result is False
    event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
    assert event_types == ["audio-start", "error"]

    # As on the buffered path, text with nothing to say does not excuse the missing samples
    _mock_speech_response(tts_client, [header_only])
    assert not await enhanced_handler.handle_event(
        Event(type="synthesize", data={"text": "...", "voice": {"name": "alloy"}})
    )

    # Sentence skipping never excuses declared samples that did not arrive.
    voice = mock_info.tts[0].voices[0]
    enhanced_handler._allow_streaming_task_id = "sentence_0"
    with pytest.raises(TtsStreamError):
        await enhanced_handler._get_tts_audio_stream("Hello", voice, task_id="sentence_0")
    assert await _play_buffered_sentence(enhanced_handler, header_only, True, 0, WAV_AUDIO_FORMAT, "Hello") is None


class _OverlapRecordingSpeech:
    """Stand-in for ``audio.speech.with_streaming_response`` that records how many requests are in flight at once."""

    def __init__(self):
        self.in_flight = 0
        self.peak = 0

    def create(self, **kwargs):
        return self

    async def __aenter__(self):
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        await asyncio.sleep(0)  # let every other request that is allowed to start do so
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.in_flight -= 1

    async def iter_bytes(self, chunk_size):
        yield b"\x00\x01" * 64


@pytest.mark.asyncio
@pytest.mark.parametrize(("configured", "expected_peak"), [(None, 6), (1, 1), (2, 2), (5, 5), (0, 6)])
async def test_tts_requests_overlap_up_to_configured_limit(
    dummy_info, dummy_clients, dummy_reader_writer, configured, expected_peak
):
    """A handler given a TTS limit synthesizes at most that many sentences at once; without one, or at 0, all."""
    stt_client, tts_client = dummy_clients
    reader, writer = dummy_reader_writer
    speech = _OverlapRecordingSpeech()
    tts_client.audio.speech.with_streaming_response = speech
    handler = OpenAIEventHandler(
        reader,
        writer,
        info=dummy_info,
        stt_client=stt_client,
        tts_client=tts_client,
        tts_limit=None if configured is None else ConcurrencyLimit(configured),
    )
    voice = handler._get_voice("voice1")
    assert voice is not None

    results = await asyncio.gather(*(handler._get_tts_audio_stream(f"Sentence {i}.", voice) for i in range(6)))

    assert all(result.audio for result in results)
    assert speech.peak == expected_peak


def _stt_limited_handler(info, stt_client, limit):
    """A handler for one more connection of a server whose connections share the STT limit."""
    writer = MagicMock(name="writer")
    writer.wait_closed = AsyncMock()
    writer.transport = Mock()
    handler = OpenAIEventHandler(
        MagicMock(name="reader"), writer, info=info, stt_client=stt_client, tts_client=None, stt_limit=limit
    )
    handler.write_event = AsyncMock()
    return handler


async def _settle():
    """Let every task that can run do so, until each waits on something the test controls."""
    for _ in range(20):
        await asyncio.sleep(0)


def _written_event_types(handler):
    return [call.args[0].type for call in handler.write_event.call_args_list]


def _written_transcripts(handler):
    return [
        Transcript.from_event(call.args[0]).text
        for call in handler.write_event.call_args_list
        if Transcript.is_type(call.args[0].type)
    ]


@pytest.mark.asyncio
async def test_stt_limit_applies_to_http_transcriptions_across_connections(mock_info, mock_clients):
    """Connections sharing a one-slot limit send their recordings one at a time, and all are transcribed."""
    from openai.types.audio.transcription import Transcription

    stt_client, _ = mock_clients
    in_flight = 0
    peak = 0
    respond = asyncio.Event()

    async def create(**kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await respond.wait()
        in_flight -= 1
        return Transcription(text="Hello")

    stt_client.audio.transcriptions.create = AsyncMock(side_effect=create)
    limit = ConcurrencyLimit(1)
    handlers = [_stt_limited_handler(mock_info, stt_client, limit) for _ in range(3)]

    tasks = [asyncio.create_task(_stream_audio(handler)) for handler in handlers]
    await _settle()

    assert stt_client.audio.transcriptions.create.await_count == 1
    respond.set()
    await asyncio.gather(*tasks)

    assert peak == 1
    assert stt_client.audio.transcriptions.create.await_count == 3
    assert [_written_transcripts(handler) for handler in handlers] == [["Hello"]] * 3
    assert not limit.full


@pytest.mark.asyncio
async def test_stt_limit_slot_is_released_when_an_http_transcription_fails(mock_info, mock_clients):
    """A failed request frees its slot, so the next connection is not held up."""
    from openai.types.audio.transcription import Transcription

    stt_client, _ = mock_clients
    stt_client.audio.transcriptions.create = AsyncMock(
        side_effect=[RuntimeError("backend down"), Transcription(text="Hi")]
    )
    limit = ConcurrencyLimit(1)
    failing, succeeding = (_stt_limited_handler(mock_info, stt_client, limit) for _ in range(2))

    await _stream_audio(failing)
    assert not limit.full
    await asyncio.wait_for(_stream_audio(succeeding), timeout=1)

    assert "error" in _written_event_types(failing)
    assert _written_transcripts(succeeding) == ["Hi"]
    assert not limit.full


@pytest.mark.asyncio
async def test_stt_limit_http_slot_is_released_before_the_error_is_written(mock_info, mock_clients):
    """A failed request frees its slot even while a client is slow to take the error."""
    stt_client, _ = mock_clients
    stt_client.audio.transcriptions.create = AsyncMock(side_effect=RuntimeError("backend down"))
    limit = ConcurrencyLimit(1)
    handler = _stt_limited_handler(mock_info, stt_client, limit)
    client_reads = asyncio.Event()

    async def write_event(event):
        if event.type == "error":
            await client_reads.wait()

    handler.write_event = AsyncMock(side_effect=write_event)

    request = asyncio.create_task(_stream_audio(handler))
    await _settle()
    assert not request.done()
    assert not limit.full

    client_reads.set()
    await asyncio.wait_for(request, timeout=1)
    assert _written_event_types(handler) == ["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True])
async def test_stt_limit_http_slot_is_released_before_the_transcript_is_written(mock_info, mock_clients, streamed):
    """A client slow to take the transcript does not hold the slot once the backend's response is read."""
    from openai.types.audio.transcription import Transcription

    stt_client, _ = mock_clients
    stt_client.audio.transcriptions.create = AsyncMock(
        return_value=_HiStream() if streamed else Transcription(text="Hi")
    )
    limit = ConcurrencyLimit(1)
    handler = _stt_limited_handler(mock_info, stt_client, limit)
    client_reads = asyncio.Event()

    async def write_event(event):
        if Transcript.is_type(event.type):
            await client_reads.wait()

    handler.write_event = AsyncMock(side_effect=write_event)

    with patch("wyoming_openai.handler.AsyncStream", _HiStream):
        request = asyncio.create_task(_stream_audio(handler))
        await _settle()
        assert not request.done()
        assert not limit.full

        client_reads.set()
        await asyncio.wait_for(request, timeout=1)

    assert _written_transcripts(handler) == ["Hi"]


def _completing_realtime_connect(transcript="Hi"):
    """A Realtime connect that opens a new session for every stream, each completing with the transcript."""
    def connect(**kwargs):
        completed = _FakeRealtimeServerEvent(
            "conversation.item.input_audio_transcription.completed", transcript=transcript
        )
        return _FakeRealtimeConnectionManager(_FakeRealtimeConnection([completed]))

    return Mock(side_effect=connect)


def _realtime_limited_handlers(enhanced_handler, mock_info, stt_client, limit, count):
    """Handlers for `count` connections that transcribe over Realtime and share the STT limit."""
    handlers = [_stt_limited_handler(mock_info, stt_client, limit) for _ in range(count)]
    for handler in handlers:
        handler._stt_realtime_models = enhanced_handler._stt_realtime_models
    return handlers


REALTIME_AUDIO_START = Event(type="audio-start", data={"rate": 24000, "width": 2, "channels": 1})


@pytest.mark.asyncio
async def test_stt_limit_holds_a_realtime_slot_from_audio_start_until_the_session_ends(
    enhanced_handler, mock_info, mock_clients
):
    """A second Realtime stream opens its session only once the first one has ended."""
    stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    stt_client.realtime.connect = _completing_realtime_connect()
    limit = ConcurrencyLimit(1)
    first, second = _realtime_limited_handlers(enhanced_handler, mock_info, stt_client, limit, 2)

    await first.handle_event(REALTIME_AUDIO_START)
    assert limit.full
    second_start = asyncio.create_task(second.handle_event(REALTIME_AUDIO_START))
    await _settle()

    assert stt_client.realtime.connect.call_count == 1
    assert not second_start.done()

    await first.handle_event(
        Event(type="audio-chunk", data={"rate": 24000, "width": 2, "channels": 1}, payload=b"\x00\x01" * 10)
    )
    assert await first.handle_event(Event(type="audio-stop"))
    await asyncio.wait_for(second_start, timeout=1)

    assert stt_client.realtime.connect.call_count == 2
    assert limit.full
    assert await second.handle_event(Event(type="audio-stop"))
    assert _written_transcripts(first) == ["Hi"]
    assert _written_transcripts(second) == ["Hi"]
    assert not limit.full


@pytest.mark.asyncio
async def test_stt_limit_realtime_slot_is_released_when_the_client_disconnects(
    enhanced_handler, mock_info, mock_clients
):
    """A Realtime stream the client never stops frees its slot when the connection goes away."""
    stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    stt_client.realtime.connect = _completing_realtime_connect()
    limit = ConcurrencyLimit(1)
    (handler,) = _realtime_limited_handlers(enhanced_handler, mock_info, stt_client, limit, 1)

    await handler.handle_event(REALTIME_AUDIO_START)
    assert limit.full
    await handler.disconnect()

    assert not limit.full


@pytest.mark.asyncio
async def test_stt_limit_realtime_slot_is_released_when_the_session_cannot_open(
    enhanced_handler, mock_info, mock_clients
):
    """A Realtime session that fails to open does not keep the slot it waited for."""
    stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    stt_client.realtime.connect = Mock(side_effect=RuntimeError("cannot connect"))
    limit = ConcurrencyLimit(1)
    (handler,) = _realtime_limited_handlers(enhanced_handler, mock_info, stt_client, limit, 1)

    await handler.handle_event(REALTIME_AUDIO_START)

    assert not limit.full


class _GatedCloseManager(_FakeRealtimeConnectionManager):
    """A Realtime connection manager whose closing handshake lasts until the gate opens."""

    def __init__(self, connection, gate):
        super().__init__(connection)
        self.gate = gate

    async def __aexit__(self, exc_type, exc, tb):
        await self.gate.wait()
        await super().__aexit__(exc_type, exc, tb)


@pytest.mark.asyncio
async def test_stt_limit_realtime_slot_is_released_while_the_websocket_is_still_closing(
    enhanced_handler, mock_info, mock_clients
):
    """As for Realtime TTS, the limit counts sessions in flight: the next one does not wait for the last close."""
    stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    gate = asyncio.Event()

    def connect(**kwargs):
        completed = _FakeRealtimeServerEvent("conversation.item.input_audio_transcription.completed", transcript="Hi")
        return _GatedCloseManager(_FakeRealtimeConnection([completed]), gate)

    stt_client.realtime.connect = Mock(side_effect=connect)
    limit = ConcurrencyLimit(1)
    first, second = _realtime_limited_handlers(enhanced_handler, mock_info, stt_client, limit, 2)

    await first.handle_event(REALTIME_AUDIO_START)
    assert await first.handle_event(Event(type="audio-stop"))
    await _settle()
    assert _written_transcripts(first) == ["Hi"]
    assert first._background_tasks  # Its websocket is still closing
    assert not limit.full

    await asyncio.wait_for(second.handle_event(REALTIME_AUDIO_START), timeout=1)
    assert stt_client.realtime.connect.call_count == 2

    gate.set()
    assert await second.handle_event(Event(type="audio-stop"))
    await asyncio.wait_for(asyncio.gather(first._drain_background_tasks(), second._drain_background_tasks()), 1)
    assert not limit.full


@pytest.mark.asyncio
async def test_completed_realtime_session_closes_before_the_transcript_is_written(
    enhanced_handler, mock_info, mock_clients
):
    """A client slow to take the transcript does not keep the finished session, or its slot, open."""
    stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    completed = _FakeRealtimeServerEvent("conversation.item.input_audio_transcription.completed", transcript="Hi")
    manager = _FakeRealtimeConnectionManager(_FakeRealtimeConnection([completed]))
    stt_client.realtime.connect = Mock(return_value=manager)
    limit = ConcurrencyLimit(1)
    (handler,) = _realtime_limited_handlers(enhanced_handler, mock_info, stt_client, limit, 1)
    client_reads = asyncio.Event()

    async def write_event(event):
        if Transcript.is_type(event.type):
            await client_reads.wait()

    handler.write_event = AsyncMock(side_effect=write_event)

    await handler.handle_event(REALTIME_AUDIO_START)
    request = asyncio.create_task(handler.handle_event(Event(type="audio-stop")))
    await _settle()

    assert not request.done()
    assert manager.exited
    assert not limit.full

    client_reads.set()
    assert await asyncio.wait_for(request, timeout=1)
    assert _written_event_types(handler) == ["transcript-start", "transcript", "transcript-stop"]


def _record_client_io(handler):
    """Record, in order, the event types the handler writes and when it closes its writer."""
    io_log = []

    async def write_event(event):
        io_log.append(event.type)

    handler.write_event = AsyncMock(side_effect=write_event)
    handler.writer.close = Mock(side_effect=lambda: io_log.append("close"))
    return io_log


class _HiStream(_FakeTranscriptionStream):
    """A streamed /v1/audio/transcriptions response: one delta, then the final text."""

    async def __aiter__(self):
        yield Mock(type="transcript.text.delta", delta="Hi")
        yield Mock(type="transcript.text.done", text="Hi")


def _gate_backend_response(handler, stt_client, realtime, mock_info, mock_clients):
    """
    Hold the backend's answer to an audio-stop: returns (entered, respond), set once the request reached the
    backend, and to let a successful "Hi" through. A Realtime answer brings a delta first.
    """
    from openai.types.audio.transcription import Transcription

    entered = asyncio.Event()
    respond = asyncio.Event()
    if realtime:
        _use_realtime_stt_model(handler, mock_info, mock_clients)
        connection = _FakeRealtimeConnection([])

        async def commit():
            entered.set()
            await respond.wait()
            await connection.events.put(
                _FakeRealtimeServerEvent("conversation.item.input_audio_transcription.delta", delta="Hi")
            )
            await connection.events.put(
                _FakeRealtimeServerEvent("conversation.item.input_audio_transcription.completed", transcript="Hi")
            )

        connection.input_audio_buffer.commit = commit
        stt_client.realtime.connect = Mock(return_value=_FakeRealtimeConnectionManager(connection))
    else:

        async def create(**kwargs):
            entered.set()
            await respond.wait()
            return Transcription(text="Hi")

        stt_client.audio.transcriptions.create = AsyncMock(side_effect=create)
    return entered, respond


@pytest.mark.asyncio
@pytest.mark.parametrize("realtime", [False, True])
async def test_stop_cancels_the_stt_request_in_flight(enhanced_handler, mock_info, mock_clients, caplog, realtime):
    """
    stop() cancels a request the backend is still working on and answers it with the shutdown error, before the
    connection closes; nothing of the request follows, even once the backend would have answered.
    """
    stt_client, _ = mock_clients
    handler = enhanced_handler
    entered, respond = _gate_backend_response(handler, stt_client, realtime, mock_info, mock_clients)
    audio_format = {"rate": 24000, "width": 2, "channels": 1}
    await handler.handle_event(Event(type="audio-start", data=audio_format))
    await handler.handle_event(Event(type="audio-chunk", data=audio_format, payload=b"\x00\x01" * 10))
    io_log = _record_client_io(handler)

    with caplog.at_level(logging.INFO, logger="wyoming_openai.handler"):
        audio_stop = asyncio.create_task(handler.handle_event(Event(type="audio-stop")))
        await asyncio.wait_for(entered.wait(), timeout=1)
        await asyncio.wait_for(handler.stop(), timeout=3)
        # Ends once stop() has answered the client and closed the connection
        assert await asyncio.wait_for(audio_stop, timeout=1) is False
        respond.set()
        await _settle()

    assert "Cancelling the STT request in flight" in caplog.text
    # Cut short by the shutdown, which is not a backend failure
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert [entry for entry in io_log if entry != "close"] == ["error"]
    assert io_log.index("error") < io_log.index("close")


@pytest.mark.asyncio
@pytest.mark.parametrize("realtime", [False, True])
async def test_stt_request_waiting_for_a_slot_is_not_sent_after_stop(
    enhanced_handler, mock_info, mock_clients, caplog, realtime
):
    """A request still waiting for a slot when the server stops is cancelled and never reaches the backend."""
    from openai.types.audio.transcription import Transcription

    stt_client, _ = mock_clients
    if realtime:
        _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    stt_client.realtime.connect = _completing_realtime_connect()
    stt_client.audio.transcriptions.create = AsyncMock(return_value=Transcription(text="Hi"))
    limit = ConcurrencyLimit(1)
    other = await limit.acquire()  # Held by another connection
    (handler,) = _realtime_limited_handlers(enhanced_handler, mock_info, stt_client, limit, 1)
    audio_format = {"rate": 24000, "width": 2, "channels": 1}

    async def stream():
        await handler.handle_event(Event(type="audio-start", data=audio_format))
        await handler.handle_event(Event(type="audio-chunk", data=audio_format, payload=b"\x00\x01" * 10))
        return await handler.handle_event(Event(type="audio-stop"))

    request = asyncio.create_task(stream())
    await _settle()
    assert not request.done()

    with caplog.at_level(logging.INFO, logger="wyoming_openai.handler"):
        await handler.stop()
        other.release()  # The other connection is stopped too
        await asyncio.wait_for(request, timeout=1)

    assert "Cancelling the STT request in flight" in caplog.text
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]
    stt_client.realtime.connect.assert_not_called()
    stt_client.audio.transcriptions.create.assert_not_called()
    assert _written_event_types(handler) == ["error"]
    assert not limit.full


@pytest.mark.asyncio
async def test_stop_reports_a_realtime_failure_that_is_still_cleaning_up(enhanced_handler, mock_info, mock_clients):
    """stop() during the cleanup of a failed commit adds no second error to the failure's, sent before the close."""
    stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    connection = _FakeRealtimeConnection([])
    connection.input_audio_buffer.commit = AsyncMock(side_effect=RuntimeError("commit failed"))
    stt_client.realtime.connect = Mock(return_value=_FakeRealtimeConnectionManager(connection))
    handler = enhanced_handler
    await handler.handle_event(Event(type="audio-start", data={"rate": 24000, "width": 2, "channels": 1}))

    # A receive task slow to stop, so the failed audio-stop is still in its cleanup when stop() runs
    receiver_stops = asyncio.Event()
    original_receive_task = handler._realtime_receive_task
    assert original_receive_task is not None
    original_receive_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await original_receive_task

    async def slow_receiver():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await receiver_stops.wait()
            raise

    handler._realtime_receive_task = asyncio.create_task(slow_receiver())
    io_log = _record_client_io(handler)

    audio_stop = asyncio.create_task(handler.handle_event(Event(type="audio-stop")))
    await _settle()
    assert not audio_stop.done()
    stopping = asyncio.create_task(handler.stop())
    await _settle()
    receiver_stops.set()
    await asyncio.wait_for(asyncio.gather(audio_stop, stopping), timeout=1)

    assert io_log.count("error") == 1
    assert io_log.index("error") < io_log.index("close")
    assert "transcript" not in io_log


@pytest.mark.asyncio
@pytest.mark.parametrize(("realtime", "terminal"), [(True, "transcript"), (False, "transcript"), (True, "error")])
async def test_stop_adds_no_second_response_while_the_final_write_drains(
    enhanced_handler, mock_info, mock_clients, realtime, terminal
):
    """A transcript or error already queued is the client's response, even while its write waits on the client."""
    from openai.types.audio.transcription import Transcription

    stt_client, _ = mock_clients
    if realtime:
        _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
        if terminal == "error":
            connection = _FakeRealtimeConnection([])
            connection.input_audio_buffer.commit = AsyncMock(side_effect=RuntimeError("commit failed"))
            stt_client.realtime.connect = Mock(return_value=_FakeRealtimeConnectionManager(connection))
        else:
            stt_client.realtime.connect = _completing_realtime_connect()
    stt_client.audio.transcriptions.create = AsyncMock(return_value=Transcription(text="Hi"))
    handler = enhanced_handler
    audio_format = {"rate": 24000, "width": 2, "channels": 1}
    await handler.handle_event(Event(type="audio-start", data=audio_format))
    await handler.handle_event(Event(type="audio-chunk", data=audio_format, payload=b"\x00\x01" * 10))
    client_reads = asyncio.Event()
    written = []

    async def write_event(event):
        # Queued first, as Wyoming does with the whole event, then the drain waits for the client
        written.append(event.type)
        if event.type == terminal:
            await client_reads.wait()

    handler.write_event = AsyncMock(side_effect=write_event)

    audio_stop = asyncio.create_task(handler.handle_event(Event(type="audio-stop")))
    await _settle()
    assert written[-1] == terminal

    stopping = asyncio.create_task(handler.stop())
    await _settle()
    client_reads.set()
    await asyncio.wait_for(asyncio.gather(audio_stop, stopping, return_exceptions=True), timeout=3)

    assert written.count("error") == (1 if terminal == "error" else 0)
    assert written.count("transcript") == (1 if terminal == "transcript" else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("realtime", [False, True])
async def test_stop_cancels_stt_before_it_waits_on_the_synthesis(enhanced_handler, mock_info, mock_clients, realtime):
    """
    stop() cancels a request still waiting for a slot before it stops the synthesis, so a slot freed while the
    synthesis stops does not let that request reach the backend.
    """
    from openai.types.audio.transcription import Transcription

    stt_client, _ = mock_clients
    if realtime:
        _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    stt_client.realtime.connect = _completing_realtime_connect()
    stt_client.audio.transcriptions.create = AsyncMock(return_value=Transcription(text="Hi"))
    limit = ConcurrencyLimit(1)
    other = await limit.acquire()  # Held by another connection
    (handler,) = _realtime_limited_handlers(enhanced_handler, mock_info, stt_client, limit, 1)
    playback_stopping = asyncio.Event()
    playback_stopped = asyncio.Event()

    async def stop_sentence_playback(error=None):
        playback_stopping.set()
        await playback_stopped.wait()
        return True

    handler._stop_sentence_playback = stop_sentence_playback
    audio_format = {"rate": 24000, "width": 2, "channels": 1}

    async def stream():
        await handler.handle_event(Event(type="audio-start", data=audio_format))
        await handler.handle_event(Event(type="audio-chunk", data=audio_format, payload=b"\x00\x01" * 10))
        return await handler.handle_event(Event(type="audio-stop"))

    request = asyncio.create_task(stream())
    await _settle()
    assert not request.done()

    stopping = asyncio.create_task(handler.stop())
    await asyncio.wait_for(playback_stopping.wait(), timeout=1)
    other.release()  # The other connection's request ends while the synthesis stops
    await _settle()
    playback_stopped.set()
    await asyncio.wait_for(asyncio.gather(request, stopping), timeout=3)

    stt_client.realtime.connect.assert_not_called()
    stt_client.audio.transcriptions.create.assert_not_called()
    assert _written_event_types(handler) == ["error"]
    assert not limit.full


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True])
async def test_stop_during_the_http_transcript_start_sends_one_terminal_error(
    enhanced_handler, mock_clients, streamed
):
    """
    stop() running while transcript-start drains answers with its error, and nothing else of the transcript
    follows: neither the transcript of a non-streamed response nor the next chunk of a streamed one.
    """
    from openai.types.audio.transcription import Transcription

    stt_client, _ = mock_clients
    handler = enhanced_handler
    next_delta = asyncio.Event()

    class _SlowStream(_FakeTranscriptionStream):
        async def __aiter__(self):
            await next_delta.wait()
            yield Mock(type="transcript.text.delta", delta="Hi")
            yield Mock(type="transcript.text.done", text="Hi")

    stt_client.audio.transcriptions.create = AsyncMock(
        return_value=_SlowStream() if streamed else Transcription(text="Hi")
    )
    start_queued = asyncio.Event()
    client_reads = asyncio.Event()
    io_log = _record_client_io(handler)

    async def write_event(event):
        io_log.append(event.type)
        if event.type == "transcript-start":
            start_queued.set()
            await client_reads.wait()

    handler.write_event = AsyncMock(side_effect=write_event)

    with patch("wyoming_openai.handler.AsyncStream", _SlowStream):
        audio_stop = asyncio.create_task(_stream_audio(handler))
        await asyncio.wait_for(start_queued.wait(), timeout=1)
        stopping = asyncio.create_task(handler.stop())
        await _settle()
        client_reads.set()
        next_delta.set()
        await asyncio.wait_for(asyncio.gather(audio_stop, stopping), timeout=3)

    assert [entry for entry in io_log if entry != "close"] == ["transcript-start", "error"]
    assert io_log.index("error") < io_log.index("close")


@pytest.mark.asyncio
async def test_stop_cancels_a_realtime_session_still_opening(enhanced_handler, mock_info, mock_clients):
    """stop() while the session's settings are being sent cancels the opening: only the shutdown error is written."""
    stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    connection = _FakeRealtimeConnection([])
    settings_sent = asyncio.Event()
    update_entered = asyncio.Event()

    async def update(**kwargs):
        update_entered.set()
        await settings_sent.wait()

    connection.session.update = AsyncMock(side_effect=update)
    manager = _FakeRealtimeConnectionManager(connection)
    stt_client.realtime.connect = Mock(return_value=manager)
    handler = enhanced_handler
    written = []
    handler.write_event = AsyncMock(side_effect=lambda event: written.append(event.type))

    audio_start = asyncio.create_task(handler.handle_event(REALTIME_AUDIO_START))
    await asyncio.wait_for(update_entered.wait(), timeout=1)
    await asyncio.wait_for(handler.stop(), timeout=3)
    await asyncio.wait_for(audio_start, timeout=1)
    settings_sent.set()
    await _settle()

    assert written == ["error"]
    assert manager.exited


@pytest.mark.asyncio
async def test_stt_events_after_stop_are_ignored(enhanced_handler, mock_info, mock_clients):
    """Once stop() has answered the stream in progress, later STT events start nothing and write nothing."""
    stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    stt_client.realtime.connect = _completing_realtime_connect()
    handler = enhanced_handler
    written = []
    handler.write_event = AsyncMock(side_effect=lambda event: written.append(event.type))

    await handler.handle_event(REALTIME_AUDIO_START)
    await handler.stop()
    for event in (
        Event(type="transcribe", data={"name": "no-such-model"}),
        REALTIME_AUDIO_START,
        Event(type="audio-chunk", data={"rate": 24000, "width": 2, "channels": 1}, payload=b"\x00\x01" * 10),
        Event(type="audio-stop"),
    ):
        assert await handler.handle_event(event) is True

    assert written == ["transcript-start", "error"]
    assert stt_client.realtime.connect.call_count == 1


@pytest.mark.asyncio
async def test_audio_start_that_stop_overtakes_opens_no_session(enhanced_handler, mock_info, mock_clients):
    """An audio-start still closing the stream before it when stop() runs does not open a session afterwards."""
    stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    stt_client.realtime.connect = _completing_realtime_connect()
    handler = enhanced_handler
    await handler.handle_event(REALTIME_AUDIO_START)
    stop_queued = asyncio.Event()
    client_reads = asyncio.Event()

    async def write_event(event):
        if event.type == "transcript-stop":
            # The abandoned stream's transcript-stop drains on a client slow to read
            stop_queued.set()
            await client_reads.wait()

    handler.write_event = AsyncMock(side_effect=write_event)

    audio_start = asyncio.create_task(handler.handle_event(REALTIME_AUDIO_START))
    await asyncio.wait_for(stop_queued.wait(), timeout=1)
    await asyncio.wait_for(handler.stop(), timeout=3)
    client_reads.set()
    await asyncio.wait_for(audio_start, timeout=1)

    assert stt_client.realtime.connect.call_count == 1


@pytest.mark.asyncio
async def test_repeated_stop_sends_one_shutdown_error_while_recording(enhanced_handler):
    """A second stop() while the first one's error is still being written does not send another."""
    handler = enhanced_handler
    error_queued = asyncio.Event()
    client_reads = asyncio.Event()
    written = []

    async def write_event(event):
        written.append(event.type)
        if event.type == "error":
            error_queued.set()
            await client_reads.wait()

    handler.write_event = AsyncMock(side_effect=write_event)
    await handler.handle_event(Event(type="audio-start", data={"rate": 16000, "width": 2, "channels": 1}))

    first = asyncio.create_task(handler.stop())
    await asyncio.wait_for(error_queued.wait(), timeout=1)
    second = asyncio.create_task(handler.stop())
    await _settle()
    client_reads.set()
    await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), timeout=3)

    assert written.count("error") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("realtime", [False, True])
async def test_stop_answers_an_audio_stop_cancelled_before_its_request_started(
    enhanced_handler, mock_info, mock_clients, realtime
):
    """An audio-stop and stop() in the same turn of the loop: the request never runs, and stop() answers it."""
    from openai.types.audio.transcription import Transcription

    stt_client, _ = mock_clients
    if realtime:
        _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
        stt_client.realtime.connect = _completing_realtime_connect()
    stt_client.audio.transcriptions.create = AsyncMock(return_value=Transcription(text="Hi"))
    handler = enhanced_handler
    audio_format = {"rate": 24000, "width": 2, "channels": 1}
    await handler.handle_event(Event(type="audio-start", data=audio_format))
    await handler.handle_event(Event(type="audio-chunk", data=audio_format, payload=b"\x00\x01" * 10))
    written = []
    handler.write_event = AsyncMock(side_effect=lambda event: written.append(event.type))

    # Scheduled together: the audio-stop schedules its request, which stop() cancels before its first step
    audio_stop = asyncio.create_task(handler.handle_event(Event(type="audio-stop")))
    stopping = asyncio.create_task(handler.stop())
    await asyncio.wait_for(asyncio.gather(audio_stop, stopping), timeout=3)

    stt_client.audio.transcriptions.create.assert_not_called()
    assert written == ["error"]


@pytest.mark.asyncio
async def test_stop_waits_for_a_request_that_outlives_its_cancellation(enhanced_handler, mock_clients, monkeypatch):
    """
    A backend call that finishes despite its cancellation is the one answer: stop() waits for it, with no time
    limit, and adds no error after its transcript.
    """
    from openai.types.audio.transcription import Transcription

    monkeypatch.setattr("wyoming_openai.handler.SHUTDOWN_DRAIN_TIMEOUT", 0.01)
    stt_client, _ = mock_clients
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def stubborn_create(**kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await finish.wait()
        return Transcription(text="Hi")

    stt_client.audio.transcriptions.create = AsyncMock(side_effect=stubborn_create)
    handler = enhanced_handler
    io_log = _record_client_io(handler)

    request = asyncio.create_task(_stream_audio(handler))
    await asyncio.wait_for(entered.wait(), timeout=1)
    stopping = asyncio.create_task(handler.stop())
    await asyncio.sleep(0.05)
    assert not stopping.done()

    finish.set()
    await asyncio.wait_for(asyncio.gather(request, stopping), timeout=3)

    assert [entry for entry in io_log if entry != "close"] == ["transcript-start", "transcript", "transcript-stop"]
    assert io_log.index("transcript-stop") < io_log.index("close")


@pytest.mark.asyncio
async def test_concurrent_stops_let_a_cancelled_response_finish_closing(enhanced_handler, mock_clients):
    """A second stop() while the first waits for a streamed response to close does not cut that close short."""
    stt_client, _ = mock_clients
    streaming = asyncio.Event()
    close_may_finish = asyncio.Event()

    class _SlowToCloseStream(_FakeTranscriptionStream):
        async def __aiter__(self):
            streaming.set()
            await asyncio.Event().wait()
            yield

        async def __aexit__(self, exc_type, exc, tb):
            await close_may_finish.wait()
            self.closed = True

    stream = _SlowToCloseStream()
    stt_client.audio.transcriptions.create = AsyncMock(return_value=stream)
    handler = enhanced_handler

    with patch("wyoming_openai.handler.AsyncStream", _SlowToCloseStream):
        request = asyncio.create_task(_stream_audio(handler))
        await asyncio.wait_for(streaming.wait(), timeout=1)
        first = asyncio.create_task(handler.stop())
        await _settle()
        second = asyncio.create_task(handler.stop())
        await _settle()
        assert not second.done()

        close_may_finish.set()
        await asyncio.wait_for(asyncio.gather(request, first, second), timeout=3)

    assert stream.closed
    assert _written_event_types(handler).count("error") == 1


@pytest.mark.asyncio
async def test_cancelling_the_handler_during_realtime_startup_cleanup_ends_it(mock_info, mock_clients):
    """Ctrl+C while a failed session start waits for its receiver to stop still ends the handler."""
    stt_client, _ = mock_clients
    mock_info.asr[0].models[0].name = "gpt-realtime-whisper"
    receiver_cancelled = asyncio.Event()
    receiver_may_stop = asyncio.Event()

    class _SlowToStopConnection(_FakeRealtimeConnection):
        async def __aiter__(self):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                receiver_cancelled.set()
                await receiver_may_stop.wait()
                raise
            yield

    connection = _SlowToStopConnection([])

    async def reject_settings(**kwargs):
        # Fails once the receiver is running, so the cleanup has a receiver to wait for
        for _ in range(3):
            await asyncio.sleep(0)
        raise RuntimeError("settings rejected")

    connection.session.update = AsyncMock(side_effect=reject_settings)
    stt_client.realtime.connect = Mock(return_value=_FakeRealtimeConnectionManager(connection))
    reader = asyncio.StreamReader()
    writer = MagicMock(name="writer")
    writer.wait_closed = AsyncMock()
    handler = OpenAIEventHandler(
        reader,
        writer,
        info=mock_info,
        stt_client=stt_client,
        tts_client=None,
        stt_realtime_models=["gpt-realtime-whisper"],
    )
    handler.write_event = AsyncMock()
    reader.feed_data(
        json.dumps({"type": "audio-start", "data": {"rate": 24000, "width": 2, "channels": 1}}).encode() + b"\n"
    )

    run_task = asyncio.create_task(handler.run())
    try:
        # The session failed to start, and its cleanup waits for the receiver to stop
        await asyncio.wait_for(receiver_cancelled.wait(), timeout=1)
        run_task.cancel()
        await _settle()
        receiver_may_stop.set()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(run_task, timeout=1)
        assert run_task.cancelled()
    finally:
        receiver_may_stop.set()
        if not run_task.done():
            run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["http-stop", "http-self-cancel", "realtime-stop", "realtime-start"])
async def test_a_cancelled_error_from_the_backend_fails_the_request(enhanced_handler, mock_info, mock_clients, stage):
    """
    A CancelledError the backend raises, or a cancellation it requests of its own task, without a shutdown fails the
    request like other errors do.
    """
    stt_client, _ = mock_clients
    handler = enhanced_handler
    if stage == "http-stop":
        stt_client.audio.transcriptions.create = AsyncMock(side_effect=asyncio.CancelledError)
    elif stage == "http-self-cancel":

        async def self_cancelling_create(**kwargs):
            task = asyncio.current_task()
            assert task is not None
            task.cancel()
            await asyncio.sleep(0)

        stt_client.audio.transcriptions.create = AsyncMock(side_effect=self_cancelling_create)
    else:
        _use_realtime_stt_model(handler, mock_info, mock_clients)
        connection = _FakeRealtimeConnection([])
        if stage == "realtime-stop":
            connection.input_audio_buffer.commit = AsyncMock(side_effect=asyncio.CancelledError)
        else:
            connection.session.update = AsyncMock(side_effect=asyncio.CancelledError)
        stt_client.realtime.connect = Mock(return_value=_FakeRealtimeConnectionManager(connection))
    written = []
    handler.write_event = AsyncMock(
        side_effect=lambda event: written.append(Error.from_event(event).text if event.type == "error" else event.type)
    )
    audio_format = {"rate": 24000, "width": 2, "channels": 1}

    await asyncio.wait_for(handler.handle_event(Event(type="audio-start", data=audio_format)), timeout=1)
    await handler.handle_event(Event(type="audio-chunk", data=audio_format, payload=b"\x00\x01" * 10))
    assert await asyncio.wait_for(handler.handle_event(Event(type="audio-stop")), timeout=1) is False

    assert written.count(STT_FAILED_ERROR) == 1
    assert written[-1] == STT_FAILED_ERROR


@pytest.mark.asyncio
async def test_cancelling_the_handler_ends_it_even_if_the_backend_ignores_the_cancellation(mock_info, mock_clients):
    """Ctrl+C during a request whose backend call catches the cancellation and returns still ends the handler."""
    from openai.types.audio.transcription import Transcription

    stt_client, _ = mock_clients
    entered = asyncio.Event()

    async def stubborn_create(**kwargs):
        entered.set()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.Event().wait()
        return Transcription(text="Hi")

    stt_client.audio.transcriptions.create = AsyncMock(side_effect=stubborn_create)
    reader = asyncio.StreamReader()
    writer = MagicMock(name="writer")
    writer.wait_closed = AsyncMock()
    handler = OpenAIEventHandler(reader, writer, info=mock_info, stt_client=stt_client, tts_client=None)
    handler.write_event = AsyncMock()
    for event in ({"type": "audio-start", "data": {"rate": 16000, "width": 2, "channels": 1}}, {"type": "audio-stop"}):
        reader.feed_data(json.dumps(event).encode() + b"\n")

    run_task = asyncio.create_task(handler.run())
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(run_task, timeout=1)
        assert run_task.cancelled()
    finally:
        if not run_task.done():
            run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_receiver_stopped_by_a_failed_chunk_writes_nothing_after_the_shutdown_error(
    enhanced_handler, mock_info, mock_clients
):
    """
    A receiver that an audio-chunk's cleanup is still stopping when stop() answers writes nothing more, even if the
    connection hands it one last delta.
    """
    stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    receiver_cancelled = asyncio.Event()
    last_delta = asyncio.Event()

    class _LastDeltaConnection(_FakeRealtimeConnection):
        async def __aiter__(self):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                receiver_cancelled.set()
                # Clears its cancellation too, so the receiver cannot tell from that alone that it is being stopped
                task = asyncio.current_task()
                assert task is not None
                task.uncancel()
                await last_delta.wait()
                yield _FakeRealtimeServerEvent("conversation.item.input_audio_transcription.delta", delta="late")
                raise

    connection = _LastDeltaConnection([])
    connection.input_audio_buffer.append = AsyncMock(side_effect=RuntimeError("socket gone"))
    stt_client.realtime.connect = Mock(return_value=_FakeRealtimeConnectionManager(connection))
    handler = enhanced_handler
    written = []
    handler.write_event = AsyncMock(side_effect=lambda event: written.append(event.type))
    audio_format = {"rate": 24000, "width": 2, "channels": 1}

    await handler.handle_event(Event(type="audio-start", data=audio_format))
    await _settle()  # The receiver waits for events
    # The append fails, and the chunk's cleanup waits for the receiver to stop
    chunk = asyncio.create_task(
        handler.handle_event(Event(type="audio-chunk", data=audio_format, payload=b"\x00\x01" * 10))
    )
    await asyncio.wait_for(receiver_cancelled.wait(), timeout=1)
    await asyncio.wait_for(handler.stop(), timeout=3)
    last_delta.set()
    await asyncio.wait_for(chunk, timeout=1)

    assert written[written.index("error"):] == ["error"]
    assert "transcript-chunk" not in written


@pytest.mark.asyncio
async def test_server_stop_answers_a_half_closed_client_whose_session_start_ignored_the_cancellation(
    mock_info, mock_clients
):
    """
    Over a real connection: a Realtime start whose settings update catches the shutdown's cancellation and returns
    still leaves the answer to stop(), though the client stopped sending and the server has read its end of stream.
    """
    stt_client, _ = mock_clients
    mock_info.asr[0].models[0].name = "gpt-realtime-whisper"
    connection = _FakeRealtimeConnection([])
    update_entered = asyncio.Event()

    async def stubborn_update(**kwargs):
        update_entered.set()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.Event().wait()

    connection.session.update = AsyncMock(side_effect=stubborn_update)
    stt_client.realtime.connect = Mock(return_value=_FakeRealtimeConnectionManager(connection))
    handlers: list[OpenAIEventHandler] = []

    def handler_factory(reader, writer):
        handler = OpenAIEventHandler(
            reader,
            writer,
            info=mock_info,
            stt_client=stt_client,
            tts_client=None,
            stt_realtime_models=["gpt-realtime-whisper"],
        )
        handlers.append(handler)
        return handler

    server = AsyncTcpServer("127.0.0.1", 0)
    server_task = asyncio.create_task(server.run(handler_factory), name="wyoming-test-server")
    client_writer: asyncio.StreamWriter | None = None
    try:
        async with asyncio.timeout(2):
            while server._server is None:
                await asyncio.sleep(0)
        listener = server._server
        assert isinstance(listener, asyncio.Server)
        port = listener.sockets[0].getsockname()[1]
        client_reader, client_writer = await asyncio.open_connection("127.0.0.1", port)
        await async_write_event(
            Event(type="audio-start", data={"rate": 24000, "width": 2, "channels": 1}), client_writer
        )
        await asyncio.wait_for(update_entered.wait(), timeout=2)
        client_writer.write_eof()
        async with asyncio.timeout(2):
            while not handlers[0].reader.at_eof():
                await asyncio.sleep(0.01)

        server._stop_requested = True
        received = []
        async with asyncio.timeout(3):
            await server.stop()
            while (event := await async_read_event(client_reader)) is not None:
                received.append(event)

        # The start that caught the cancellation goes no further, so the session's transcript is never started
        assert [event.type for event in received] == ["error"]
        assert Error.from_event(received[-1]).text == SHUTDOWN_ERROR
    finally:
        if client_writer is not None:
            client_writer.close()
            with contextlib.suppress(Exception):
                await client_writer.wait_closed()
        if not server_task.done():
            server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await server_task


@pytest.mark.asyncio
async def test_cancelling_the_handler_during_a_failed_chunks_cleanup_ends_it(mock_info, mock_clients):
    """
    Ctrl+C while a failed chunk's cleanup waits for the receiver ends the handler, even if the receiver catches the
    cancellation and ends normally.
    """
    stt_client, _ = mock_clients
    mock_info.asr[0].models[0].name = "gpt-realtime-whisper"
    receiver_cancelled = asyncio.Event()
    may_yield = asyncio.Event()

    class _StubbornConnection(_FakeRealtimeConnection):
        async def __aiter__(self):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                receiver_cancelled.set()
                while not may_yield.is_set():
                    with contextlib.suppress(asyncio.CancelledError):
                        await may_yield.wait()
                yield _FakeRealtimeServerEvent("conversation.item.input_audio_transcription.delta", delta="late")
                raise

    connection = _StubbornConnection([])
    connection.input_audio_buffer.append = AsyncMock(side_effect=RuntimeError("socket gone"))
    stt_client.realtime.connect = Mock(return_value=_FakeRealtimeConnectionManager(connection))
    reader = asyncio.StreamReader()
    writer = MagicMock(name="writer")
    writer.wait_closed = AsyncMock()
    handler = OpenAIEventHandler(
        reader,
        writer,
        info=mock_info,
        stt_client=stt_client,
        tts_client=None,
        stt_realtime_models=["gpt-realtime-whisper"],
    )
    handler.write_event = AsyncMock()
    audio_format = {"rate": 24000, "width": 2, "channels": 1}
    payload = b"\x00\x01" * 10

    run_task = asyncio.create_task(handler.run())
    try:
        reader.feed_data(json.dumps({"type": "audio-start", "data": audio_format}).encode() + b"\n")
        await _settle()  # The session is open, and the receiver waits for events
        reader.feed_data(
            json.dumps({"type": "audio-chunk", "data": audio_format, "payload_length": len(payload)}).encode()
            + b"\n"
            + payload
        )
        await asyncio.wait_for(receiver_cancelled.wait(), timeout=1)
        run_task.cancel()
        await _settle()
        may_yield.set()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(run_task, timeout=1)
        assert run_task.cancelled()
    finally:
        may_yield.set()
        if not run_task.done():
            run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_a_cancelled_error_from_the_realtime_connection_fails_the_session(
    enhanced_handler, mock_info, mock_clients
):
    """A CancelledError the connection raises by itself fails the session: the client is told, and the slot freed."""
    stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    drop = asyncio.Event()

    class _DroppingConnection(_FakeRealtimeConnection):
        async def __aiter__(self):
            await drop.wait()
            raise asyncio.CancelledError
            yield

    stt_client.realtime.connect = Mock(return_value=_FakeRealtimeConnectionManager(_DroppingConnection([])))
    limit = ConcurrencyLimit(1)
    (handler,) = _realtime_limited_handlers(enhanced_handler, mock_info, stt_client, limit, 1)
    written = []
    handler.write_event = AsyncMock(
        side_effect=lambda event: written.append(Error.from_event(event).text if event.type == "error" else event.type)
    )
    audio_format = {"rate": 24000, "width": 2, "channels": 1}

    await handler.handle_event(Event(type="audio-start", data=audio_format))
    drop.set()
    await _settle()
    await handler.handle_event(Event(type="audio-chunk", data=audio_format, payload=b"\x00\x01" * 10))

    assert written == ["transcript-start", STT_FAILED_ERROR]
    assert not limit.full


@pytest.mark.asyncio
async def test_disconnect_cancelled_while_waiting_for_stop_leaves_the_connection_to_it(enhanced_handler, mock_clients):
    """
    A disconnect() cancelled while it waits for stop() closes the recording but not the connection, which stop()
    still answers before it closes it.
    """
    from openai.types.audio.transcription import Transcription

    stt_client, _ = mock_clients
    stt_client.audio.transcriptions.create = AsyncMock(return_value=Transcription(text="Hi"))
    handler = enhanced_handler
    audio_format = {"rate": 16000, "width": 2, "channels": 1}
    await handler.handle_event(Event(type="audio-start", data=audio_format))
    await handler.handle_event(Event(type="audio-chunk", data=audio_format, payload=b"\x00\x01" * 10))
    playback_stopping = asyncio.Event()
    playback_stopped = asyncio.Event()

    async def stop_sentence_playback(error=None):
        playback_stopping.set()
        await playback_stopped.wait()
        return True

    handler._stop_sentence_playback = stop_sentence_playback
    io_log = _record_client_io(handler)

    # Scheduled together, so the audio-stop's request never starts and leaves its recording behind
    audio_stop = asyncio.create_task(handler.handle_event(Event(type="audio-stop")))
    stopping = asyncio.create_task(handler.stop())
    await asyncio.wait_for(playback_stopping.wait(), timeout=1)
    assert await asyncio.wait_for(audio_stop, timeout=1) is False
    disconnecting = asyncio.create_task(handler.disconnect())
    await _settle()
    disconnecting.cancel()
    await asyncio.gather(disconnecting, return_exceptions=True)

    assert "close" not in io_log
    assert handler._wav_buffer is None
    playback_stopped.set()
    await asyncio.wait_for(stopping, timeout=3)

    assert [entry for entry in io_log if entry != "close"] == ["error"]
    assert io_log.index("error") < io_log.index("close")
    stt_client.audio.transcriptions.create.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["commit", "session-update"])
async def test_no_realtime_operation_starts_after_the_shutdown_began(enhanced_handler, mock_info, mock_clients, stage):
    """
    A backend call that catches the shutdown's cancellation and returns is the request's last: neither the commit
    after the final flush nor the settings after the connect are sent.
    """
    stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    connection = _FakeRealtimeConnection([])
    commit = AsyncMock()
    stubborn_call = asyncio.Event()

    async def stubborn():
        stubborn_call.set()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.Event().wait()

    if stage == "commit":
        appends = 0

        async def append(*, audio):
            nonlocal appends
            appends += 1
            if appends == 2:  # The final flush of the resampler
                await stubborn()

        connection.input_audio_buffer.append = AsyncMock(side_effect=append)
        connection.input_audio_buffer.commit = commit
        manager = _FakeRealtimeConnectionManager(connection)
    else:

        class _StubbornEnterManager(_FakeRealtimeConnectionManager):
            async def enter(self):
                await stubborn()
                return await super().enter()

        manager = _StubbornEnterManager(connection)
        connection.session.update = AsyncMock()
    stt_client.realtime.connect = Mock(return_value=manager)
    handler = enhanced_handler
    written = []
    handler.write_event = AsyncMock(side_effect=lambda event: written.append(event.type))
    audio_format = {"rate": 16000, "width": 2, "channels": 1}

    if stage == "commit":
        await handler.handle_event(Event(type="audio-start", data=audio_format))
        await handler.handle_event(Event(type="audio-chunk", data=audio_format, payload=b"\x00\x00" * 10))
        request = asyncio.create_task(handler.handle_event(Event(type="audio-stop")))
    else:
        request = asyncio.create_task(handler.handle_event(Event(type="audio-start", data=audio_format)))
    await asyncio.wait_for(stubborn_call.wait(), timeout=1)
    await asyncio.wait_for(handler.stop(), timeout=3)
    await asyncio.wait_for(request, timeout=1)

    if stage == "commit":
        commit.assert_not_called()
    else:
        connection.session.update.assert_not_called()
    assert written.count("error") == 1
    assert written[-1] == "error"


@pytest.mark.asyncio
async def test_stop_during_the_cleanup_of_a_failed_realtime_start_sends_one_error_before_the_close(
    enhanced_handler, mock_info, mock_clients
):
    """
    A Realtime start that fails on a CancelledError of the backend's own, with stop() running while its session
    closes, gets exactly one error, before the connection closes.
    """
    stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    receiver_cancelled = asyncio.Event()
    receiver_may_stop = asyncio.Event()

    class _SlowToStopConnection(_FakeRealtimeConnection):
        async def __aiter__(self):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                receiver_cancelled.set()
                await receiver_may_stop.wait()
                raise
            yield

    connection = _SlowToStopConnection([])

    async def cancelling_update(**kwargs):
        # Fails once the receiver is running, so closing the session has a receiver to wait for
        for _ in range(3):
            await asyncio.sleep(0)
        raise asyncio.CancelledError

    connection.session.update = AsyncMock(side_effect=cancelling_update)
    stt_client.realtime.connect = Mock(return_value=_FakeRealtimeConnectionManager(connection))
    handler = enhanced_handler
    io_log = _record_client_io(handler)

    audio_start = asyncio.create_task(handler.handle_event(REALTIME_AUDIO_START))
    await asyncio.wait_for(receiver_cancelled.wait(), timeout=1)
    stopping = asyncio.create_task(handler.stop())
    await _settle()
    receiver_may_stop.set()
    await asyncio.wait_for(asyncio.gather(audio_start, stopping), timeout=3)

    assert io_log.count("error") == 1
    assert io_log.index("error") < io_log.index("close")


@pytest.mark.asyncio
async def test_a_streamed_response_returned_despite_the_shutdown_is_not_read(enhanced_handler, mock_clients):
    """A create() that catches the shutdown's cancellation and returns a stream: the stream is closed, not read."""
    stt_client, _ = mock_clients
    entered = asyncio.Event()

    class _WatchedStream(_FakeTranscriptionStream):
        read = False

        async def __aiter__(self):
            self.read = True
            yield Mock(type="transcript.text.delta", delta="Hi")
            yield Mock(type="transcript.text.done", text="Hi")

    stream = _WatchedStream()

    async def stubborn_create(**kwargs):
        entered.set()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.Event().wait()
        return stream

    stt_client.audio.transcriptions.create = AsyncMock(side_effect=stubborn_create)
    handler = enhanced_handler
    io_log = _record_client_io(handler)

    with patch("wyoming_openai.handler.AsyncStream", _WatchedStream):
        request = asyncio.create_task(_stream_audio(handler))
        await asyncio.wait_for(entered.wait(), timeout=1)
        await asyncio.wait_for(handler.stop(), timeout=3)
        await asyncio.wait_for(request, timeout=1)

    assert not stream.read
    assert stream.closed
    assert [entry for entry in io_log if entry != "close"] == ["error"]


@pytest.mark.asyncio
async def test_realtime_start_cancelled_by_a_cancelled_stop_closes_its_own_session(
    enhanced_handler, mock_info, mock_clients
):
    """
    A Realtime start that stop() cancelled closes what it opened itself, so its session and slot are not left
    behind when stop() is cancelled before it could clean up.
    """
    stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    connection = _FakeRealtimeConnection([])
    update_entered = asyncio.Event()
    may_return = asyncio.Event()

    async def stubborn_update(**kwargs):
        update_entered.set()
        while not may_return.is_set():
            with contextlib.suppress(asyncio.CancelledError):
                await may_return.wait()

    connection.session.update = AsyncMock(side_effect=stubborn_update)
    manager = _FakeRealtimeConnectionManager(connection)
    stt_client.realtime.connect = Mock(return_value=manager)
    limit = ConcurrencyLimit(1)
    (handler,) = _realtime_limited_handlers(enhanced_handler, mock_info, stt_client, limit, 1)

    audio_start = asyncio.create_task(handler.handle_event(REALTIME_AUDIO_START))
    await asyncio.wait_for(update_entered.wait(), timeout=1)
    stopping = asyncio.create_task(handler.stop())
    await _settle()
    stopping.cancel()
    await asyncio.gather(stopping, return_exceptions=True)
    assert limit.full  # stop() was cancelled before it could close the session

    may_return.set()
    await asyncio.wait_for(audio_start, timeout=1)
    await asyncio.wait_for(handler._drain_background_tasks(), timeout=1)

    assert not limit.full
    assert manager.exited
    assert handler._realtime_receive_task is None


@pytest.mark.asyncio
@pytest.mark.parametrize("realtime", [False, True])
async def test_a_cancelled_error_after_the_transcript_adds_no_second_answer(
    enhanced_handler, mock_info, mock_clients, realtime
):
    """A CancelledError while transcript-stop is written, after the transcript was queued, sends no error after it."""
    from openai.types.audio.transcription import Transcription

    stt_client, _ = mock_clients
    if realtime:
        _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
        stt_client.realtime.connect = _completing_realtime_connect()
    stt_client.audio.transcriptions.create = AsyncMock(return_value=Transcription(text="Hi"))
    handler = enhanced_handler
    audio_format = {"rate": 24000, "width": 2, "channels": 1}
    await handler.handle_event(Event(type="audio-start", data=audio_format))
    await handler.handle_event(Event(type="audio-chunk", data=audio_format, payload=b"\x00\x01" * 10))
    written = []

    async def write_event(event):
        written.append(event.type)
        if event.type == "transcript-stop":
            raise asyncio.CancelledError

    handler.write_event = AsyncMock(side_effect=write_event)

    assert await asyncio.wait_for(handler.handle_event(Event(type="audio-stop")), timeout=1) is False

    assert written.count("transcript") == 1
    assert "error" not in written


@pytest.mark.asyncio
async def test_cancelled_stop_and_disconnect_leave_no_realtime_session_behind(
    enhanced_handler, mock_info, mock_clients
):
    """
    An audio-stop whose request never started, with stop() cancelled before it cleaned up and disconnect() cancelled
    while it waited for stop(): the open Realtime session is still closed, and its slot freed.
    """
    stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    socket_closed = asyncio.Event()
    # A closing handshake that takes a while, so ending before it finishes would leave it to the event loop's teardown
    manager = _GatedCloseManager(_FakeRealtimeConnection([]), socket_closed)
    stt_client.realtime.connect = Mock(return_value=manager)
    limit = ConcurrencyLimit(1)
    (handler,) = _realtime_limited_handlers(enhanced_handler, mock_info, stt_client, limit, 1)
    client_closed = asyncio.Event()

    async def wait_closed():
        await client_closed.wait()

    # stop() then waits in its cleanup for the client connection to close, so disconnect() waits for it meanwhile
    handler.writer.wait_closed = AsyncMock(side_effect=wait_closed)

    await handler.handle_event(REALTIME_AUDIO_START)
    assert limit.full
    # Scheduled together: the audio-stop's request is cancelled before its first step
    audio_stop = asyncio.create_task(handler.handle_event(Event(type="audio-stop")))
    stopping = asyncio.create_task(handler.stop())
    await asyncio.sleep(0)
    # Cancelled while it waits for that request, so before it closes the Realtime session
    stopping.cancel()
    assert await asyncio.wait_for(audio_stop, timeout=1) is False
    await _settle()
    assert not stopping.done()

    disconnecting = asyncio.create_task(handler.disconnect())
    await _settle()
    disconnecting.cancel()
    await _settle()

    assert not limit.full
    assert handler._realtime_receive_task is None
    # disconnect() waits for the websocket close it started, since stop() has drained its own already
    assert not disconnecting.done()
    socket_closed.set()
    await asyncio.gather(disconnecting, return_exceptions=True)
    assert manager.exited
    client_closed.set()
    await asyncio.gather(stopping, return_exceptions=True)


@pytest.mark.asyncio
async def test_background_drain_is_bounded_when_a_close_ignores_its_cancellation(enhanced_handler, caplog):
    """A cleanup task that catches its cancellation and keeps going cannot hold a bounded drain up past its limit."""
    handler = enhanced_handler
    loop = asyncio.get_running_loop()
    may_end = asyncio.Event()
    # Ends on its own after a while, so a drain that waits for it unbounded makes this test fail rather than hang
    loop.call_later(0.5, may_end.set)

    async def stubborn_close():
        while not may_end.is_set():
            with contextlib.suppress(asyncio.CancelledError):
                await may_end.wait()

    handler._run_in_background(stubborn_close(), name="stubborn_close")
    try:
        started = loop.time()
        with caplog.at_level(logging.WARNING, logger="wyoming_openai.handler"):
            await handler._drain_background_tasks(0.01)
        assert loop.time() - started < 0.3
        assert "did not end once cancelled" in caplog.text
    finally:
        await asyncio.wait_for(handler._drain_background_tasks(), timeout=2)


@pytest.mark.asyncio
async def test_an_interrupted_background_drain_cancels_what_it_waited_for(enhanced_handler):
    """A drain that is itself cancelled cancels the cleanup it was waiting for instead of leaving it to nobody."""
    handler = enhanced_handler
    close_cancelled = asyncio.Event()

    async def slow_close():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            close_cancelled.set()
            raise

    handler._run_in_background(slow_close(), name="slow_close")
    draining = asyncio.create_task(handler._drain_background_tasks())
    await _settle()
    draining.cancel()
    await asyncio.gather(draining, return_exceptions=True)
    await _settle()

    assert close_cancelled.is_set()
    assert not handler._background_tasks


@pytest.mark.asyncio
async def test_disconnect_cancelled_twice_leaves_no_websocket_close_behind(enhanced_handler, mock_info, mock_clients):
    """
    A disconnect() cancelled again while it waits for the websocket close its cleanup started cancels that close
    rather than leaving it to nobody.
    """
    stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    # A closing handshake that never finishes on its own
    manager = _GatedCloseManager(_FakeRealtimeConnection([]), asyncio.Event())
    stt_client.realtime.connect = Mock(return_value=manager)
    limit = ConcurrencyLimit(1)
    (handler,) = _realtime_limited_handlers(enhanced_handler, mock_info, stt_client, limit, 1)
    client_closed = asyncio.Event()

    async def wait_closed():
        await client_closed.wait()

    handler.writer.wait_closed = AsyncMock(side_effect=wait_closed)

    await handler.handle_event(REALTIME_AUDIO_START)
    # As in the test above: stop() is cancelled before it closes the session, and disconnect() while it waits
    audio_stop = asyncio.create_task(handler.handle_event(Event(type="audio-stop")))
    stopping = asyncio.create_task(handler.stop())
    await asyncio.sleep(0)
    stopping.cancel()
    assert await asyncio.wait_for(audio_stop, timeout=1) is False
    await _settle()
    disconnecting = asyncio.create_task(handler.disconnect())
    await _settle()
    disconnecting.cancel()
    await _settle()
    assert not disconnecting.done()  # Waiting for the held websocket close

    disconnecting.cancel()
    await asyncio.gather(disconnecting, return_exceptions=True)
    await _settle()

    assert not handler._background_tasks
    assert not limit.full
    client_closed.set()
    await asyncio.gather(stopping, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("uncancel", [False, True])
async def test_overlapping_drains_cancel_a_cleanup_task_once(enhanced_handler, uncancel):
    """
    A drain interrupted while a close it shares with an earlier, timed-out drain still unwinds from that drain's
    cancellation does not cancel it again, which would cut its own cleanup short, also if the close cleared its
    cancellation count.
    """
    handler = enhanced_handler
    unwinding = asyncio.Event()
    may_finish = asyncio.Event()
    outcome = {}

    async def close():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            unwinding.set()
            if uncancel:
                task = asyncio.current_task()
                assert task is not None
                task.uncancel()
            try:
                await may_finish.wait()
                outcome["finished"] = True
            except asyncio.CancelledError:
                outcome["finished"] = False
                raise
            raise

    handler._run_in_background(close(), name="close")
    # As stop()'s drain: times out, cancels the close, and gives up on it while it unwinds
    await handler._drain_background_tasks(0.01)
    assert unwinding.is_set()
    # As disconnect()'s, over the same close, and itself interrupted
    draining = asyncio.create_task(handler._drain_background_tasks())
    await _settle()
    draining.cancel()
    await asyncio.gather(draining, return_exceptions=True)
    may_finish.set()
    await _settle()

    assert outcome == {"finished": True}
    assert not handler._background_tasks


@pytest.mark.asyncio
async def test_disconnect_cancelled_again_while_stopping_the_receiver_still_waits_for_the_close(
    enhanced_handler, mock_info, mock_clients
):
    """
    A disconnect() cancelled again while its cleanup waits for the Realtime receiver still waits for the websocket
    close that cleanup starts.
    """
    stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    receiver_cancelled = asyncio.Event()

    class _SlowToStopConnection(_FakeRealtimeConnection):
        async def __aiter__(self):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                receiver_cancelled.set()
                await asyncio.Event().wait()
            yield

    socket_closed = asyncio.Event()
    manager = _GatedCloseManager(_SlowToStopConnection([]), socket_closed)
    stt_client.realtime.connect = Mock(return_value=manager)
    limit = ConcurrencyLimit(1)
    (handler,) = _realtime_limited_handlers(enhanced_handler, mock_info, stt_client, limit, 1)
    client_closed = asyncio.Event()

    async def wait_closed():
        await client_closed.wait()

    handler.writer.wait_closed = AsyncMock(side_effect=wait_closed)

    await handler.handle_event(REALTIME_AUDIO_START)
    await _settle()  # The receiver waits for events
    # stop() is cancelled before it closes the session, and disconnect() while it waits for stop()
    audio_stop = asyncio.create_task(handler.handle_event(Event(type="audio-stop")))
    stopping = asyncio.create_task(handler.stop())
    await asyncio.sleep(0)
    stopping.cancel()
    assert await asyncio.wait_for(audio_stop, timeout=1) is False
    await _settle()
    disconnecting = asyncio.create_task(handler.disconnect())
    await _settle()
    disconnecting.cancel()
    await asyncio.wait_for(receiver_cancelled.wait(), timeout=1)
    await _settle()
    # Cancelled again while its cleanup waits for the receiver
    disconnecting.cancel()
    await _settle()

    assert not disconnecting.done()  # Waiting for the websocket close
    socket_closed.set()
    await asyncio.gather(disconnecting, return_exceptions=True)
    assert manager.exited
    assert not handler._background_tasks
    assert not limit.full
    client_closed.set()
    await asyncio.gather(stopping, return_exceptions=True)


@pytest.mark.asyncio
async def test_no_read_of_a_streamed_response_starts_after_a_transcript_start_that_outlived_the_shutdown(
    enhanced_handler, mock_clients
):
    """A transcript-start write that catches the shutdown's cancellation and returns is followed by no read."""
    stt_client, _ = mock_clients
    start_written = asyncio.Event()

    class _WatchedStream(_FakeTranscriptionStream):
        read = False

        async def __aiter__(self):
            self.read = True
            yield Mock(type="transcript.text.delta", delta="Hi")
            yield Mock(type="transcript.text.done", text="Hi")

    stream = _WatchedStream()
    stt_client.audio.transcriptions.create = AsyncMock(return_value=stream)
    handler = enhanced_handler
    written = []

    async def write_event(event):
        written.append(event.type)
        if event.type == "transcript-start":
            start_written.set()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.Event().wait()

    handler.write_event = AsyncMock(side_effect=write_event)

    with patch("wyoming_openai.handler.AsyncStream", _WatchedStream):
        request = asyncio.create_task(_stream_audio(handler))
        await asyncio.wait_for(start_written.wait(), timeout=1)
        await asyncio.wait_for(handler.stop(), timeout=3)
        await asyncio.wait_for(request, timeout=1)

    assert not stream.read
    assert stream.closed
    assert written == ["transcript-start", "error"]


@pytest.mark.asyncio
async def test_a_stream_cut_off_by_the_shutdown_is_not_taken_for_the_transcript(enhanced_handler, mock_clients):
    """
    A streamed response that ends without its completed transcription because the shutdown cut it off is answered
    by stop(), not with the deltas so far as the transcript.
    """
    stt_client, _ = mock_clients
    delta_read = asyncio.Event()

    class _CutOffStream(_FakeTranscriptionStream):
        async def __aiter__(self):
            yield Mock(type="transcript.text.delta", delta="Hel")
            delta_read.set()
            # Cut off by the shutdown's cancellation, which it takes as the end of the stream
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.Event().wait()

    stt_client.audio.transcriptions.create = AsyncMock(return_value=_CutOffStream())
    handler = enhanced_handler
    written = []
    handler.write_event = AsyncMock(side_effect=lambda event: written.append(event.type))

    with patch("wyoming_openai.handler.AsyncStream", _CutOffStream):
        request = asyncio.create_task(_stream_audio(handler))
        await asyncio.wait_for(delta_read.wait(), timeout=1)
        await asyncio.wait_for(handler.stop(), timeout=3)
        await asyncio.wait_for(request, timeout=1)

    assert written == ["transcript-start", "transcript-chunk", "error"]


@pytest.mark.asyncio
async def test_a_stream_that_ends_without_completion_outside_a_shutdown_keeps_its_deltas(
    enhanced_handler, mock_clients
):
    """Outside a shutdown, the deltas of a stream that ends without its completed transcription are the transcript."""
    stt_client, _ = mock_clients

    class _DeltasOnlyStream(_FakeTranscriptionStream):
        async def __aiter__(self):
            yield Mock(type="transcript.text.delta", delta="Hel")
            yield Mock(type="transcript.text.delta", delta="lo")

    stt_client.audio.transcriptions.create = AsyncMock(return_value=_DeltasOnlyStream())
    handler = enhanced_handler

    with patch("wyoming_openai.handler.AsyncStream", _DeltasOnlyStream):
        await _stream_audio(handler)

    assert _written_transcripts(handler) == ["Hello"]


@pytest.mark.asyncio
async def test_stt_limit_is_shared_by_http_and_realtime_requests(mock_info, mock_clients):
    """A Realtime session on one connection holds the only slot; an HTTP request on another waits for it to end."""
    from openai.types.audio.transcription import Transcription

    stt_client, _ = mock_clients
    stt_client.realtime.connect = _completing_realtime_connect()
    stt_client.audio.transcriptions.create = AsyncMock(return_value=Transcription(text="Hello"))
    limit = ConcurrencyLimit(1)
    realtime_handler = _stt_limited_handler(mock_info, stt_client, limit)
    realtime_handler._stt_realtime_models = {mock_info.asr[0].models[0].name}
    http_handler = _stt_limited_handler(mock_info, stt_client, limit)

    await realtime_handler.handle_event(REALTIME_AUDIO_START)
    assert limit.full
    http_request = asyncio.create_task(_stream_audio(http_handler))
    await _settle()
    stt_client.audio.transcriptions.create.assert_not_called()

    assert await realtime_handler.handle_event(Event(type="audio-stop"))
    await asyncio.wait_for(http_request, timeout=1)

    assert _written_transcripts(realtime_handler) == ["Hi"]
    assert _written_transcripts(http_handler) == ["Hello"]
    assert not limit.full


@pytest.mark.asyncio
async def test_tts_limit_is_shared_across_connections(dummy_info, dummy_clients, dummy_reader_writer):
    """Connections sharing a TTS limit of 2 have at most 2 requests in flight between them."""
    stt_client, tts_client = dummy_clients
    reader, writer = dummy_reader_writer
    speech = _OverlapRecordingSpeech()
    tts_client.audio.speech.with_streaming_response = speech
    limit = ConcurrencyLimit(2)
    handlers = [
        OpenAIEventHandler(
            reader, writer, info=dummy_info, stt_client=stt_client, tts_client=tts_client, tts_limit=limit
        )
        for _ in range(2)
    ]
    voice = handlers[0]._get_voice("voice1")
    assert voice is not None

    results = await asyncio.gather(
        *(handler._get_tts_audio_stream(f"Sentence {i}.", voice) for handler in handlers for i in range(3))
    )

    assert all(result.audio for result in results)
    assert speech.peak == 2
    assert not limit.full


@pytest.mark.asyncio
@pytest.mark.parametrize("stuck_on", ["audio-start", "audio-chunk"])
async def test_tts_client_that_stops_taking_audio_fails_the_synthesis_and_frees_its_slot(
    enhanced_handler, mock_clients, monkeypatch, caplog, stuck_on
):
    """
    A client that does not take a chunk of audio in time fails the synthesis, so it cannot keep the slot, and the log
    says why.
    """
    monkeypatch.setattr("wyoming_openai.handler.CLIENT_WRITE_TIMEOUT", 0.05)
    _, tts_client = mock_clients
    _mock_speech_response(tts_client, [REALTIME_TTS_PCM] * 3)
    limit = ConcurrencyLimit(1)
    handler = enhanced_handler
    handler._tts_limit = limit
    handler._tts_extra_body = {"response_format": "pcm"}
    written = []

    async def write_event(event):
        written.append(event.type)
        if event.type == stuck_on:
            # The client stopped reading
            await asyncio.Event().wait()

    handler.write_event = AsyncMock(side_effect=write_event)

    assert (
        await asyncio.wait_for(handler.handle_event(Event(type="synthesize", data={"text": "Hello world."})), 2)
        is False
    )

    assert "error" in written
    assert not limit.full
    assert f"Client did not take its {stuck_on} event within 0.05 seconds" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("stuck_on", ["transcript-start", "transcript-chunk"])
async def test_stt_client_that_stops_reading_a_streamed_transcript_fails_the_request_and_frees_its_slot(
    mock_info, mock_clients, monkeypatch, caplog, stuck_on
):
    """
    A client that does not take a transcript chunk in time fails the request, so it cannot keep the slot, and the log
    says why.
    """
    monkeypatch.setattr("wyoming_openai.handler.CLIENT_WRITE_TIMEOUT", 0.05)
    stt_client, _ = mock_clients
    stt_client.audio.transcriptions.create = AsyncMock(return_value=_HiStream())
    limit = ConcurrencyLimit(1)
    handler = _stt_limited_handler(mock_info, stt_client, limit)
    written = []

    async def write_event(event):
        written.append(Error.from_event(event).text if event.type == "error" else event.type)
        if event.type == stuck_on:
            await asyncio.Event().wait()

    handler.write_event = AsyncMock(side_effect=write_event)
    audio_format = {"rate": 16000, "width": 2, "channels": 1}
    await handler.handle_event(Event(type="audio-start", data=audio_format))
    await handler.handle_event(Event(type="audio-chunk", data=audio_format, payload=b"\x00\x01" * 10))

    with patch("wyoming_openai.handler.AsyncStream", _HiStream):
        assert await asyncio.wait_for(handler.handle_event(Event(type="audio-stop")), timeout=2) is False

    assert written[-1] == STT_FAILED_ERROR
    assert "transcript" not in written
    assert not limit.full
    assert f"Client did not take its {stuck_on} event within 0.05 seconds" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("stuck_on", ["transcript-start", "transcript-chunk"])
async def test_realtime_client_that_stops_reading_fails_the_session_and_frees_its_slot(
    enhanced_handler, mock_info, mock_clients, monkeypatch, caplog, stuck_on
):
    """
    A client that does not take the Realtime transcript-start, or a chunk the receiver writes, in time fails the
    session, which gives its slot back, and the log says why.
    """
    monkeypatch.setattr("wyoming_openai.handler.CLIENT_WRITE_TIMEOUT", 0.05)
    stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    connection = _FakeRealtimeConnection([])
    connection.events.put_nowait(
        _FakeRealtimeServerEvent("conversation.item.input_audio_transcription.delta", delta="Hi")
    )
    stt_client.realtime.connect = Mock(return_value=_FakeRealtimeConnectionManager(connection))
    limit = ConcurrencyLimit(1)
    (handler,) = _realtime_limited_handlers(enhanced_handler, mock_info, stt_client, limit, 1)
    written = []

    async def write_event(event):
        written.append(Error.from_event(event).text if event.type == "error" else event.type)
        if event.type == stuck_on:
            await asyncio.Event().wait()

    handler.write_event = AsyncMock(side_effect=write_event)
    audio_format = {"rate": 24000, "width": 2, "channels": 1}

    await asyncio.wait_for(handler.handle_event(Event(type="audio-start", data=audio_format)), timeout=2)
    await asyncio.sleep(0.2)
    # The receiver's failure is seen with the next chunk
    await handler.handle_event(Event(type="audio-chunk", data=audio_format, payload=b"\x00\x01" * 10))

    assert written.count(STT_FAILED_ERROR) == 1
    assert not limit.full
    assert f"Client did not take its {stuck_on} event within 0.05 seconds" in caplog.text


@pytest.mark.asyncio
async def test_realtime_client_too_slow_after_audio_stop_is_not_reported_as_a_missing_transcript(
    enhanced_handler, mock_info, mock_clients, monkeypatch, caplog
):
    """
    A receiver write that times out while audio-stop waits for the transcript fails the request as the client's
    doing, not as a backend that sent no transcript within REALTIME_STT_TRANSCRIPT_TIMEOUT.
    """
    monkeypatch.setattr("wyoming_openai.handler.CLIENT_WRITE_TIMEOUT", 0.05)
    stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    # Delivered once the audio is committed, so the receiver writes it while audio-stop waits for the transcript
    delta = _FakeRealtimeServerEvent("conversation.item.input_audio_transcription.delta", delta="Hi")
    stt_client.realtime.connect = Mock(return_value=_FakeRealtimeConnectionManager(_FakeRealtimeConnection([delta])))
    limit = ConcurrencyLimit(1)
    (handler,) = _realtime_limited_handlers(enhanced_handler, mock_info, stt_client, limit, 1)
    written = []

    async def write_event(event):
        written.append(Error.from_event(event).text if event.type == "error" else event.type)
        if event.type == "transcript-chunk":
            await asyncio.Event().wait()  # The client stopped reading

    handler.write_event = AsyncMock(side_effect=write_event)
    audio_format = {"rate": 24000, "width": 2, "channels": 1}
    await handler.handle_event(Event(type="audio-start", data=audio_format))
    await handler.handle_event(Event(type="audio-chunk", data=audio_format, payload=b"\x00\x01" * 10))

    assert await asyncio.wait_for(handler.handle_event(Event(type="audio-stop")), timeout=2) is False

    assert written.count(STT_FAILED_ERROR) == 1
    assert not limit.full
    assert "Error during realtime transcription: Client did not take its transcript-chunk event" in caplog.text
    assert "No Realtime transcript received" not in caplog.text


@pytest.mark.asyncio
async def test_realtime_timeout_of_another_limit_after_audio_stop_is_not_reported_as_a_missing_transcript(
    enhanced_handler, mock_info, mock_clients, caplog
):
    """
    A TimeoutError that is not the transcript deadline's, such as a transport timeout the receiver fails with while
    audio-stop waits for the transcript, is reported as it is.
    """
    stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    delta = _FakeRealtimeServerEvent("conversation.item.input_audio_transcription.delta", delta="Hi")
    stt_client.realtime.connect = Mock(return_value=_FakeRealtimeConnectionManager(_FakeRealtimeConnection([delta])))
    handler = enhanced_handler
    written = []

    async def write_event(event):
        written.append(Error.from_event(event).text if event.type == "error" else event.type)
        if event.type == "transcript-chunk":
            raise TimeoutError("transport timed out")

    handler.write_event = AsyncMock(side_effect=write_event)
    audio_format = {"rate": 24000, "width": 2, "channels": 1}
    await handler.handle_event(Event(type="audio-start", data=audio_format))
    await handler.handle_event(Event(type="audio-chunk", data=audio_format, payload=b"\x00\x01" * 10))

    assert await asyncio.wait_for(handler.handle_event(Event(type="audio-stop")), timeout=2) is False

    assert written.count(STT_FAILED_ERROR) == 1
    assert "Error during realtime transcription: transport timed out" in caplog.text
    assert "No Realtime transcript received" not in caplog.text


@pytest.mark.asyncio
async def test_realtime_timeout_raised_while_the_transcript_deadline_cancels_is_reported_as_it_is(
    enhanced_handler, mock_info, mock_clients, monkeypatch, caplog
):
    """
    A TimeoutError that a stalled commit raises when the transcript deadline cancels it is the commit's own, and is
    reported as it is, not as a missing transcript.
    """
    monkeypatch.setattr("wyoming_openai.handler.REALTIME_STT_TRANSCRIPT_TIMEOUT", 0.05)
    stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    connection = _FakeRealtimeConnection([])

    async def commit():
        try:
            await asyncio.Event().wait()  # Stalls until the deadline cancels it
        except asyncio.CancelledError:
            raise TimeoutError("transport failed during cancellation") from None

    connection.input_audio_buffer.commit = commit
    stt_client.realtime.connect = Mock(return_value=_FakeRealtimeConnectionManager(connection))
    handler = enhanced_handler
    written = []

    async def write_event(event):
        written.append(Error.from_event(event).text if event.type == "error" else event.type)

    handler.write_event = AsyncMock(side_effect=write_event)
    audio_format = {"rate": 24000, "width": 2, "channels": 1}
    await handler.handle_event(Event(type="audio-start", data=audio_format))
    await handler.handle_event(Event(type="audio-chunk", data=audio_format, payload=b"\x00\x01" * 10))

    assert await asyncio.wait_for(handler.handle_event(Event(type="audio-stop")), timeout=2) is False

    assert written.count(STT_FAILED_ERROR) == 1
    assert "Error during realtime transcription: transport failed during cancellation" in caplog.text
    assert "No Realtime transcript received" not in caplog.text


@pytest.mark.asyncio
async def test_timeout_a_write_raises_while_the_client_write_limit_cancels_it_is_passed_on_as_it_is(
    enhanced_handler, monkeypatch
):
    """
    A TimeoutError that a write raises when the client write limit cancels it is the write's own, and is passed on as
    it is rather than as the client being too slow.
    """
    monkeypatch.setattr("wyoming_openai.handler.CLIENT_WRITE_TIMEOUT", 0.05)

    async def write_event(event):
        try:
            await asyncio.Event().wait()  # Stalls until the limit cancels it
        except asyncio.CancelledError:
            raise TimeoutError("transport failed during cancellation") from None

    enhanced_handler.write_event = AsyncMock(side_effect=write_event)

    with pytest.raises(TimeoutError, match="transport failed during cancellation"):
        await asyncio.wait_for(enhanced_handler._write_event_or_fail(Event(type="transcript-start")), timeout=2)


@pytest.mark.asyncio
async def test_realtime_session_without_audio_fails_and_frees_its_slot(
    enhanced_handler, mock_info, mock_clients, monkeypatch
):
    """A Realtime session that gets no audio within the idle limit fails, closes, and gives its slot back."""
    monkeypatch.setattr("wyoming_openai.handler.REALTIME_STT_IDLE_TIMEOUT", 0.05)
    stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    manager = _FakeRealtimeConnectionManager(_FakeRealtimeConnection([]))
    stt_client.realtime.connect = Mock(return_value=manager)
    limit = ConcurrencyLimit(1)
    (handler,) = _realtime_limited_handlers(enhanced_handler, mock_info, stt_client, limit, 1)
    written = []
    handler.write_event = AsyncMock(
        side_effect=lambda event: written.append(Error.from_event(event).text if event.type == "error" else event.type)
    )

    await handler.handle_event(REALTIME_AUDIO_START)
    assert limit.full
    # Polled rather than slept for a fixed time, so a slow scheduler only delays the test
    async with asyncio.timeout(5):
        while STT_FAILED_ERROR not in written:
            await asyncio.sleep(0.01)
    await asyncio.wait_for(handler._drain_background_tasks(), timeout=1)

    assert written == ["transcript-start", STT_FAILED_ERROR]
    assert not limit.full
    assert handler._realtime_idle_timer is None
    assert manager.exited
    assert await handler.handle_event(Event(type="audio-stop")) is False


@pytest.mark.asyncio
async def test_only_audio_restarts_the_realtime_idle_limit(enhanced_handler, mock_info, mock_clients):
    """A chunk with audio moves the idle deadline on; an empty chunk does not, so it cannot keep the session."""
    stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    stt_client.realtime.connect = _completing_realtime_connect()
    handler = enhanced_handler
    audio_format = {"rate": 24000, "width": 2, "channels": 1}

    await handler.handle_event(Event(type="audio-start", data=audio_format))
    timer = handler._realtime_idle_timer
    assert timer is not None
    started_deadline = timer.when()
    await asyncio.sleep(0.02)

    await handler.handle_event(Event(type="audio-chunk", data=audio_format, payload=b""))
    assert handler._realtime_idle_timer is timer
    await handler.handle_event(Event(type="audio-chunk", data=audio_format, payload=b"\x00\x01" * 10))
    rearmed = handler._realtime_idle_timer
    assert rearmed is not None and rearmed is not timer
    assert rearmed.when() > started_deadline

    assert await handler.handle_event(Event(type="audio-stop"))
    assert _written_transcripts(handler) == ["Hi"]
    # Stopped with the session, so no timer outlives it
    assert handler._realtime_idle_timer is None


@pytest.mark.asyncio
async def test_audio_right_after_the_idle_limit_expired_keeps_the_session(enhanced_handler, mock_info, mock_clients):
    """Audio that arrives between the expiry of the idle limit and the failure it schedules keeps the session."""
    stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    stt_client.realtime.connect = _completing_realtime_connect()
    handler = enhanced_handler
    audio_format = {"rate": 24000, "width": 2, "channels": 1}
    await handler.handle_event(Event(type="audio-start", data=audio_format))
    audio_stream = handler._audio_stream
    assert audio_stream is not None

    handler._on_realtime_idle(audio_stream)
    await handler.handle_event(Event(type="audio-chunk", data=audio_format, payload=b"\x00\x01" * 10))
    await _settle()

    assert not audio_stream.failed
    assert await handler.handle_event(Event(type="audio-stop"))
    assert _written_transcripts(handler) == ["Hi"]
    assert "error" not in _written_event_types(handler)


def _slow_to_stop_realtime_connection(receiver_cancelled, receiver_may_stop, *events):
    """A Realtime connection whose receiver takes until `receiver_may_stop` to stop once cancelled."""

    class _SlowToStopConnection(_FakeRealtimeConnection):
        async def __aiter__(self):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                receiver_cancelled.set()
                await receiver_may_stop.wait()
                raise
            yield

    return _SlowToStopConnection(list(events))


@pytest.mark.asyncio
async def test_idle_failure_leaves_a_session_started_while_it_closes_alone(enhanced_handler, mock_info, mock_clients):
    """
    A new audio-start while the idle failure waits for the old session's receiver opens a healthy session: the idle
    failure's error came before it, and nothing of the new session is changed afterwards.
    """
    stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    receiver_cancelled = asyncio.Event()
    receiver_may_stop = asyncio.Event()
    old_session = _FakeRealtimeConnectionManager(
        _slow_to_stop_realtime_connection(receiver_cancelled, receiver_may_stop)
    )
    new_session = _FakeRealtimeConnectionManager(_FakeRealtimeConnection([]))
    stt_client.realtime.connect = Mock(side_effect=[old_session, new_session])
    handler = enhanced_handler
    written = []
    handler.write_event = AsyncMock(
        side_effect=lambda event: written.append(Error.from_event(event).text if event.type == "error" else event.type)
    )
    await handler.handle_event(REALTIME_AUDIO_START)
    await _settle()  # The receiver waits for events
    old_stream = handler._audio_stream
    assert old_stream is not None

    handler._on_realtime_idle(old_stream)
    await asyncio.wait_for(receiver_cancelled.wait(), timeout=1)
    await handler.handle_event(REALTIME_AUDIO_START)
    receiver_may_stop.set()
    await asyncio.wait_for(handler._drain_background_tasks(), timeout=1)

    new_stream = handler._audio_stream
    assert new_stream is not None and new_stream is not old_stream
    assert not new_stream.failed
    assert handler._is_recording
    assert handler._realtime_connection is new_session.connection
    assert written == ["transcript-start", STT_FAILED_ERROR, "transcript-stop", "transcript-start"]


@pytest.mark.asyncio
async def test_idle_failure_is_answered_once_when_audio_stop_and_stop_come_while_it_closes(
    enhanced_handler, mock_info, mock_clients
):
    """
    An audio-stop and stop() while the idle failure waits for the receiver: the idle failure's error was queued
    first, and is the one answer, before the connection closes.
    """
    stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    receiver_cancelled = asyncio.Event()
    receiver_may_stop = asyncio.Event()
    stt_client.realtime.connect = Mock(
        return_value=_FakeRealtimeConnectionManager(
            _slow_to_stop_realtime_connection(receiver_cancelled, receiver_may_stop)
        )
    )
    handler = enhanced_handler
    await handler.handle_event(REALTIME_AUDIO_START)
    await _settle()  # The receiver waits for events
    io_log = _record_client_io(handler)
    audio_stream = handler._audio_stream
    assert audio_stream is not None

    handler._on_realtime_idle(audio_stream)
    await asyncio.wait_for(receiver_cancelled.wait(), timeout=1)
    assert await handler.handle_event(Event(type="audio-stop")) is False
    stopping = asyncio.create_task(handler.stop())
    await _settle()
    receiver_may_stop.set()
    await asyncio.wait_for(stopping, timeout=3)

    assert io_log.count("error") == 1
    assert io_log.index("error") < io_log.index("close")


@pytest.mark.asyncio
async def test_idle_failure_cancelled_during_its_error_write_still_closes_the_session(
    enhanced_handler, mock_info, mock_clients
):
    """An idle failure cancelled while its error is written still closes the session it detached."""
    stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    manager = _FakeRealtimeConnectionManager(_FakeRealtimeConnection([]))
    stt_client.realtime.connect = Mock(return_value=manager)
    handler = enhanced_handler
    await handler.handle_event(REALTIME_AUDIO_START)
    audio_stream = handler._audio_stream
    assert audio_stream is not None
    error_writing = asyncio.Event()

    async def write_event(event):
        if event.type == "error":
            error_writing.set()
            await asyncio.Event().wait()  # A client slow to take it

    handler.write_event = AsyncMock(side_effect=write_event)

    handler._on_realtime_idle(audio_stream)
    await asyncio.wait_for(error_writing.wait(), timeout=1)
    (idle_failure,) = [task for task in handler._background_tasks if task.get_name() == "realtime_stt_idle"]
    idle_failure.cancel()
    await asyncio.gather(idle_failure, return_exceptions=True)
    await asyncio.wait_for(handler._drain_background_tasks(), timeout=1)

    assert manager.exited


@pytest.mark.asyncio
async def test_disconnect_waits_for_the_websocket_close_the_idle_failure_starts_while_it_drains(
    enhanced_handler, mock_info, mock_clients
):
    """A drain also waits for cleanup that the tasks it waits for start meanwhile, as the idle failure's close."""
    stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    receiver_cancelled = asyncio.Event()
    receiver_may_stop = asyncio.Event()
    socket_closed = asyncio.Event()
    manager = _GatedCloseManager(
        _slow_to_stop_realtime_connection(receiver_cancelled, receiver_may_stop), socket_closed
    )
    stt_client.realtime.connect = Mock(return_value=manager)
    handler = enhanced_handler
    await handler.handle_event(REALTIME_AUDIO_START)
    await _settle()  # The receiver waits for events
    audio_stream = handler._audio_stream
    assert audio_stream is not None

    handler._on_realtime_idle(audio_stream)
    await asyncio.wait_for(receiver_cancelled.wait(), timeout=1)
    assert await handler.handle_event(Event(type="audio-stop")) is False
    disconnecting = asyncio.create_task(handler.disconnect())
    await _settle()
    receiver_may_stop.set()  # The idle failure starts the websocket close now, after the drain began
    await _settle()

    assert not disconnecting.done()
    socket_closed.set()
    await asyncio.wait_for(disconnecting, timeout=1)
    assert manager.exited
    assert not handler._background_tasks


@pytest.mark.asyncio
async def test_drain_that_cancels_the_idle_failure_ends_with_its_websocket_closed(
    enhanced_handler, mock_info, mock_clients
):
    """
    A drain whose time runs out while the idle failure waits for the receiver cancels it, and ends with the
    session's websocket closed: the close runs in the idle failure's own task, which the drain waits for after that.
    """
    stt_client = _use_realtime_stt_model(enhanced_handler, mock_info, mock_clients)
    receiver_cancelled = asyncio.Event()

    class _MultiStepCloseManager(_FakeRealtimeConnectionManager):
        """A closing handshake that takes a few turns of the event loop, as a real one does."""

        async def __aexit__(self, exc_type, exc, tb):
            for _ in range(10):
                await asyncio.sleep(0)
            await super().__aexit__(exc_type, exc, tb)

    manager = _MultiStepCloseManager(_slow_to_stop_realtime_connection(receiver_cancelled, asyncio.Event()))
    stt_client.realtime.connect = Mock(return_value=manager)
    handler = enhanced_handler
    await handler.handle_event(REALTIME_AUDIO_START)
    await _settle()  # The receiver waits for events
    audio_stream = handler._audio_stream
    assert audio_stream is not None

    handler._on_realtime_idle(audio_stream)
    await asyncio.wait_for(receiver_cancelled.wait(), timeout=1)
    # As stop()'s drain: the close starts only when the drain cancels the idle failure
    await asyncio.wait_for(handler._drain_background_tasks(0.01), timeout=1)

    assert manager.exited
    assert not handler._background_tasks


@pytest.mark.asyncio
async def test_idle_limit_does_not_end_a_realtime_session_waiting_for_its_transcript(
    enhanced_handler, mock_info, mock_clients, monkeypatch
):
    """After audio-stop the stream has all its audio, so the wait for the transcript is not taken for idleness."""
    monkeypatch.setattr("wyoming_openai.handler.REALTIME_STT_IDLE_TIMEOUT", 0.05)
    stt_client, _ = mock_clients
    handler = enhanced_handler
    entered, respond = _gate_backend_response(handler, stt_client, True, mock_info, mock_clients)
    audio_format = {"rate": 24000, "width": 2, "channels": 1}
    await handler.handle_event(Event(type="audio-start", data=audio_format))
    await handler.handle_event(Event(type="audio-chunk", data=audio_format, payload=b"\x00\x01" * 10))

    audio_stop = asyncio.create_task(handler.handle_event(Event(type="audio-stop")))
    await asyncio.wait_for(entered.wait(), timeout=1)
    await asyncio.sleep(0.2)
    assert "error" not in _written_event_types(handler)
    respond.set()

    assert await asyncio.wait_for(audio_stop, timeout=1)
    assert _written_transcripts(handler) == ["Hi"]


@pytest.mark.asyncio
async def test_failing_a_realtime_stream_twice_reports_it_once(enhanced_handler):
    """The idle limit and a failing chunk may both fail a stream; the client is told once."""
    handler = enhanced_handler
    audio_stream = _realtime_stream()

    await handler._fail_realtime_audio_stream(audio_stream)
    await handler._fail_realtime_audio_stream(audio_stream)

    assert _written_event_types(handler) == ["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("half_close", [False, True])
@pytest.mark.parametrize("realtime", [False, True])
async def test_server_stop_sends_the_shutdown_error_before_closing_an_stt_request(
    mock_info, mock_clients, realtime, half_close
):
    """
    Over a real connection, a request stop() cancels is answered with the shutdown error before the close, also for
    a client that has stopped sending and only reads.
    """
    stt_client, _ = mock_clients
    entered = asyncio.Event()
    if realtime:
        mock_info.asr[0].models[0].name = "gpt-realtime-whisper"
        connection = _FakeRealtimeConnection([])

        async def commit():
            entered.set()
            await asyncio.Event().wait()

        connection.input_audio_buffer.commit = commit
        stt_client.realtime.connect = Mock(return_value=_FakeRealtimeConnectionManager(connection))
    else:

        async def create(**kwargs):
            entered.set()
            await asyncio.Event().wait()

        stt_client.audio.transcriptions.create = AsyncMock(side_effect=create)

    def handler_factory(reader, writer):
        return OpenAIEventHandler(
            reader,
            writer,
            info=mock_info,
            stt_client=stt_client,
            tts_client=None,
            stt_realtime_models=["gpt-realtime-whisper"] if realtime else None,
        )

    # Port 0, so the system picks a free one when the server binds, with no window for another process to take it
    server = AsyncTcpServer("127.0.0.1", 0)
    server_task = asyncio.create_task(server.run(handler_factory), name="wyoming-test-server")
    client_writer: asyncio.StreamWriter | None = None
    try:
        async with asyncio.timeout(2):
            while server._server is None:
                await asyncio.sleep(0)
        listener = server._server
        assert isinstance(listener, asyncio.Server)
        port = listener.sockets[0].getsockname()[1]
        client_reader, client_writer = await asyncio.open_connection("127.0.0.1", port)
        audio_format = {"rate": 24000, "width": 2, "channels": 1}
        for event in (
            Event(type="audio-start", data=audio_format),
            Event(type="audio-chunk", data=audio_format, payload=b"\x00\x01" * 10),
            Event(type="audio-stop"),
        ):
            await async_write_event(event, client_writer)
        await asyncio.wait_for(entered.wait(), timeout=2)
        if half_close:
            # Nothing more to send: the server reads end of stream as soon as it reads again
            client_writer.write_eof()

        # The SIGTERM path
        server._stop_requested = True
        received = []
        async with asyncio.timeout(3):
            await server.stop()
            while (event := await async_read_event(client_reader)) is not None:
                received.append(event)

        assert [event.type for event in received] == (["transcript-start", "error"] if realtime else ["error"])
        assert Error.from_event(received[-1]).text == SHUTDOWN_ERROR
    finally:
        if client_writer is not None:
            client_writer.close()
            with contextlib.suppress(Exception):
                await client_writer.wait_closed()
        if not server_task.done():
            server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await server_task


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "model_name", "expected"),
    [
        # A singular-language model never gets both fields, on the official API, a proxy or another server
        (OpenAIBackend.OPENAI, "gpt-realtime-whisper", {"languages": ["en", "de"]}),
        (OpenAIBackend.SPEACHES, "gpt-realtime-whisper", {"languages": ["en", "de"], "prompt": "Test prompt"}),
        (OpenAIBackend.OPENAI, "gpt-live-transcribe", {"languages": ["en", "de"], "prompt": "Test prompt"}),
    ],
)
async def test_realtime_transcription_extra_body_languages_replace_the_request_language(
    enhanced_handler, mock_info, mock_clients, backend, model_name, expected
):
    """Test an extra_body `languages` list is passed through and suppresses the request language."""
    stt_client, _ = mock_clients
    stt_client.backend = backend
    mock_info.asr[0].models[0].name = model_name
    enhanced_handler._stt_realtime_models = {model_name}
    enhanced_handler._stt_realtime_extra_body = {"languages": ["en", "de"]}

    assert await enhanced_handler.handle_event(Event(type="transcribe", data={"language": "en", "name": model_name}))

    session = enhanced_handler._get_realtime_transcription_session(_selected_realtime_stream(enhanced_handler))
    transcription = session["audio"]["input"]["transcription"]
    assert transcription == {"model": model_name, **expected}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "model_name", "expect_prompt"),
    [
        # OpenAI's Realtime sessions for this model do not take a prompt; the other model and servers do
        (OpenAIBackend.OPENAI, "gpt-realtime-whisper", False),
        (OpenAIBackend.OPENAI, "gpt-live-transcribe", True),
        (OpenAIBackend.SPEACHES, "gpt-realtime-whisper", True),
    ],
)
async def test_realtime_transcription_session_leaves_out_an_unsupported_prompt(
    enhanced_handler, mock_info, mock_clients, backend, model_name, expect_prompt
):
    """Test STT_PROMPT is only sent to Realtime models that support one."""
    stt_client, _ = mock_clients
    stt_client.backend = backend
    mock_info.asr[0].models[0].name = model_name
    enhanced_handler._stt_realtime_models = {model_name}

    assert await enhanced_handler.handle_event(Event(type="transcribe", data={"name": model_name}))

    session = enhanced_handler._get_realtime_transcription_session(_selected_realtime_stream(enhanced_handler))
    transcription = session["audio"]["input"]["transcription"]
    assert ("prompt" in transcription) is expect_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", [OpenAIBackend.OPENAI, OpenAIBackend.SPEACHES])
async def test_transcribe_extra_body_languages_replace_the_request_language(
    enhanced_handler, mock_info, mock_clients, backend
):
    """Test the HTTP path sends an extra_body `languages` list instead of `language` for any model."""
    stt_client, _ = mock_clients
    stt_client.backend = backend
    mock_info.asr[0].models[0].name = "whisper-1"
    # A singular field in the same body gives way too, as an aliased model may reject it
    enhanced_handler._stt_extra_body = {"language": "en", "languages": ["en", "de"], "keywords": ["Wyoming"]}

    call_args = await _transcribe_over_http(enhanced_handler, stt_client, "whisper-1")

    assert call_args["language"] is omit
    assert call_args["languages"] is omit
    assert "language" not in call_args["extra_body"]
    assert call_args["extra_body"]["languages"] == ["en", "de"]
    assert call_args["extra_body"]["keywords"] == ["Wyoming"]


@pytest.mark.parametrize(
    ("selected_program", "expected_voice", "expected_streamed"),
    [
        # Without a selection, the first program: the one Home Assistant shows, here only Realtime voices
        (None, "alloy (gpt-realtime-2.1-mini)", True),
        ("openai", "alloy (gpt-4o-mini-tts)", False),
    ],
)
@pytest.mark.asyncio
async def test_nameless_voice_request_uses_the_program_a_client_shows(
    dummy_reader_writer, selected_program, expected_voice, expected_streamed
):
    """Test a synthesis without a voice gets one of the program the client sees, not one of another program."""
    realtime_model = "gpt-realtime-2.1-mini"
    voices = create_tts_voices(
        ["gpt-4o-mini-tts", realtime_model],
        [realtime_model],
        ["alloy", "marin"],
        "https://api.openai.com/v1",
        ["en"],
        openai_realtime_models=[realtime_model],
    )
    info = create_info([], create_tts_programs(voices, tts_streaming_models=[realtime_model]))
    assert [program.name for program in info.tts] == ["openai-streaming", "openai"]
    handler = OpenAIEventHandler(
        *dummy_reader_writer, info=info, stt_client=None, tts_client=MagicMock(), tts_realtime_models=[realtime_model]
    )
    handler.write_event = AsyncMock()
    if selected_program:
        assert await handler.handle_event(Event(type="select-program", data={"name": selected_program}))

    assert await handler.handle_event(SynthesizeStart().event())

    assert handler._synthesis.voice is not None
    assert handler._synthesis.voice.name == expected_voice
    assert handler._synthesis.streamed is expected_streamed


@pytest.mark.asyncio
async def test_voice_of_another_program_is_not_moved_to_the_selected_one(dummy_reader_writer, caplog):
    """Test a suffixed voice name the selected program lacks, but another offers, is refused, not remapped."""
    voices = create_tts_voices(
        ["tts-1", "gpt-4o-mini-tts"], ["gpt-4o-mini-tts"], ["alloy"], "https://api.openai.com/v1", ["en"]
    )
    info = create_info([], create_tts_programs(voices, tts_streaming_models=["gpt-4o-mini-tts"]))
    handler = OpenAIEventHandler(*dummy_reader_writer, info=info, stt_client=None, tts_client=MagicMock())
    handler.write_event = AsyncMock()
    assert await handler.handle_event(Event(type="select-program", data={"name": "openai"}))

    with caplog.at_level(logging.WARNING, logger="wyoming_openai.handler"):
        assert handler._validate_tts_voice_and_language("alloy (gpt-4o-mini-tts)", None) is None

    assert "offered by another program than the selected openai" in caplog.text
    # A name no program offers any more still falls back
    fallback = handler._validate_tts_voice_and_language("alloy (tts-1-hd)", None)
    assert fallback is not None and fallback.name == "alloy (tts-1)"


def test_nameless_voice_request_prefers_the_speech_api(enhanced_handler, mock_info):
    """Test a request that names no voice does not default onto a Realtime model listed first."""
    speech_voice, realtime_voice = _add_realtime_voice_first(mock_info, enhanced_handler)

    assert enhanced_handler._validate_tts_voice_and_language(None, None) is speech_voice

    # Realtime is still the default when it is the only transport
    mock_info.tts[0].voices.remove(speech_voice)
    assert enhanced_handler._validate_tts_voice_and_language(None, None) is realtime_voice


def _french_voice(attribution):
    """A speech API voice that only advertises French."""
    return TtsVoiceModel(
        name="french",
        model_name="tts-1",
        description="French",
        attribution=attribution,
        installed=True,
        languages=["fr"],
        version=None,
    )


def test_language_only_voice_selection_prefers_compatible_speech_voice(enhanced_handler, mock_info):
    speech_voice, realtime_voice = _add_realtime_voice_first(mock_info, enhanced_handler)
    realtime_voice.languages = ["fr"]
    french_voice = _french_voice(speech_voice.attribution)
    mock_info.tts[0].voices.append(french_voice)

    assert enhanced_handler._validate_tts_voice_and_language(None, "fr") is french_voice
    assert enhanced_handler._validate_tts_voice_and_language(None, "en") is speech_voice
    assert enhanced_handler._validate_tts_voice_and_language(None, "en-US") is speech_voice
    assert enhanced_handler._validate_tts_voice_and_language(None, "de") is None
    # A named voice is never switched just to satisfy the language, but a regional tag of its language is accepted
    assert enhanced_handler._validate_tts_voice_and_language("alloy", "fr") is None
    assert enhanced_handler._validate_tts_voice_and_language("alloy", "en_US") is speech_voice
    mock_info.tts[0].voices.remove(french_voice)
    assert enhanced_handler._validate_tts_voice_and_language(None, "fr") is realtime_voice


def test_language_only_voice_failure_names_the_language_only_when_voices_exist(enhanced_handler, mock_info, caplog):
    """Without any TTS voice the language is not what is missing."""
    with caplog.at_level(logging.ERROR, logger="wyoming_openai.handler"):
        assert enhanced_handler._validate_tts_voice_and_language(None, "de") is None
    assert "No TTS voice supports language de" in caplog.text
    assert "Available languages: ['en']" in caplog.text
    assert "WYOMING_LANGUAGES" in caplog.text

    caplog.clear()
    mock_info.tts = []
    with caplog.at_level(logging.ERROR, logger="wyoming_openai.handler"):
        assert enhanced_handler._validate_tts_voice_and_language(None, "de") is None
    assert "No TTS voices specified" in caplog.text
    assert "supports language" not in caplog.text


def test_language_only_voice_selection_stays_in_default_or_selected_program(enhanced_handler, mock_info):
    english = mock_info.tts[0]
    french_voice = _french_voice(english.voices[0].attribution)
    french = Mock(voices=[french_voice])
    mock_info.tts.append(french)
    assert enhanced_handler._validate_tts_voice_and_language(None, "fr") is None
    enhanced_handler._selected_tts_program = french
    assert enhanced_handler._validate_tts_voice_and_language(None, "fr") is french_voice
    assert enhanced_handler._validate_tts_voice_and_language(None, "en") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("named", [False, True])
async def test_serialized_tts_language_reaches_sentence_segmentation(enhanced_handler, mock_info, streaming, named):
    """Exercise the public event round trip, which dropped language for named voices before Wyoming 1.11."""
    french = _french_voice(mock_info.tts[0].voices[0].attribution)
    mock_info.tts[0].voices.append(french)
    voice = SynthesizeVoice(name="french" if named else None, language="fr", speaker="speaker")
    segmenter = Mock()
    segmenter.segment.return_value = ["Bonjour."]
    enhanced_handler._play_sentences = AsyncMock(return_value=True)
    enhanced_handler._finish_tts_audio_stream = AsyncMock(return_value=True)
    with patch("wyoming_openai.handler._get_boundary_detector", return_value=segmenter) as get_segmenter:
        if streaming:
            enhanced_handler._is_tts_voice_streaming = Mock(return_value=True)
            assert await enhanced_handler.handle_event(SynthesizeStart(voice=voice).event())
            assert enhanced_handler._synthesis.voice is french
            assert await enhanced_handler.handle_event(SynthesizeChunk(text="Bonjour.").event())
        else:
            enhanced_handler._splits_plain_synthesize = Mock(return_value=True)
            assert await enhanced_handler.handle_event(Synthesize(text="Bonjour.", voice=voice).event())
            assert enhanced_handler._play_sentences.call_args.args[1] is french
        get_segmenter.assert_called_with("fr")


@pytest.mark.asyncio
async def test_realtime_tts_aligns_audio_deltas_to_whole_samples(enhanced_handler, realtime_tts):
    """Test a delta that splits a 16-bit sample is carried into the next chunk."""
    pcm = b"\x01\x02" * 50
    realtime_tts(
        [
            _FakeRealtimeServerEvent("response.output_audio.delta", delta=base64.b64encode(pcm[:33]).decode()),
            _FakeRealtimeServerEvent("response.output_audio.delta", delta=base64.b64encode(pcm[33:]).decode()),
            _FakeRealtimeServerEvent("response.done", response=Mock(status="completed")),
        ]
    )

    assert await enhanced_handler.handle_event(
        Event(type="synthesize", data={"text": "Hello world", "voice": {"name": "alloy"}})
    )

    events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
    chunks = [event.payload for event in events if event.type == "audio-chunk"]
    assert [len(chunk) for chunk in chunks] == [32, 68]
    assert b"".join(chunks) == pcm
    # 50 frames at 24 kHz
    assert events[-1].data["timestamp"] == int(50 / 24000 * 1000)


@pytest.mark.asyncio
async def test_realtime_tts_transport_refuses_unspeakable_text(enhanced_handler, mock_info, mock_clients, realtime_tts):
    """Test the Realtime transport itself never hands punctuation-only text to the model."""
    _, tts_client = mock_clients
    realtime_tts(_realtime_tts_events())

    voice = mock_info.tts[0].voices[0]
    result = await enhanced_handler._get_tts_audio_stream("...", voice)

    assert result.audio == b""
    tts_client.realtime.connect.assert_not_called()
    # Playing it is a success that sends nothing
    audio_format = enhanced_handler._get_tts_audio_format(voice)
    assert await _play_buffered_sentence(enhanced_handler, result.audio, True, 5, audio_format, "...") == 5
    enhanced_handler.write_event.assert_not_called()


@pytest.mark.asyncio
async def test_streaming_realtime_tts_with_nothing_to_speak_sends_an_empty_audio_stream(
    enhanced_handler, mock_info, mock_clients, realtime_tts
):
    """Test a streaming synthesis of unspeakable text never reaches the model and ends as an empty stream."""
    _, tts_client = mock_clients
    mock_info.tts[0].supports_synthesize_streaming = True
    realtime_tts(_realtime_tts_events())

    assert await enhanced_handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="alloy")).event())
    assert await enhanced_handler.handle_event(SynthesizeChunk(text="... \U0001f642").event())
    assert await enhanced_handler.handle_event(Event(type="synthesize-stop"))

    tts_client.realtime.connect.assert_not_called()
    events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
    # Clients build their output from audio-start, so they still get one, in the voice's format
    assert [event.type for event in events] == ["audio-start", "audio-stop", "synthesize-stopped"]
    assert events[0].data["rate"] == 24000
    assert events[1].data["timestamp"] == 0


def _header_only_wav_at_16khz():
    """Build a WAV without samples whose header names another rate than the audio that follows it."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"")
    return wav_buffer.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize("silent_chunks", [[], [_header_only_wav_at_16khz()]])
async def test_incremental_tts_skipped_first_sentence_does_not_fix_the_audio_format(
    enhanced_handler, mock_info, mock_clients, silent_chunks
):
    """Test audio-start carries the format of the first sentence with audio, not the fallback or an empty WAV's."""
    _, tts_client = mock_clients
    mock_info.tts[0].supports_synthesize_streaming = True
    assert await enhanced_handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="alloy")).event())
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(b"\x00\x01" * 10)

    _mock_speech_response(
        tts_client,
        lambda text: [wav_buffer.getvalue()] if any(char.isalnum() for char in text) else silent_chunks,
    )

    assert await enhanced_handler._process_ready_sentences(["..."])
    enhanced_handler.write_event.assert_not_called()
    assert enhanced_handler._synthesis.audio_started is False

    assert await enhanced_handler._process_ready_sentences(["Hello there.", "Bye now."])
    events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
    assert [event.type for event in events] == ["audio-start", "audio-chunk", "audio-chunk"]
    assert {event.data["rate"] for event in events} == {22050}


@pytest.mark.asyncio
async def test_buffered_truncated_wav_without_samples_fails(enhanced_handler):
    """Test the buffered path fails a WAV that declares PCM data and delivers none, as the direct path does."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(b"\x00\x01" * 240)
    header_only = wav_buffer.getvalue()[:-480]

    assert await _play_buffered_sentence(enhanced_handler, header_only, True, 0, WAV_AUDIO_FORMAT) is None
    enhanced_handler.write_event.assert_not_called()


@pytest.mark.asyncio
async def test_disconnect_does_not_wait_for_realtime_stt_close(enhanced_handler):
    """Test the Wyoming client is released before a Realtime transcription websocket finishes closing."""
    release_close = asyncio.Event()
    connection = _FakeRealtimeConnection([])
    close_connection = connection.close

    async def slow_close():
        await release_close.wait()
        await close_connection()

    connection.close = slow_close
    enhanced_handler._realtime_connection_manager = _FakeRealtimeConnectionManager(connection)
    enhanced_handler._realtime_connection = connection

    disconnect = asyncio.create_task(enhanced_handler.disconnect())
    await asyncio.sleep(0.01)
    enhanced_handler.writer.close.assert_called_once()
    assert not disconnect.done()

    release_close.set()
    await disconnect
    assert connection.closed is True


@pytest.mark.asyncio
async def test_realtime_tts_fails_when_connection_closes_mid_response(enhanced_handler, realtime_tts):
    """Test a websocket that closes before `response.done` fails the synthesis instead of ending the audio."""
    _, manager = realtime_tts([*_realtime_tts_events()[:1], None])

    result = await enhanced_handler.handle_event(
        Event(type="synthesize", data={"text": "Hello world", "voice": {"name": "alloy"}})
    )

    assert result is False
    event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
    assert event_types == ["audio-start", "audio-chunk", "error"]
    await enhanced_handler._drain_background_tasks()
    assert manager.exited is True


@pytest.mark.asyncio
async def test_http_tts_zero_size_wav_does_not_play_trailing_riff_chunks(enhanced_handler, mock_info, mock_clients):
    """Test metadata after an empty data chunk is not mistaken for audio of unknown length."""
    _, tts_client = mock_clients
    voice = mock_info.tts[0].voices[0]
    wav = _empty_wav_with_trailer(b"LIST" + struct.pack("<I", 12) + b"INFOISFT" + struct.pack("<I", 0))
    header_size = len(_header_only_wav())

    # The trailer may arrive with the header or split over later chunks
    for chunks in ([wav], [wav[:header_size], wav[header_size : header_size + 6], wav[header_size + 6 :]]):
        enhanced_handler.write_event.reset_mock()
        _mock_speech_response(tts_client, chunks)
        assert await _stream_whole_request(enhanced_handler, voice, "...", send_audio_start=True) == 0
        assert [call.args[0].type for call in enhanced_handler.write_event.call_args_list] == ["audio-start"]

    # A buffered sentence without samples leaves the stream for the next sentence to open
    enhanced_handler.write_event.reset_mock()
    enhanced_handler._synthesis.audio_started = False  # The direct calls above opened it and were never stopped
    assert await _play_buffered_sentence(enhanced_handler, wav, True, 0, WAV_AUDIO_FORMAT) == 0
    enhanced_handler.write_event.assert_not_called()
    assert enhanced_handler._synthesis.audio_started is False


def test_suffixed_voice_name_with_duplicate_counter_resolves(enhanced_handler, mock_info):
    """Test the " [n]" counter of a duplicated voice does not stop a legacy name from resolving."""
    voice = enhanced_handler._validate_tts_voice_and_language("alloy (gpt-4o-mini-tts) [1]", None)

    assert voice is mock_info.tts[0].voices[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pcm",
    [
        # Samples that spell a chunk id, with a "size" too large to be metadata
        b"LIST" + b"\xff\x7f" * 20,
        # A plausible size whose chunk never completes
        b"LIST" + struct.pack("<I", 64) + b"\x00\x01" * 8,
        # A whole chunk-shaped prefix followed by more samples
        b"LIST" + struct.pack("<I", 4) + b"\x00\x01" * 10,
    ],
)
async def test_http_tts_zero_size_wav_plays_pcm_that_starts_with_a_chunk_id(
    enhanced_handler, mock_info, mock_clients, pcm
):
    """Test audio of unknown length is not discarded because its first samples look like a RIFF chunk id."""
    _, tts_client = mock_clients
    voice = mock_info.tts[0].voices[0]

    _mock_speech_response(tts_client, [_header_only_wav(), pcm[:6], pcm[6:]])
    assert await _stream_whole_request(enhanced_handler, voice, "Hello", send_audio_start=True) is not None
    events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
    assert b"".join(event.payload for event in events if event.type == "audio-chunk") == pcm

    enhanced_handler.write_event.reset_mock()
    assert await _play_buffered_sentence(enhanced_handler, _header_only_wav() + pcm, True, 0, WAV_AUDIO_FORMAT)
    assert enhanced_handler.write_event.call_args.args[0].payload == pcm


@pytest.mark.asyncio
async def test_streaming_synthesis_resolves_its_voice_once(enhanced_handler, mock_info, mock_clients, caplog):
    """Test a legacy voice name is resolved, and warned about, once per synthesis rather than per sentence."""
    _, tts_client = mock_clients
    mock_info.tts[0].supports_synthesize_streaming = True
    enhanced_handler._tts_extra_body = {"response_format": "pcm"}
    _mock_speech_response(tts_client, [b"\x00\x01" * 10])

    with caplog.at_level(logging.WARNING, logger="wyoming_openai.handler"):
        voice = SynthesizeVoice(name="alloy (gpt-4o-mini-tts)")
        assert await enhanced_handler.handle_event(SynthesizeStart(voice=voice).event())
        assert await enhanced_handler.handle_event(SynthesizeChunk(text="First sentence. Second one. ").event())
        assert await enhanced_handler.handle_event(SynthesizeChunk(text="Third one. Tail").event())
        assert await enhanced_handler.handle_event(Event(type="synthesize-stop"))

    assert caplog.text.count("no longer advertised") == 1
    assert enhanced_handler._synthesis.voice is None
    event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
    assert event_types[0] == "audio-start"
    assert event_types[-2:] == ["audio-stop", "synthesize-stopped"]


@pytest.mark.asyncio
async def test_http_tts_empty_headerless_audio_for_unspeakable_text_is_skipped(
    enhanced_handler, mock_info, mock_clients
):
    """Test PCM backends that return nothing for punctuation do not fail the synthesis."""
    _, tts_client = mock_clients
    voice = mock_info.tts[0].voices[0]
    enhanced_handler._tts_extra_body = {"response_format": "pcm"}

    _mock_speech_response(tts_client, [])
    assert await enhanced_handler.handle_event(
        Event(type="synthesize", data={"text": "...", "voice": {"name": "alloy"}})
    )
    event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
    assert event_types == ["audio-start", "audio-stop"]

    audio_format = enhanced_handler._get_tts_audio_format(voice)
    result = await enhanced_handler._get_tts_audio_stream("...", voice)
    assert await _play_buffered_sentence(enhanced_handler, result.audio, False, 5, audio_format, "...") == 5

    assert not enhanced_handler._synthesis.skipped_speakable_sentence

    # A sentence with something to say is skipped too, but counted so the synthesis can fail without any audio
    result = await enhanced_handler._get_tts_audio_stream("Hello", voice)
    assert await _play_buffered_sentence(enhanced_handler, result.audio, False, 5, audio_format, "Hello") == 5
    assert enhanced_handler._synthesis.skipped_speakable_sentence


@pytest.mark.asyncio
async def test_incremental_http_tts_skips_sentences_without_audio(enhanced_handler, mock_info, mock_clients):
    """Test an unspeakable sentence that yields no PCM is dropped from a batch instead of aborting it."""
    _, tts_client = mock_clients
    mock_info.tts[0].supports_synthesize_streaming = True
    enhanced_handler._tts_extra_body = {"response_format": "pcm"}
    assert await enhanced_handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="alloy")).event())
    pcm = b"\x00\x01" * 10

    _mock_speech_response(tts_client, lambda text: [pcm] if any(char.isalnum() for char in text) else [])

    for sentences in (["...", "Hello there.", "Bye now."], ["Hello there.", "...", "Bye now."]):
        enhanced_handler.write_event.reset_mock()
        enhanced_handler._synthesis.audio_started = False
        enhanced_handler._synthesis.timestamp = 0
        assert await enhanced_handler._process_ready_sentences(sentences)
        event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
        assert event_types == ["audio-start", "audio-chunk", "audio-chunk"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "trailer",
    [
        b"JUNK" + struct.pack("<I", 8) + b"\x00" * 8,
        b"PEAK" + struct.pack("<I", 16) + b"\x01" * 16,
        # An odd-sized chunk is followed by a pad byte
        b"iXML" + struct.pack("<I", 5) + b"<a/> " + b"\x00",
    ],
)
async def test_zero_size_wav_does_not_play_any_riff_chunk_in_its_container(enhanced_handler, trailer):
    """Test metadata after an empty data chunk is recognised by the RIFF size, not by a list of chunk ids."""
    wav = _empty_wav_with_trailer(trailer)
    assert await _play_buffered_sentence(enhanced_handler, wav, True, 0, WAV_AUDIO_FORMAT) == 0
    enhanced_handler.write_event.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("riff_size", "data_size"),
    [(0x7FFFFFFF, 0), (0xFFFFFFFF, 0), (0, 0), (0x7FFFFFFF, 0x7FFFFFDB), (0, 480)],
)
@pytest.mark.parametrize("chunk_size", [1, 44, 524])
async def test_wav_with_a_placeholder_riff_size_is_played(enhanced_handler, riff_size, data_size, chunk_size):
    """Test a RIFF size that stands in for an unknown length marks streamed audio, which is played."""
    pcm = b"\x00\x01" * 240
    header = _header_only_wav()
    wav = header[:4] + struct.pack("<I", riff_size) + header[8:-4] + struct.pack("<I", data_size) + pcm

    framer = WavFramer(WAV_AUDIO_FORMAT)
    framed = b"".join(framer.feed(wav[i : i + chunk_size]) for i in range(0, len(wav), chunk_size))
    assert framed + framer.finish() == pcm
    assert framer.missing_bytes == 0
    assert not framer.header_missing

    assert await _play_buffered_sentence(enhanced_handler, wav, True, 0, WAV_AUDIO_FORMAT) == 10
    assert enhanced_handler.write_event.call_args.args[0].payload == pcm


@pytest.mark.parametrize("riff_size", [0x7FFFFFFF, 0xFFFFFFFF])
def test_finite_data_length_under_a_placeholder_riff_size_still_bounds_the_audio(enhanced_handler, riff_size):
    """Test a real data length is kept when only the RIFF size is a placeholder, so a trailer is not played."""
    pcm = b"\x00\x01" * 4
    header = _header_only_wav()
    wav = header[:4] + struct.pack("<I", riff_size) + header[8:-4] + struct.pack("<I", len(pcm)) + pcm
    wav += riff_chunk(b"LIST", b"INFO")

    framer = WavFramer(WAV_AUDIO_FORMAT)
    assert framer.feed(wav) + framer.finish() == pcm

    truncated = WavFramer(WAV_AUDIO_FORMAT)
    truncated.feed(wav[: len(header) + 2])
    assert truncated.missing_bytes == len(pcm) - 2


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("This sentence has enough words. Bye.", ["This sentence has enough words. Bye."]),
        ("Hi. This sentence has enough words here.", ["Hi. This sentence has enough words here."]),
        (
            "This sentence has enough words. So does this other one. Bye.",
            ["This sentence has enough words.", "So does this other one. Bye."],
        ),
    ],
)
def test_chunk_text_for_streaming_never_drops_short_chunks(enhanced_handler, text, expected):
    """Test chunks below the word minimum are merged into a neighbour instead of being left out."""
    assert enhanced_handler._chunk_text_for_streaming(text, min_words=3, max_chars=35) == expected


@pytest.mark.parametrize(
    ("requested", "spoken", "expect_warning"),
    [
        # Decomposed and composed forms of the same letters
        ("Cafe\u0301 fu\u0308r O\u0308l", "Caf\u00e9 f\u00fcr \u00d6l", False),
        ("Stra\u00dfe", "STRASSE", False),
        # Symbol-only text has nothing to compare against its spoken name
        ("+", "plus", False),
        # Numbers and symbols are spelled out in a transcript
        ("It is 72\u00b0F", "It is seventy-two degrees Fahrenheit", False),
        ("Tom & Jerry", "Tom and Jerry", False),
        ("It is \u00bd full", "It is one half full", False),
        ("Chapter \u2163", "Chapter four", False),
        ("Hello world", "Sure, I can help with that.", True),
        # Ideographs with a numeric value are part of ordinary words, not numbers a transcript spells out
        ("\u6211\u4eec\u4e00\u8d77\u53bb\u5427", "\u597d\u7684\uff0c\u6211\u6765\u5e2e\u4f60\u3002", True),
        ("\u4f60\u5728\u8bf4\u4ec0\u4e48", "\u597d\u7684\uff0c\u6211\u6765\u5e2e\u4f60\u3002", True),
    ],
)
def test_realtime_tts_fidelity_normalizes_unicode(enhanced_handler, caplog, requested, spoken, expect_warning):
    """Test the fidelity check does not warn about text that only differs in Unicode form or case folding."""
    with caplog.at_level(logging.WARNING, logger="wyoming_openai.handler"):
        enhanced_handler._check_realtime_tts_fidelity(requested, spoken)

    assert ("differs from the request" in caplog.text) is expect_warning


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_voice", [True, False])
@pytest.mark.parametrize("pcm_response", [True, False])
async def test_all_empty_speakable_sentences_fail_at_stop_and_reset(
    enhanced_handler, mock_info, mock_clients, caplog, explicit_voice, pcm_response
):
    """An all-empty response fails at finalization, with a named voice and with the default one."""
    _, tts_client = mock_clients
    mock_info.tts[0].supports_synthesize_streaming = True
    if pcm_response:
        enhanced_handler._tts_extra_body = {"response_format": "pcm"}
    _mock_speech_response(tts_client, [] if pcm_response else [_header_only_wav()])
    voice = SynthesizeVoice(name="alloy") if explicit_voice else None
    assert await enhanced_handler.handle_event(SynthesizeStart(voice=voice).event())
    assert await enhanced_handler.handle_event(SynthesizeChunk(text="First sentence. Second sentence.").event())
    enhanced_handler.write_event.assert_not_called()
    assert not await enhanced_handler.handle_event(Event(type="synthesize-stop"))
    assert [c.args[0].type for c in enhanced_handler.write_event.call_args_list] == ["error", "synthesize-stopped"]
    assert "no audio for any speakable sentence" in caplog.text
    assert tts_client.audio.speech.with_streaming_response.create.call_count == 2
    assert not enhanced_handler._synthesis.skipped_speakable_sentence
    await asyncio.sleep(0)  # Let completed tasks' tracking callbacks run.
    assert not enhanced_handler._synthesis_tasks

    # A following punctuation-only synthesis must still finish as an empty success.
    enhanced_handler.write_event.reset_mock()
    assert await enhanced_handler.handle_event(SynthesizeStart(voice=voice).event())
    assert await enhanced_handler.handle_event(SynthesizeChunk(text="...").event())
    assert await enhanced_handler.handle_event(Event(type="synthesize-stop"))
    assert [c.args[0].type for c in enhanced_handler.write_event.call_args_list] == [
        "audio-start", "audio-stop", "synthesize-stopped"
    ]


@pytest.mark.asyncio
async def test_empty_symbol_sentence_is_skipped_before_spoken_text(enhanced_handler, mock_info, mock_clients, caplog):
    """The review's === probe warns and continues with Next part."""
    _, tts_client = mock_clients
    mock_info.tts[0].supports_synthesize_streaming = True
    enhanced_handler._tts_extra_body = {"response_format": "pcm"}
    _mock_speech_response(tts_client, lambda text: [] if text == "===" else [REALTIME_TTS_PCM])
    assert await enhanced_handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="alloy")).event())
    assert await enhanced_handler._process_ready_sentences(["===", "Next part."])
    assert await enhanced_handler.handle_event(Event(type="synthesize-stop"))
    assert "Skipping sentence without audio: ===" in caplog.text
    events = [c.args[0] for c in enhanced_handler.write_event.call_args_list]
    assert [e.type for e in events] == ["audio-start", "audio-chunk", "audio-stop", "synthesize-stopped"]
    assert events[1].payload == REALTIME_TTS_PCM
    assert events[2].data["timestamp"] == 10


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_plain_realtime_tts_sentence_splitting(enhanced_handler, mock_info, realtime_tts, streaming):
    """Only streaming Realtime voices split plain requests, with a single ordered audio stream."""
    mock_info.tts[0].supports_synthesize_streaming = streaming
    sentences = ["First sentence.", "Second sentence.", "Third sentence.", "Fourth sentence."]
    requests = sentences if streaming else [" ".join(sentences)]
    scripted = []
    payloads = []
    for i, sentence in enumerate(requests):
        events = _realtime_tts_events(spoken_text=sentence)
        pcm = bytes([i, 1]) * 240
        events[0] = _FakeRealtimeServerEvent("response.output_audio.delta", delta=base64.b64encode(pcm).decode())
        payloads.append(pcm)
        scripted.append(events)
    managers = realtime_tts.each(scripted)
    assert await enhanced_handler.handle_event(
        Event(type="synthesize", data={"text": " ".join(sentences), "voice": {"name": "alloy"}})
    )
    assert len(managers) == len(requests)
    assert [m.connection.response.created[0]["input"][0]["content"][0]["text"] for m in managers] == requests
    events = [c.args[0] for c in enhanced_handler.write_event.call_args_list]
    assert [e.type for e in events] == ["audio-start"] + ["audio-chunk"] * len(requests) + ["audio-stop"]
    assert [e.payload for e in events if e.type == "audio-chunk"] == payloads
    assert events[-1].data["timestamp"] == 10 * len(requests)
    await enhanced_handler._drain_background_tasks()
    assert all(m.exited and m.connection.closed for m in managers)
    assert not enhanced_handler._synthesis_tasks
    assert not enhanced_handler._synthesis.skipped_speakable_sentence


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_index", [0, 1])
async def test_plain_realtime_tts_failed_sentence_closes_audio_only(
    enhanced_handler, mock_info, realtime_tts, failure_index
):
    """Plain-flow failures cancel outstanding requests and emit an error, never audio-stop or synthesize-stopped."""
    mock_info.tts[0].supports_synthesize_streaming = True
    scripted = [_realtime_tts_events() for _ in range(3)]
    scripted[failure_index] = [_FakeRealtimeServerEvent("error", error={"message": "boom"})]
    managers = realtime_tts.each(scripted)
    assert not await enhanced_handler.handle_event(
        Event(type="synthesize", data={"text": "First sentence. Second sentence. Third sentence."})
    )
    events = [c.args[0] for c in enhanced_handler.write_event.call_args_list]
    assert [e.type for e in events] == (["audio-start", "audio-chunk", "error"] if failure_index else ["error"])
    assert not enhanced_handler._synthesis.audio_started
    await asyncio.sleep(0)  # Let completed tasks' tracking callbacks run.
    assert not enhanced_handler._synthesis_tasks
    await enhanced_handler._drain_background_tasks()
    assert all(m.exited and m.connection.closed for m in managers if m.entered)


@pytest.mark.asyncio
async def test_plain_realtime_tts_cancellation_closes_tasks_and_audio(enhanced_handler, mock_info, realtime_tts):
    """Cancellation after one sentence plays reports the failure and closes every pending websocket."""
    mock_info.tts[0].supports_synthesize_streaming = True
    managers = realtime_tts.each([_realtime_tts_events(), [], []])
    task = asyncio.create_task(enhanced_handler.handle_event(
        Event(type="synthesize", data={"text": "First sentence. Second sentence. Third sentence."})
    ))
    try:
        async with asyncio.timeout(1):
            while len(managers) < 3 or len(enhanced_handler.write_event.call_args_list) < 2:
                await asyncio.sleep(0)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert [c.args[0].type for c in enhanced_handler.write_event.call_args_list] == [
        "audio-start", "audio-chunk", "error"
    ]
    assert not enhanced_handler._synthesis_tasks
    assert not enhanced_handler._synthesis.audio_started
    await enhanced_handler._drain_background_tasks()
    assert all(m.exited and m.connection.closed for m in managers)


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup", ["stop", "disconnect"])
@pytest.mark.parametrize("incremental_input", [False, True])
@pytest.mark.parametrize("with_audio", [False, True])
async def test_unsplit_realtime_tts_shutdown_cancels_request(
    enhanced_handler, mock_info, realtime_tts, cleanup, incremental_input, with_audio
):
    """
    Shutdown cancels unsplit synthesis and drains its websocket cleanup before returning.
    stop() reports the synthesis it cut short; a client that disconnected is told nothing. Only a streaming
    session gets an audio stop for the audio it has: on its own a synthesize request would be complete with it.
    """
    mock_info.tts[0].supports_synthesize_streaming = False
    connection, manager = realtime_tts(_realtime_tts_events()[:1] if with_audio else [])
    connection.close = AsyncMock(wraps=connection.close)
    if incremental_input:
        assert await enhanced_handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="alloy")).event())
        assert await enhanced_handler.handle_event(SynthesizeChunk(text="Hello world").event())
        event = Event(type="synthesize-stop")
    else:
        event = Event(type="synthesize", data={"text": "Hello world"})

    # Recorded rather than asserted here: the handler swallows exceptions raised while writing events
    writer_closed_at_write = []

    async def write_event(event):
        writer_closed_at_write.append(enhanced_handler.writer.close.called)

    enhanced_handler.write_event.side_effect = write_event
    task = asyncio.create_task(enhanced_handler.handle_event(event))
    try:
        async with asyncio.timeout(1):
            while not connection.response.created or (
                with_audio and len(enhanced_handler.write_event.call_args_list) < 2
            ):
                await asyncio.sleep(0)
            await getattr(enhanced_handler, cleanup)()

        assert task.done()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert manager.exited and connection.closed
        connection.close.assert_awaited_once()
        assert not enhanced_handler._synthesis_tasks
        assert not enhanced_handler._background_tasks
        events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
        expected_event_types = ["audio-start", "audio-chunk"] if with_audio else []
        if cleanup == "stop":
            expected_event_types.append("error")
        if with_audio and incremental_input:
            expected_event_types.append("audio-stop")
        assert [event.type for event in events] == expected_event_types
        assert not any(writer_closed_at_write)
        if with_audio and incremental_input:
            assert events[-1].data["timestamp"] == 10
        assert not enhanced_handler._synthesis.audio_started
    finally:
        # Failure-only cleanup; successful shutdown must have already cancelled the request.
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_server_stop_cancels_unsplit_realtime_tts(enhanced_handler, mock_info, realtime_tts):
    """Exercise the real server/handler lifecycle without relying on event-loop shutdown for cancellation."""
    mock_info.tts[0].supports_synthesize_streaming = False
    connection, manager = realtime_tts([])
    connection.close = AsyncMock(wraps=connection.close)
    enhanced_handler.reader = asyncio.StreamReader()
    server = AsyncTcpServer("127.0.0.1", 0)
    await server._handler_callback(
        lambda reader, writer: enhanced_handler, enhanced_handler.reader, enhanced_handler.writer
    )
    tasks = list(server._handlers)
    enhanced_handler.reader.feed_data(
        json.dumps({"type": "synthesize", "data": {"text": "Hello world"}}).encode() + b"\n"
    )
    try:
        async with asyncio.timeout(1):
            while not connection.response.created:
                await asyncio.sleep(0)
            await server.stop()
            await asyncio.gather(*tasks, return_exceptions=True)
        assert all(task.cancelled() for task in tasks)
        assert not server._handlers
        assert manager.exited and connection.closed
        connection.close.assert_awaited_once()
        assert not enhanced_handler._synthesis_tasks
        assert not enhanced_handler._background_tasks
        # The client is told why it gets no audio before the connection closes
        events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
        assert [(event.type, event.data["text"]) for event in events] == [("error", "Server is shutting down")]
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="Windows event loops treat a full socket differently")
async def test_a_connection_the_handler_ends_on_a_client_that_stopped_reading_is_aborted(mock_info, monkeypatch):
    """
    A connection the handler ends while its client is not reading, as after a request failed for that, is aborted
    within SHUTDOWN_CLOSE_TIMEOUT, so it neither stays open on the data left in it nor holds up the server's
    shutdown, which waits for every connection although no handler is left to stop.
    """
    monkeypatch.setattr("wyoming_openai.handler.SHUTDOWN_CLOSE_TIMEOUT", 0.05)
    monkeypatch.setattr("wyoming_openai.handler.TTS_STREAM_STOP_TIMEOUT", 0.05)
    handlers: list[OpenAIEventHandler] = []

    def handler_factory(reader, writer):
        handler = OpenAIEventHandler(reader, writer, info=mock_info, stt_client=None, tts_client=None)
        handlers.append(handler)
        return handler

    server = AsyncTcpServer("127.0.0.1", 0)
    server_task = asyncio.create_task(server.run(handler_factory), name="wyoming-test-server")
    flood_task: asyncio.Task[None] | None = None
    client_writer: asyncio.StreamWriter | None = None
    client_socket: socket.socket | None = None
    try:
        async with asyncio.timeout(2):
            while server._server is None:
                await asyncio.sleep(0)
        listener = server._server
        assert isinstance(listener, asyncio.Server)
        port = listener.sockets[0].getsockname()[1]

        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        client_socket.setblocking(False)
        await asyncio.get_running_loop().sock_connect(client_socket, ("127.0.0.1", port))
        _, client_writer = await asyncio.open_connection(sock=client_socket, limit=1024)

        async with asyncio.timeout(2):
            while not handlers:
                await asyncio.sleep(0)
        handler = handlers[0]
        payload = b"\x00\x01" * (128 * 1024)

        async def flood() -> None:
            while True:
                await handler.write_event(AudioChunk(audio=payload, rate=24000, width=2, channels=1).event())

        # Audio the client leaves unread, as a synthesis it stopped reading leaves behind
        flood_task = asyncio.create_task(flood(), name="wyoming-test-flood")
        async with asyncio.timeout(2):
            while handler.writer.transport.get_write_buffer_size() == 0:
                await asyncio.sleep(0.01)

        # A request that fails and ends the connection, as one whose client did not take its events does
        await async_write_event(Event(type="transcribe", data={"name": "not-a-model"}), client_writer)
        async with asyncio.timeout(2):
            while server._handlers:
                await asyncio.sleep(0.01)

        # The handler has ended, so stop() has none to stop; the connection is closed all the same
        async with asyncio.timeout(0.5):
            await handler.writer.wait_closed()
        server._stop_requested = True
        async with asyncio.timeout(2):
            await server.stop()
            await asyncio.wait_for(asyncio.shield(server_task), 0.5)
    finally:
        if flood_task is not None:
            flood_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await flood_task
        if client_writer is not None:
            client_writer.close()
            with contextlib.suppress(Exception):
                await client_writer.wait_closed()
        elif client_socket is not None:
            client_socket.close()
        if not server_task.done():
            server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await server_task


@pytest.mark.asyncio
async def test_a_close_that_times_out_on_its_own_is_not_taken_for_a_client_that_stopped_reading(
    enhanced_handler, caplog
):
    """A TimeoutError of the close itself leaves the connection closed: it is neither aborted nor reported as stuck."""
    enhanced_handler.writer.wait_closed = AsyncMock(side_effect=TimeoutError("TCP close failed"))

    with caplog.at_level(logging.DEBUG, logger="wyoming_openai.handler"):
        await asyncio.wait_for(enhanced_handler._close_writer(), timeout=1)

    enhanced_handler.writer.transport.abort.assert_not_called()
    assert "did not take the rest of its data" not in caplog.text
    assert "Waiting for the client connection to close failed: TCP close failed" in caplog.text


@pytest.mark.asyncio
async def test_a_close_that_fails_after_its_wait_was_cancelled_is_not_logged_as_never_retrieved(enhanced_handler):
    """The close a cancelled wait leaves running has its failure read, so asyncio does not log it as unretrieved."""
    loop = asyncio.get_running_loop()
    reported: list[dict] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: reported.append(context))
    may_fail = asyncio.Event()

    async def wait_closed():
        await may_fail.wait()
        raise ConnectionResetError("reset while closing")

    enhanced_handler.writer.wait_closed = wait_closed
    try:
        closing = asyncio.create_task(enhanced_handler._close_writer())
        await _settle()  # Waiting for the close
        closing.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await closing
        enhanced_handler.writer.transport.abort.assert_called_once()
        may_fail.set()
        await _settle()  # The close fails, with nobody waiting for it
        del closing
        gc.collect()
        await _settle()
    finally:
        loop.set_exception_handler(previous_handler)

    assert not [context for context in reported if "never retrieved" in str(context.get("message"))], reported


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="Windows event loops treat a full socket differently")
async def test_a_handler_cancelled_while_it_closes_a_stalled_connection_aborts_it(mock_info, monkeypatch):
    """
    A handler cancelled while it waits for a stalled connection to close, as at Ctrl+C, aborts it rather than leave
    it open, and leaves the close waiter usable, so the server's shutdown does not wait for the connection.
    """
    monkeypatch.setattr("wyoming_openai.handler.SHUTDOWN_CLOSE_TIMEOUT", 30.0)  # Longer than the test
    monkeypatch.setattr("wyoming_openai.handler.TTS_STREAM_STOP_TIMEOUT", 0.05)
    handlers: list[OpenAIEventHandler] = []

    def handler_factory(reader, writer):
        handler = OpenAIEventHandler(reader, writer, info=mock_info, stt_client=None, tts_client=None)
        handlers.append(handler)
        return handler

    server = AsyncTcpServer("127.0.0.1", 0)
    server_task = asyncio.create_task(server.run(handler_factory), name="wyoming-test-server")
    flood_task: asyncio.Task[None] | None = None
    client_writer: asyncio.StreamWriter | None = None
    client_socket: socket.socket | None = None
    try:
        async with asyncio.timeout(2):
            while server._server is None:
                await asyncio.sleep(0)
        listener = server._server
        assert isinstance(listener, asyncio.Server)
        port = listener.sockets[0].getsockname()[1]

        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        client_socket.setblocking(False)
        await asyncio.get_running_loop().sock_connect(client_socket, ("127.0.0.1", port))
        _, client_writer = await asyncio.open_connection(sock=client_socket, limit=1024)

        async with asyncio.timeout(2):
            while not handlers:
                await asyncio.sleep(0)
        handler = handlers[0]
        (run_task,) = server._handlers
        payload = b"\x00\x01" * (128 * 1024)

        async def flood() -> None:
            while True:
                await handler.write_event(AudioChunk(audio=payload, rate=24000, width=2, channels=1).event())

        flood_task = asyncio.create_task(flood(), name="wyoming-test-flood")
        async with asyncio.timeout(2):
            while handler.writer.transport.get_write_buffer_size() == 0:
                await asyncio.sleep(0.01)

        # The request fails and ends the connection, whose close then waits on the audio the client does not take
        await async_write_event(Event(type="transcribe", data={"name": "not-a-model"}), client_writer)
        async with asyncio.timeout(2):
            while not handler.writer.transport.is_closing():
                await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        assert not run_task.done()

        run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run_task

        # Aborted, and the close waiter still completes rather than raise its own cancellation
        async with asyncio.timeout(0.5):
            await handler.writer.wait_closed()
        server._stop_requested = True
        async with asyncio.timeout(2):
            await server.stop()
            await asyncio.wait_for(asyncio.shield(server_task), 0.5)
    finally:
        if flood_task is not None:
            flood_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await flood_task
        if client_writer is not None:
            client_writer.close()
            with contextlib.suppress(Exception):
                await client_writer.wait_closed()
        elif client_socket is not None:
            client_socket.close()
        if not server_task.done():
            server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await server_task


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="Windows event loops treat a full socket differently")
async def test_server_stop_aborts_a_client_that_stopped_reading(mock_info, monkeypatch):
    """Test a client that leaves audio unread cannot hold up shutdown of the real server."""
    monkeypatch.setattr("wyoming_openai.handler.SHUTDOWN_CLOSE_TIMEOUT", 0.05)
    handlers: list[OpenAIEventHandler] = []

    def handler_factory(reader, writer):
        handler = OpenAIEventHandler(reader, writer, info=mock_info, stt_client=None, tts_client=None)
        handlers.append(handler)
        return handler

    server = AsyncTcpServer("127.0.0.1", 0)
    server_task = asyncio.create_task(server.run(handler_factory), name="wyoming-test-server")
    flood_task: asyncio.Task[None] | None = None
    client_writer: asyncio.StreamWriter | None = None
    client_socket: socket.socket | None = None
    try:
        async with asyncio.timeout(2):
            while server._server is None:
                await asyncio.sleep(0)
        listener = server._server
        assert isinstance(listener, asyncio.Server)
        port = listener.sockets[0].getsockname()[1]

        # A small receive buffer leaves the server's audio queued as soon as the test client stops reading
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        client_socket.setblocking(False)
        await asyncio.get_running_loop().sock_connect(client_socket, ("127.0.0.1", port))
        _, client_writer = await asyncio.open_connection(sock=client_socket, limit=1024)

        async with asyncio.timeout(2):
            while not handlers:
                await asyncio.sleep(0)
        handler = handlers[0]

        payload = b"\x00\x01" * (128 * 1024)

        async def flood() -> None:
            while True:
                await handler.write_event(AudioChunk(audio=payload, rate=24000, width=2, channels=1).event())

        flood_task = asyncio.create_task(flood(), name="wyoming-test-flood")
        async with asyncio.timeout(2):
            while handler.writer.transport.get_write_buffer_size() == 0:
                await asyncio.sleep(0.01)
        assert not flood_task.done()

        # The SIGTERM path: stop the handlers and the listener, then wait for run() to return
        server._stop_requested = True
        async with asyncio.timeout(2):
            await server.stop()
            # The stalled transport is aborted after SHUTDOWN_CLOSE_TIMEOUT, so run() returns well within this
            # bound instead of waiting for the client to take the rest of the audio
            await asyncio.wait_for(asyncio.shield(server_task), 0.5)
    finally:
        if flood_task is not None:
            flood_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await flood_task
        if client_writer is not None:
            client_writer.close()
            with contextlib.suppress(Exception):
                await client_writer.wait_closed()
        elif client_socket is not None:
            client_socket.close()
        if not server_task.done():
            server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await server_task


@pytest.mark.asyncio
@pytest.mark.parametrize("incremental_input", [False, True])
async def test_unsplit_http_tts_shutdown_cancels_request(enhanced_handler, mock_info, mock_clients, incremental_input):
    """The shared unsplit path also cancels HTTP responses and reports the synthesis that was cut short."""
    _, tts_client = mock_clients
    mock_info.tts[0].supports_synthesize_streaming = False
    enhanced_handler._tts_extra_body = {"response_format": "pcm"}
    stalled = asyncio.Event()
    closed = asyncio.Event()

    async def respond(text):
        try:
            yield REALTIME_TTS_PCM
            stalled.set()
            await asyncio.Event().wait()
        finally:
            closed.set()

    _mock_speech_response(tts_client, respond)
    if incremental_input:
        assert await enhanced_handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="alloy")).event())
        assert await enhanced_handler.handle_event(SynthesizeChunk(text="Hello world").event())
        event = Event(type="synthesize-stop")
    else:
        event = Event(type="synthesize", data={"text": "Hello world"})

    task = asyncio.create_task(enhanced_handler.handle_event(event))
    try:
        async with asyncio.timeout(1):
            await stalled.wait()
            await enhanced_handler.stop()
        assert task.done()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set()
        assert not enhanced_handler._synthesis_tasks
        assert [call.args[0].type for call in enhanced_handler.write_event.call_args_list] == [
            "audio-start", "audio-chunk", "error", *(["audio-stop"] if incremental_input else [])
        ]
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_incremental_realtime_tts_uses_one_websocket_per_sentence(enhanced_handler, mock_info, realtime_tts):
    """Test concurrent Realtime sentences play in order over their own websockets, which are all closed."""
    mock_info.tts[0].supports_synthesize_streaming = True
    sentences = ["One sentence.", "Two sentences.", "Three sentences."]
    managers = realtime_tts.each([_realtime_tts_events(spoken_text=sentence) for sentence in sentences])
    assert await enhanced_handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="alloy")).event())

    assert await enhanced_handler._process_ready_sentences(sentences)

    event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
    assert event_types == ["audio-start", "audio-chunk", "audio-chunk", "audio-chunk"]
    assert len(managers) == 3
    spoken = [manager.connection.response.created[0]["input"][0]["content"][0]["text"] for manager in managers]
    assert spoken == sentences
    await enhanced_handler._drain_background_tasks()
    assert all(manager.exited and manager.connection.closed for manager in managers)


@pytest.mark.asyncio
async def test_incremental_realtime_tts_failed_sentence_aborts_the_others(enhanced_handler, mock_info, realtime_tts):
    """Test a sentence failing on its own websocket stops the audio once and leaves no task or socket behind."""
    mock_info.tts[0].supports_synthesize_streaming = True
    managers = realtime_tts.each(
        [
            _realtime_tts_events(spoken_text="One sentence."),
            [_FakeRealtimeServerEvent("error", error={"message": "boom"})],
            _realtime_tts_events(spoken_text="Three sentences."),
        ]
    )
    assert await enhanced_handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="alloy")).event())

    assert not await enhanced_handler._process_ready_sentences(["One sentence.", "Two sentences.", "Three sentences."])

    event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
    assert event_types == ["audio-start", "audio-chunk", "error", "audio-stop", "synthesize-stopped"]
    await asyncio.sleep(0)
    assert not enhanced_handler._synthesis_tasks
    await enhanced_handler._drain_background_tasks()
    assert all(manager.exited for manager in managers if manager.entered)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tts_extra_body", "chunks"),
    [
        # A PCM backend with nothing to say sends no bytes; a WAV one a header without samples
        ({"response_format": "pcm"}, []),
        (None, [_header_only_wav()]),
    ],
)
async def test_streaming_http_tts_with_nothing_to_speak_is_synthesized_once(
    enhanced_handler, mock_info, mock_clients, tts_extra_body, chunks
):
    """Test sentences that produced no audio are not sent to the backend again at synthesize-stop."""
    _, tts_client = mock_clients
    mock_info.tts[0].supports_synthesize_streaming = True
    enhanced_handler._tts_extra_body = tts_extra_body
    _mock_speech_response(tts_client, chunks)

    assert await enhanced_handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="alloy")).event())
    assert await enhanced_handler.handle_event(SynthesizeChunk(text="... !!!").event())
    assert await enhanced_handler.handle_event(Event(type="synthesize-stop"))

    assert tts_client.audio.speech.with_streaming_response.create.call_count == 1
    event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
    assert event_types == ["audio-start", "audio-stop", "synthesize-stopped"]
    assert enhanced_handler._synthesis.audio_started is False


@pytest.mark.asyncio
@pytest.mark.parametrize("payload_size", [8192, 60000, 70000])
async def test_zero_size_wav_does_not_play_a_large_trailing_riff_chunk(
    enhanced_handler, mock_info, mock_clients, payload_size
):
    """Test metadata far larger than a stream chunk is not played, whether it arrives whole or split up."""
    _, tts_client = mock_clients
    voice = mock_info.tts[0].voices[0]
    wav = _empty_wav_with_trailer(riff_chunk(b"JUNK", b"\x01" * payload_size))

    assert await _play_buffered_sentence(enhanced_handler, wav, True, 0, WAV_AUDIO_FORMAT) == 0
    enhanced_handler.write_event.assert_not_called()

    # The speech response is read in pieces of TTS_CHUNK_SIZE, so the chunk is never seen whole at once
    _mock_speech_response(tts_client, [wav[offset : offset + 2048] for offset in range(0, len(wav), 2048)])
    assert await _stream_whole_request(enhanced_handler, voice, "...", send_audio_start=True) == 0
    assert [call.args[0].type for call in enhanced_handler.write_event.call_args_list] == ["audio-start"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wav",
    [
        _empty_wav_with_trailer(riff_chunk(b"LIST", b"\x01" * 40000) + riff_chunk(b"C2PA", b"\x02" * 40000)),
        # A response cut off inside a chunk the RIFF size covers
        _empty_wav_with_trailer(riff_chunk(b"LIST", b"\x01" * 12) + riff_chunk(b"C2PA", b"\x02" * 100))[:-90],
        # A finite RIFF size is taken at its word however much metadata it covers
        _empty_wav_with_trailer(riff_chunk(b"JUNK", b"\x01" * 64000) * 17),
    ],
    # Named, so the payload does not become the test id
    ids=["several_chunks", "cut_off_inside_a_chunk", "over_a_mebibyte"],
)
async def test_zero_size_wav_does_not_play_a_trailer_of_several_riff_chunks(
    enhanced_handler, mock_info, mock_clients, wav
):
    """Test everything the RIFF size covers after an empty data chunk is metadata, even when it is cut off."""
    _, tts_client = mock_clients
    voice = mock_info.tts[0].voices[0]

    assert await _play_buffered_sentence(enhanced_handler, wav, True, 0, WAV_AUDIO_FORMAT) == 0
    enhanced_handler.write_event.assert_not_called()

    _mock_speech_response(tts_client, [wav[offset : offset + 2048] for offset in range(0, len(wav), 2048)])
    assert await _stream_whole_request(enhanced_handler, voice, "...", send_audio_start=True) == 0
    assert [call.args[0].type for call in enhanced_handler.write_event.call_args_list] == ["audio-start"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("batches", "played_event_types", "expected_stop_timestamp"),
    [
        # The cancelled sentence streams directly, into a stream the batch before it opened; the 10 ms it
        # sent before it was cancelled count towards the stop
        ([["Hello there."], ["Stalls here."]], ["audio-start", "audio-chunk", "audio-chunk"], 20),
        # The cancelled sentence is still being buffered behind the one that opened the stream
        ([["Hello there.", "Stalls here."]], ["audio-start", "audio-chunk"], 10),
    ],
)
async def test_stop_closes_the_audio_an_earlier_sentence_opened(
    enhanced_handler, mock_info, mock_clients, batches, played_event_types, expected_stop_timestamp
):
    """Test a sentence cancelled by stop() is reported and the stream an earlier sentence opened gets its audio-stop."""
    _, tts_client = mock_clients
    mock_info.tts[0].supports_synthesize_streaming = True
    assert await enhanced_handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="alloy")).event())
    enhanced_handler._tts_extra_body = {"response_format": "pcm"}
    stalled = asyncio.Event()

    async def respond(text):
        yield b"\x00\x01" * 240
        if text.startswith("Stalls"):
            stalled.set()
            await asyncio.Event().wait()

    _mock_speech_response(tts_client, respond)

    async def synthesize():
        for sentences in batches:
            await enhanced_handler._process_ready_sentences(sentences)

    task = asyncio.create_task(synthesize())
    async with asyncio.timeout(1):
        await stalled.wait()
        # Everything before the stalled sentence has been played
        while len(enhanced_handler.write_event.call_args_list) < len(played_event_types):
            await asyncio.sleep(0)

    with patch("wyoming.server.AsyncEventHandler.stop", new=AsyncMock()):
        await enhanced_handler.stop()
    with pytest.raises(asyncio.CancelledError):
        await task

    events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
    assert [event.type for event in events] == [*played_event_types, "error", "audio-stop"]
    assert events[-1].data["timestamp"] == expected_stop_timestamp
    assert enhanced_handler._synthesis.audio_started is False


@pytest.mark.asyncio
async def test_failed_sentence_stops_the_audio_at_what_was_sent(enhanced_handler, mock_info, mock_clients):
    """Test a sentence failing midway in a stream an earlier one opened is stopped after the audio it sent."""
    _, tts_client = mock_clients
    mock_info.tts[0].supports_synthesize_streaming = True
    assert await enhanced_handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="alloy")).event())
    enhanced_handler._tts_extra_body = {"response_format": "pcm"}

    async def respond(text):
        yield b"\x00\x01" * 240
        if text.startswith("Fails"):
            raise RuntimeError("backend went away")

    _mock_speech_response(tts_client, respond)

    assert await enhanced_handler._process_ready_sentences(["Hello there."])
    assert not await enhanced_handler._process_ready_sentences(["Fails here."])

    events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
    assert [event.type for event in events] == [
        "audio-start",
        "audio-chunk",
        "audio-chunk",
        "error",
        "audio-stop",
        "synthesize-stopped",
    ]
    assert events[4].data["timestamp"] == 20


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("send_audio_start", "expected_event_types"),
    [
        (True, ["audio-start", "audio-chunk", "audio-stop"]),
        # A stream an earlier sentence opened is closed all the same
        (False, ["audio-chunk", "audio-stop"]),
    ],
)
async def test_cancelled_tts_stream_closes_the_audio_it_opened(
    enhanced_handler, mock_info, mock_clients, send_audio_start, expected_event_types
):
    """Test a sentence cancelled after its audio started still ends the stream, once, whoever opened it."""
    _, tts_client = mock_clients
    voice = mock_info.tts[0].voices[0]
    enhanced_handler._tts_extra_body = {"response_format": "pcm"}
    first_chunk_written = asyncio.Event()

    async def respond(text):
        yield b"\x00\x01" * 240
        first_chunk_written.set()
        await asyncio.Event().wait()

    _mock_speech_response(tts_client, respond)

    enhanced_handler._synthesis.audio_started = not send_audio_start
    task = enhanced_handler._create_synthesis_task(
        _stream_whole_request(enhanced_handler, voice, "Hello", send_audio_start=send_audio_start), name="sentence"
    )
    await first_chunk_written.wait()
    # Playback leaves the stream alone; whoever stops the synthesis closes it
    await enhanced_handler._stop_sentence_playback()
    await enhanced_handler._stop_sentence_playback()
    assert task.cancelled()
    assert enhanced_handler.write_event.call_args_list[-1].args[0].data["timestamp"] == 10

    event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
    assert event_types == expected_event_types


@pytest.mark.asyncio
async def test_headerless_wav_response_is_reported_once_per_synthesis(
    enhanced_handler, mock_info, mock_clients, caplog
):
    """Test raw PCM answering a WAV request warns for the first sentence that shows it, buffered or streamed."""
    _, tts_client = mock_clients
    voice = mock_info.tts[0].voices[0]
    pcm = b"\x00\x01" * 40000
    warning = "TTS response is not a WAV file; playing it as raw PCM"

    with caplog.at_level(logging.WARNING, logger="wyoming_openai.handler"):
        assert await _play_buffered_sentence(enhanced_handler, pcm, True, 0, WAV_AUDIO_FORMAT)
        assert caplog.text.count(warning) == 1
        events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
        assert b"".join(event.payload for event in events if event.type == "audio-chunk") == pcm
        enhanced_handler.write_event.reset_mock()

        _mock_speech_response(tts_client, [pcm[:2048], pcm[2048:]])
        assert await _stream_whole_request(enhanced_handler, voice, "Hello", send_audio_start=True)
        assert caplog.text.count(warning) == 1
        # Played from the first read on, not held back as if a header might still come
        events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
        assert [event.payload for event in events if event.type == "audio-chunk"][-2:] == [pcm[:2048], pcm[2048:]]

        enhanced_handler._synthesis = _SynthesisState()
        assert await _play_buffered_sentence(enhanced_handler, pcm, True, 0, WAV_AUDIO_FORMAT)
        assert caplog.text.count(warning) == 2


def test_chunk_text_for_streaming_min_words_without_max_chars(enhanced_handler):
    """A word minimum alone still splits the text, closing each chunk once it meets the minimum."""
    text = "One. Two is here. Three is a longer sentence. Four. Five five five five. Six."
    chunks = enhanced_handler._chunk_text_for_streaming(text, min_words=3, max_chars=None)
    assert chunks == ["One. Two is here.", "Three is a longer sentence.", "Four. Five five five five. Six."]


async def _run_incremental_synthesis(handler, tts_client, text_chunks):
    """Feed text through synthesize-start/chunk/stop and return the texts sent to /v1/audio/speech."""
    handler._tts_extra_body = {"response_format": "pcm"}
    _mock_speech_response(tts_client, [REALTIME_TTS_PCM])
    assert await handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="alloy")).event())
    for text in text_chunks:
        assert await handler.handle_event(SynthesizeChunk(text=text).event())
    assert await handler.handle_event(Event(type="synthesize-stop"))
    return [call.kwargs["input"] for call in tts_client.audio.speech.with_streaming_response.create.call_args_list]


@pytest.mark.asyncio
async def test_incremental_synthesis_applies_min_words(enhanced_handler, mock_info, mock_clients):
    """Sentences below the word minimum are held and merged instead of each getting a request."""
    mock_info.tts[0].supports_synthesize_streaming = True
    enhanced_handler._tts_streaming_min_words = 4
    text_chunks = ["Yes. Sure. ", "Maybe so. This is ", "the last one. Bye"]
    requests = await _run_incremental_synthesis(enhanced_handler, mock_clients[1], text_chunks)

    assert requests == ["Yes. Sure. Maybe so.", " This is the last one.", " Bye"]
    assert "".join(requests) == "".join(text_chunks)
    types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
    assert types.count("audio-start") == 1 and types.count("audio-stop") == 1 and types[-1] == "synthesize-stopped"


@pytest.mark.asyncio
async def test_incremental_synthesis_applies_max_chars(enhanced_handler, mock_info, mock_clients):
    """Sentences that arrive together are merged up to the character maximum."""
    mock_info.tts[0].supports_synthesize_streaming = True
    enhanced_handler._tts_streaming_max_chars = 12
    requests = await _run_incremental_synthesis(enhanced_handler, mock_clients[1], ["Yes. Sure. Maybe so. Okay. End"])

    assert requests == ["Yes. Sure.", " Maybe so.", " Okay.", " End"]


@pytest.mark.asyncio
async def test_incremental_synthesis_without_limits_sends_each_sentence(enhanced_handler, mock_info, mock_clients):
    """Without limits every completed sentence is its own request, as before."""
    mock_info.tts[0].supports_synthesize_streaming = True
    requests = await _run_incremental_synthesis(enhanced_handler, mock_clients[1], ["Yes. Sure. Maybe so. End"])

    assert requests == ["Yes.", " Sure.", " Maybe so.", " End"]


@pytest.mark.asyncio
async def test_abort_is_not_held_up_by_a_client_that_stopped_reading(
    enhanced_handler, mock_info, mock_clients, monkeypatch
):
    """
    An abort does not wait on a client that stopped reading: it is offered one event, and none of those after it.
    The stream an earlier sentence opened still counts as closed.
    """
    monkeypatch.setattr("wyoming_openai.handler.TTS_STREAM_STOP_TIMEOUT", 0.01)
    mock_info.tts[0].supports_synthesize_streaming = True
    enhanced_handler._tts_extra_body = {"response_format": "pcm"}

    async def respond(text):
        if "Second" in text:
            raise RuntimeError("backend died")
        yield REALTIME_TTS_PCM

    _mock_speech_response(mock_clients[1], respond)

    async def write_event(event):
        if event.type in ("error", "audio-stop", "synthesize-stopped"):
            await asyncio.Event().wait()

    enhanced_handler.write_event.side_effect = write_event
    assert await enhanced_handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="alloy")).event())
    async with asyncio.timeout(1):
        assert not await enhanced_handler.handle_event(SynthesizeChunk(text="First one. Second one. Third").event())

    types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
    assert types == ["audio-start", "audio-chunk", "error"]
    assert not enhanced_handler._synthesis.active and not enhanced_handler._synthesis.audio_started
    assert not enhanced_handler._synthesis_tasks


@pytest.mark.asyncio
async def test_plain_synthesis_leaves_no_state_behind(enhanced_handler, mock_clients):
    """Each synthesis starts from fresh state, so nothing of a failed one reaches the next."""
    enhanced_handler._tts_extra_body = {"response_format": "pcm"}

    async def respond(text):
        yield REALTIME_TTS_PCM
        if text == "Fails":
            raise RuntimeError("backend died")

    _mock_speech_response(mock_clients[1], respond)
    assert not await enhanced_handler.handle_event(Event(type="synthesize", data={"text": "Fails"}))
    assert enhanced_handler._synthesis == type(enhanced_handler._synthesis)()
    assert await enhanced_handler.handle_event(Event(type="synthesize", data={"text": "Works"}))

    events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
    assert [event.type for event in events] == [
        "audio-start", "audio-chunk", "error", "audio-start", "audio-chunk", "audio-stop"
    ]
    # The second stream starts from zero again
    assert [events[index].data["timestamp"] for index in (4, 5)] == [0, 10]


def _home_assistant_takes_plain_synthesis(events, handles_error_events):
    """
    Whether Home Assistant's `async_get_tts_audio` returns the audio of a plain synthesis, which it then caches.
    It ends on audio-stop, and a connection that closes before one is a failure. An error event is one too
    since Home Assistant 2026.8.2; older versions ignore it.
    """
    for event in events:
        if event.type == "error" and handles_error_events:
            return False
        if event.type == "audio-stop":
            return True
    return False


def _home_assistant_takes_streamed_synthesis(events):
    """
    Whether Home Assistant's `_read_tts_audio` (2026.8.2 and later) ends a streamed synthesis without an error.
    It ignores audio-stop and ends on synthesize-stopped and on a closed connection alike; only an error fails it.
    """
    for event in events:
        if event.type == "error":
            return False
        if event.type == "synthesize-stopped":
            return True
    return True


@pytest.mark.asyncio
@pytest.mark.parametrize("handles_error_events", [True, False])
@pytest.mark.parametrize("cut_short_by", ["backend", "stop"])
async def test_plain_synthesis_cut_short_is_not_taken_for_complete_audio(
    enhanced_handler, mock_clients, handles_error_events, cut_short_by
):
    """A plain synthesis that fails or is shut down midway must not end the way a complete one does."""
    enhanced_handler._tts_extra_body = {"response_format": "pcm"}
    stalled = asyncio.Event()

    async def respond(text):
        yield REALTIME_TTS_PCM
        if text == "Complete":
            return
        if cut_short_by == "backend":
            raise RuntimeError("backend died")
        stalled.set()
        await asyncio.Event().wait()

    _mock_speech_response(mock_clients[1], respond)
    assert await enhanced_handler.handle_event(Event(type="synthesize", data={"text": "Complete"}))
    events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
    assert _home_assistant_takes_plain_synthesis(events, handles_error_events)
    enhanced_handler.write_event.reset_mock()

    task = asyncio.create_task(enhanced_handler.handle_event(Event(type="synthesize", data={"text": "Cut short"})))
    async with asyncio.timeout(1):
        if cut_short_by == "stop":
            await stalled.wait()
            with patch("wyoming.server.AsyncEventHandler.stop", new=AsyncMock()):
                await enhanced_handler.stop()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            assert not await task

    events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
    assert [event.type for event in events] == ["audio-start", "audio-chunk", "error"]
    assert not _home_assistant_takes_plain_synthesis(events, handles_error_events)


@pytest.mark.asyncio
async def test_streaming_synthesis_failure_is_reported_before_synthesize_stopped(
    enhanced_handler, mock_info, mock_clients
):
    """A client stops reading a streamed synthesis at synthesize-stopped, so only an error ahead of it fails it."""
    mock_info.tts[0].supports_synthesize_streaming = True
    enhanced_handler._tts_extra_body = {"response_format": "pcm"}

    async def respond(text):
        yield REALTIME_TTS_PCM
        if "Fails" in text:
            raise RuntimeError("backend died")

    _mock_speech_response(mock_clients[1], respond)
    start = SynthesizeStart(voice=SynthesizeVoice(name="alloy")).event()
    assert await enhanced_handler.handle_event(start)
    assert await enhanced_handler.handle_event(SynthesizeChunk(text="Works. Then").event())
    assert await enhanced_handler.handle_event(Event(type="synthesize-stop"))
    events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
    assert "error" not in [event.type for event in events]
    assert _home_assistant_takes_streamed_synthesis(events)
    enhanced_handler.write_event.reset_mock()

    assert await enhanced_handler.handle_event(start)
    assert not await enhanced_handler.handle_event(SynthesizeChunk(text="Works. Fails here. Then").event())
    events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
    types = [event.type for event in events]
    assert types.count("error") == 1 and types[-2:] == ["audio-stop", "synthesize-stopped"]
    assert types.index("error") < types.index("audio-stop")
    assert not _home_assistant_takes_streamed_synthesis(events)


@pytest.mark.asyncio
async def test_synthesize_with_an_unsupported_voice_is_reported(enhanced_handler):
    """Test a plain synthesis that is refused for its voice tells the client instead of only closing."""
    assert not await enhanced_handler.handle_event(
        Event(type="synthesize", data={"text": "Hello", "voice": {"name": "nobody"}})
    )
    events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
    assert [(event.type, event.data["text"]) for event in events] == [("error", "Text-to-speech synthesis failed")]


@pytest.mark.asyncio
async def test_stop_reports_a_streaming_synthesis_that_waits_for_more_text(enhanced_handler, mock_info, mock_clients):
    """Test stop() cuts a streaming synthesis short between two text chunks too, when no request is pending."""
    mock_info.tts[0].supports_synthesize_streaming = True
    enhanced_handler._tts_extra_body = {"response_format": "pcm"}
    _mock_speech_response(mock_clients[1], [REALTIME_TTS_PCM])
    assert await enhanced_handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="alloy")).event())
    assert await enhanced_handler.handle_event(SynthesizeChunk(text="First one. Second").event())

    with patch("wyoming.server.AsyncEventHandler.stop", new=AsyncMock()):
        await enhanced_handler.stop()
        # Neither a second stop() nor the disconnect that follows it repeats anything
        await enhanced_handler.stop()
    await enhanced_handler.disconnect()

    assert _event_types(enhanced_handler) == ["audio-start", "audio-chunk", "error", "audio-stop"]


@pytest.mark.asyncio
async def test_stop_without_a_synthesis_writes_nothing(enhanced_handler):
    """Test a connection that is idle at shutdown is only closed."""
    with patch("wyoming.server.AsyncEventHandler.stop", new=AsyncMock()):
        await enhanced_handler.stop()

    enhanced_handler.write_event.assert_not_called()


@pytest.mark.asyncio
async def test_stop_does_not_repeat_the_error_of_a_rejected_audio_start(enhanced_handler):
    """A stream rejected at audio-start already sent its error, so shutdown does not send another."""
    await enhanced_handler.handle_event(
        Event(type="audio-start", data={"rate": 0, "width": 2, "channels": 1})
    )
    enhanced_handler.write_event.reset_mock()

    with patch("wyoming.server.AsyncEventHandler.stop", new=AsyncMock()):
        await enhanced_handler.stop()

    enhanced_handler.write_event.assert_not_called()


@pytest.mark.asyncio
async def test_stop_aborts_a_connection_a_client_stopped_reading(enhanced_handler, monkeypatch):
    """Test a writer that never finishes closing is aborted, so server shutdown does not wait on it forever."""
    monkeypatch.setattr("wyoming_openai.handler.SHUTDOWN_CLOSE_TIMEOUT", 0.01)
    wait_closed = asyncio.Event()

    async def close_never_finishes():
        await wait_closed.wait()

    def abort():
        wait_closed.set()

    enhanced_handler.writer.wait_closed.side_effect = close_never_finishes
    enhanced_handler.writer.transport.abort.side_effect = abort

    async with asyncio.timeout(1):
        await enhanced_handler.stop()

    enhanced_handler.writer.transport.abort.assert_called_once()


@pytest.mark.asyncio
async def test_stop_does_not_abort_a_connection_that_closes(enhanced_handler):
    """Test a writer that takes what was written while closing is left to close normally."""
    async with asyncio.timeout(1):
        await enhanced_handler.stop()

    enhanced_handler.writer.transport.abort.assert_not_called()


@pytest.mark.asyncio
async def test_stop_ends_a_buffered_sentence_the_client_is_slow_to_take(enhanced_handler, mock_info, mock_clients):
    """Test stop() during the write of a buffered sentence is not followed by the sentences behind it."""
    mock_info.tts[0].supports_synthesize_streaming = True
    enhanced_handler._tts_extra_body = {"response_format": "pcm"}
    _mock_speech_response(mock_clients[1], [REALTIME_TTS_PCM])
    assert await enhanced_handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="alloy")).event())

    written = []
    slow_write = asyncio.Event()
    client_reads_on = asyncio.Event()

    async def write_event(event):
        written.append(event.type)
        # The first sentence streams directly; the second one was buffered behind it
        if written.count("audio-chunk") == 2 and not slow_write.is_set():
            slow_write.set()
            await client_reads_on.wait()

    enhanced_handler.write_event.side_effect = write_event
    task = asyncio.create_task(enhanced_handler._process_ready_sentences(["One here.", "Two here.", "Three here."]))
    async with asyncio.timeout(1):
        await slow_write.wait()
        with patch("wyoming.server.AsyncEventHandler.stop", new=AsyncMock()):
            await enhanced_handler.stop()
        client_reads_on.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert written == ["audio-start", "audio-chunk", "audio-chunk", "error", "audio-stop"]
    assert not enhanced_handler._synthesis_tasks


@pytest.mark.asyncio
async def test_stop_does_not_wait_for_a_websocket_that_never_finishes_closing(enhanced_handler, monkeypatch):
    """Test the shutdown of the server is not held up by a Realtime closing handshake that stalls."""
    monkeypatch.setattr("wyoming_openai.handler.SHUTDOWN_DRAIN_TIMEOUT", 0.01)
    enhanced_handler._run_in_background(asyncio.Event().wait(), name="stalled_close")

    with patch("wyoming.server.AsyncEventHandler.stop", new=AsyncMock()):
        async with asyncio.timeout(1):
            await enhanced_handler.stop()

    await asyncio.sleep(0)  # Let the cancelled task's tracking callback run.
    assert not enhanced_handler._background_tasks


@pytest.mark.asyncio
async def test_incremental_synthesis_counts_a_sentence_without_spaces_as_a_word(
    enhanced_handler, mock_info, mock_clients
):
    """Test text in a language written without spaces meets the word minimum as it arrives, not only at the stop."""
    tts_client = mock_clients[1]
    mock_info.tts[0].supports_synthesize_streaming = True
    enhanced_handler._tts_streaming_min_words = 2
    enhanced_handler._tts_extra_body = {"response_format": "pcm"}
    _mock_speech_response(tts_client, [REALTIME_TTS_PCM])
    create = tts_client.audio.speech.with_streaming_response.create
    text_chunks = ["你好。", "我很好。", "今天天气不错。", "我们去公园吧。", "好的。", "再见。"]

    assert await enhanced_handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="alloy")).event())
    for text in text_chunks:
        assert await enhanced_handler.handle_event(SynthesizeChunk(text=text).event())
    spoken_as_it_arrived = ["你好。我很好。", "今天天气不错。我们去公园吧。"]
    assert [call.kwargs["input"] for call in create.call_args_list] == spoken_as_it_arrived

    assert await enhanced_handler.handle_event(Event(type="synthesize-stop"))
    requests = [call.kwargs["input"] for call in create.call_args_list]
    assert requests == [*spoken_as_it_arrived, "好的。再见。"]
    # The same chunks as for the text split as a whole, which joins the sentences with a space
    whole = enhanced_handler._chunk_text_for_streaming("".join(text_chunks), min_words=2)
    assert [chunk.replace(" ", "") for chunk in whole] == requests


def _event_types(handler):
    return [call.args[0].type for call in handler.write_event.call_args_list]


@pytest.mark.asyncio
async def test_streaming_synthesis_without_a_voice_speaks_sentences_as_they_arrive(
    enhanced_handler, mock_info, mock_clients
):
    """Test a synthesis that names no voice streams with the default one instead of waiting for synthesize-stop."""
    _, tts_client = mock_clients
    mock_info.tts[0].supports_synthesize_streaming = True
    enhanced_handler._tts_extra_body = {"response_format": "pcm"}
    _mock_speech_response(tts_client, [REALTIME_TTS_PCM])
    create = tts_client.audio.speech.with_streaming_response.create

    assert await enhanced_handler.handle_event(SynthesizeStart().event())
    assert await enhanced_handler.handle_event(SynthesizeChunk(text="First sentence. Second").event())

    assert [call.kwargs["input"].strip() for call in create.call_args_list] == ["First sentence."]
    assert create.call_args.kwargs["voice"] == "alloy"
    assert _event_types(enhanced_handler) == ["audio-start", "audio-chunk"]

    assert await enhanced_handler.handle_event(Event(type="synthesize-stop"))
    assert [call.kwargs["input"].strip() for call in create.call_args_list] == ["First sentence.", "Second"]
    assert _event_types(enhanced_handler)[2:] == ["audio-chunk", "audio-stop", "synthesize-stopped"]


@pytest.mark.asyncio
async def test_synthesize_start_without_any_voice_to_default_to_is_rejected(enhanced_handler, mock_info):
    """Test a synthesis that names no voice is refused at the start when there is no voice to give it."""
    mock_info.tts = []

    assert not await enhanced_handler.handle_event(SynthesizeStart().event())
    assert not enhanced_handler._synthesis.active
    # Reported, so a client that reads until the connection closes does not take it for an empty synthesis
    assert _event_types(enhanced_handler) == ["error"]


@pytest.mark.asyncio
async def test_synthesize_start_closes_the_stream_of_an_unfinished_synthesis(
    enhanced_handler, mock_info, mock_clients, caplog
):
    """Test a synthesis the client never stopped does not leave its audio stream open under the next one."""
    _, tts_client = mock_clients
    mock_info.tts[0].supports_synthesize_streaming = True
    enhanced_handler._tts_extra_body = {"response_format": "pcm"}
    _mock_speech_response(tts_client, [REALTIME_TTS_PCM])
    start = SynthesizeStart(voice=SynthesizeVoice(name="alloy")).event()

    assert await enhanced_handler.handle_event(start)
    assert await enhanced_handler.handle_event(SynthesizeChunk(text="One. Two").event())
    with caplog.at_level(logging.WARNING, logger="wyoming_openai.handler"):
        assert await enhanced_handler.handle_event(start)
    assert "during an active synthesis" in caplog.text
    assert await enhanced_handler.handle_event(SynthesizeChunk(text="Three. Four").event())
    assert await enhanced_handler.handle_event(Event(type="synthesize-stop"))

    first = ["audio-start", "audio-chunk", "audio-stop"]
    second = ["audio-start", "audio-chunk", "audio-chunk", "audio-stop", "synthesize-stopped"]
    assert _event_types(enhanced_handler) == first + second
    # The text the first synthesis still held is dropped with it
    create = tts_client.audio.speech.with_streaming_response.create
    assert [call.kwargs["input"].strip() for call in create.call_args_list] == ["One.", "Three.", "Four"]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize("text", ["", "   "])
async def test_synthesis_without_text_ends_as_an_empty_stream(
    enhanced_handler, mock_info, mock_clients, streaming, text
):
    """Test no text gives the same empty audio stream as text with nothing to say, without a backend request."""
    _, tts_client = mock_clients
    mock_info.tts[0].supports_synthesize_streaming = streaming
    tts_client.audio.speech.with_streaming_response.create = Mock()

    assert await enhanced_handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="alloy")).event())
    assert await enhanced_handler.handle_event(SynthesizeChunk(text=text).event())
    assert await enhanced_handler.handle_event(Event(type="synthesize-stop"))
    assert _event_types(enhanced_handler) == ["audio-start", "audio-stop", "synthesize-stopped"]

    enhanced_handler.write_event.reset_mock()
    assert await enhanced_handler.handle_event(Event(type="synthesize", data={"text": text}))
    assert _event_types(enhanced_handler) == ["audio-start", "audio-stop"]
    tts_client.audio.speech.with_streaming_response.create.assert_not_called()


@pytest.mark.asyncio
async def test_sentence_that_waited_for_a_request_slot_is_streamed_once_it_is_played(
    enhanced_handler, mock_info, mock_clients
):
    """Test a sentence whose request only starts when playback reaches it is not buffered whole first."""
    _, tts_client = mock_clients
    mock_info.tts[0].supports_synthesize_streaming = True
    enhanced_handler._tts_limit = ConcurrencyLimit(1)
    enhanced_handler._tts_extra_body = {"response_format": "pcm"}
    log = []

    async def respond(text):
        for index in range(3):
            log.append(f"{text.strip()} {index}")
            yield REALTIME_TTS_PCM
            await asyncio.sleep(0)

    async def write(event):
        log.append(event.type)

    _mock_speech_response(tts_client, respond)
    enhanced_handler.write_event = AsyncMock(side_effect=write)

    assert await enhanced_handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="alloy")).event())
    assert await enhanced_handler.handle_event(SynthesizeChunk(text="One. Two. Three").event())

    second = log[log.index("Two. 0") :]
    assert second.count("audio-chunk") == 3
    # Its audio is on its way to the client before the backend has delivered all of it
    assert second.index("audio-chunk") < second.index("Two. 2")


@pytest.mark.asyncio
async def test_sentence_still_being_synthesized_is_streamed_when_playback_reaches_it(
    enhanced_handler, mock_info, mock_clients
):
    """Test what a running request buffered is flushed, and the rest streamed, once it is the sentence played."""
    _, tts_client = mock_clients
    mock_info.tts[0].supports_synthesize_streaming = True
    enhanced_handler._tts_extra_body = {"response_format": "pcm"}
    log = []

    async def respond(text):
        if text.strip() == "One.":
            yield b"\x01\x00" * 240
            return
        # The request starts while the first sentence is still the one being played, so this is buffered
        log.append(f"second request starts during {enhanced_handler._allow_streaming_task_id}")
        yield b"\x02\x00" * 240
        while enhanced_handler._allow_streaming_task_id != "sentence_1":
            await asyncio.sleep(0)
        log.append("playback reaches the second sentence")
        yield b"\x03\x00" * 240
        await asyncio.sleep(0)
        log.append("backend sends the rest")
        yield b"\x04\x00" * 240

    async def write(event):
        log.append(event.payload[:1] if event.payload else event.type)

    _mock_speech_response(tts_client, respond)
    enhanced_handler.write_event = AsyncMock(side_effect=write)

    assert await enhanced_handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="alloy")).event())
    async with asyncio.timeout(1):
        assert await enhanced_handler.handle_event(SynthesizeChunk(text="One. Two. Three").event())

    # Nothing of the second sentence is written until playback reaches it; then what was held goes out first
    assert log == [
        "audio-start",
        b"\x01",
        "second request starts during sentence_0",
        "playback reaches the second sentence",
        b"\x02",
        b"\x03",
        "backend sends the rest",
        b"\x04",
    ]


@pytest.mark.asyncio
async def test_realtime_transcription_times_out_when_no_transcript_arrives(
    enhanced_handler, mock_info, mock_clients, monkeypatch, caplog
):
    """Test a Realtime server that never completes the transcription does not hold the Wyoming client forever."""
    monkeypatch.setattr("wyoming_openai.handler.REALTIME_STT_TRANSCRIPT_TIMEOUT", 0.01)
    stt_client, _ = mock_clients
    mock_info.asr[0].models[0].name = "gpt-realtime-whisper"
    enhanced_handler._stt_realtime_models = {"gpt-realtime-whisper"}
    connection = _FakeRealtimeConnection(
        [_FakeRealtimeServerEvent("conversation.item.input_audio_transcription.delta", delta="Hel")]
    )
    manager = _FakeRealtimeConnectionManager(connection)
    stt_client.realtime.connect = Mock(return_value=manager)

    assert await enhanced_handler.handle_event(Event(type="transcribe", data={"name": "gpt-realtime-whisper"}))
    await enhanced_handler.handle_event(Event(type="audio-start", data={"rate": 24000, "width": 2, "channels": 1}))
    await enhanced_handler.handle_event(
        Event(type="audio-chunk", data={"rate": 24000, "width": 2, "channels": 1}, payload=b"\x00\x01" * 100)
    )
    with caplog.at_level(logging.ERROR, logger="wyoming_openai.handler"):
        async with asyncio.timeout(1):
            assert await enhanced_handler.handle_event(Event(type="audio-stop")) is False

    assert "No Realtime transcript received" in caplog.text
    events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
    assert [event.type for event in events] == ["transcript-start", "transcript-chunk", "error"]
    await enhanced_handler._drain_background_tasks()
    assert manager.exited is True


@pytest.mark.asyncio
async def test_realtime_transcription_extra_body_null_drops_a_field(enhanced_handler, mock_info, mock_clients):
    """Test a null in the Realtime extra body removes a field the proxy would send, such as the prompt."""
    stt_client, _ = mock_clients
    stt_client.backend = OpenAIBackend.OPENAI
    mock_info.asr[0].models[0].name = "some-promptless-model"
    enhanced_handler._stt_realtime_models = {"some-promptless-model"}
    enhanced_handler._stt_realtime_extra_body = {"prompt": None, "keywords": ["Wyoming"]}

    assert await enhanced_handler.handle_event(
        Event(type="transcribe", data={"name": "some-promptless-model", "language": "en"})
    )

    session = enhanced_handler._get_realtime_transcription_session(_selected_realtime_stream(enhanced_handler))
    transcription = session["audio"]["input"]["transcription"]
    assert transcription == {"model": "some-promptless-model", "language": "en", "keywords": ["Wyoming"]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("streamed_at_start", "expected_requests"),
    [(True, ["One.", "Two"]), (False, ["One. Two"])],
)
async def test_streaming_is_decided_once_per_synthesis(
    enhanced_handler, mock_info, mock_clients, streamed_at_start, expected_requests
):
    """Test a program change midway neither repeats sentences that were spoken nor drops ones that were not."""
    _, tts_client = mock_clients
    mock_info.tts[0].supports_synthesize_streaming = streamed_at_start
    enhanced_handler._tts_extra_body = {"response_format": "pcm"}
    _mock_speech_response(tts_client, [REALTIME_TTS_PCM])

    assert await enhanced_handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="alloy")).event())
    assert await enhanced_handler.handle_event(SynthesizeChunk(text="One. Two").event())
    mock_info.tts[0].supports_synthesize_streaming = not streamed_at_start
    assert await enhanced_handler.handle_event(Event(type="synthesize-stop"))

    create = tts_client.audio.speech.with_streaming_response.create
    assert [call.kwargs["input"].strip() for call in create.call_args_list] == expected_requests
    types = _event_types(enhanced_handler)
    assert types.count("audio-start") == 1 and types[-2:] == ["audio-stop", "synthesize-stopped"]


@pytest.mark.asyncio
async def test_no_sentences_are_detected_for_a_voice_that_is_not_streamed(
    enhanced_handler, mock_info, mock_clients
):
    """Test text for a voice that is synthesized as a whole is only collected, not segmented chunk by chunk."""
    mock_info.tts[0].supports_synthesize_streaming = False

    with patch("wyoming_openai.handler._get_boundary_detector") as get_boundary_detector:
        assert await enhanced_handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="alloy")).event())
        assert await enhanced_handler.handle_event(SynthesizeChunk(text="One. Two. ").event())
        assert await enhanced_handler.handle_event(SynthesizeChunk(text="Three.").event())

        get_boundary_detector.assert_not_called()
        assert "".join(enhanced_handler._synthesis.buffer) == "One. Two. Three."
        enhanced_handler.write_event.assert_not_called()

        # The collected text is synthesized as one request when the synthesis stops
        enhanced_handler._tts_extra_body = {"response_format": "pcm"}
        _mock_speech_response(mock_clients[1], [REALTIME_TTS_PCM])
        assert await enhanced_handler.handle_event(Event(type="synthesize-stop"))
        get_boundary_detector.assert_not_called()

    create = mock_clients[1].audio.speech.with_streaming_response.create
    assert [call.kwargs["input"] for call in create.call_args_list] == ["One. Two. Three."]
    assert _event_types(enhanced_handler) == ["audio-start", "audio-chunk", "audio-stop", "synthesize-stopped"]


def test_boundary_detector_uses_the_primary_language_subtag(caplog):
    """Test a language tag is matched by its primary subtag, and an unsupported language is reported once."""
    from wyoming_openai import handler as handler_module

    caches = (handler_module._get_segmenter_language, handler_module._build_boundary_detector)
    for cache in caches:
        cache.cache_clear()
    try:
        with (
            patch("wyoming_openai.handler.BoundaryDetector") as boundary_detector,
            caplog.at_level(logging.WARNING, logger="wyoming_openai.handler"),
        ):
            detectors = {handler_module._get_boundary_detector(tag) for tag in ("en", "en-US", "EN_us", None)}
            # One shared segmenter, built once
            assert len(detectors) == 1
            boundary_detector.assert_called_once_with(lang="en")
            assert "not supported" not in caplog.text

            # 'fil' is not Finnish, and a language without a segmenter falls back to English with one warning
            for tag in ("fil", "fil-PH", "fil"):
                assert handler_module._get_boundary_detector(tag) in detectors
            boundary_detector.assert_called_once_with(lang="en")
        assert caplog.text.count("not supported by yasbd") == 1
    finally:
        for cache in caches:
            cache.cache_clear()


@pytest.mark.asyncio
async def test_realtime_transcription_failure_sends_no_delta_after_the_error(
    enhanced_handler, mock_info, mock_clients, monkeypatch
):
    """Test a delta that arrives while a failed transcription is being reported is not forwarded after it."""
    monkeypatch.setattr("wyoming_openai.handler.REALTIME_STT_TRANSCRIPT_TIMEOUT", 0.01)
    stt_client, _ = mock_clients
    mock_info.asr[0].models[0].name = "gpt-realtime-whisper"
    enhanced_handler._stt_realtime_models = {"gpt-realtime-whisper"}
    connection = _FakeRealtimeConnection([])
    stt_client.realtime.connect = Mock(return_value=_FakeRealtimeConnectionManager(connection))
    written = []

    async def write(event):
        written.append(event.type)
        if event.type == "error":
            # The server sends a late delta while the client is taking the error
            await connection.events.put(
                _FakeRealtimeServerEvent("conversation.item.input_audio_transcription.delta", delta="late")
            )
            await asyncio.sleep(0)

    enhanced_handler.write_event = AsyncMock(side_effect=write)

    assert await enhanced_handler.handle_event(Event(type="transcribe", data={"name": "gpt-realtime-whisper"}))
    await enhanced_handler.handle_event(Event(type="audio-start", data={"rate": 24000, "width": 2, "channels": 1}))
    async with asyncio.timeout(1):
        assert await enhanced_handler.handle_event(Event(type="audio-stop")) is False

    assert written == ["transcript-start", "error"]
    await enhanced_handler._drain_background_tasks()


@pytest.mark.asyncio
async def test_realtime_transcription_times_out_when_the_commit_stalls(
    enhanced_handler, mock_info, mock_clients, monkeypatch
):
    """Test the time limit also covers a commit the websocket never takes."""
    monkeypatch.setattr("wyoming_openai.handler.REALTIME_STT_TRANSCRIPT_TIMEOUT", 0.01)
    stt_client, _ = mock_clients
    mock_info.asr[0].models[0].name = "gpt-realtime-whisper"
    enhanced_handler._stt_realtime_models = {"gpt-realtime-whisper"}
    connection = _FakeRealtimeConnection([])
    connection.input_audio_buffer.commit = AsyncMock(side_effect=asyncio.Event().wait)
    stt_client.realtime.connect = Mock(return_value=_FakeRealtimeConnectionManager(connection))

    assert await enhanced_handler.handle_event(Event(type="transcribe", data={"name": "gpt-realtime-whisper"}))
    await enhanced_handler.handle_event(Event(type="audio-start", data={"rate": 24000, "width": 2, "channels": 1}))
    async with asyncio.timeout(1):
        assert await enhanced_handler.handle_event(Event(type="audio-stop")) is False

    assert _event_types(enhanced_handler) == ["transcript-start", "error"]
    await enhanced_handler._drain_background_tasks()


@pytest.mark.asyncio
async def test_float_wav_response_is_played_as_pcm16(enhanced_handler, mock_info, mock_clients):
    """Test a backend answering with IEEE float WAV, as in issue #72, is converted rather than played as static."""
    _, tts_client = mock_clients
    voice = mock_info.tts[0].voices[0]
    wav = float_wav([0.0, 0.5, -0.5, 1.0] * 100)
    _mock_speech_response(tts_client, [wav[:30], wav[30:1001], wav[1001:]])

    assert await _stream_whole_request(enhanced_handler, voice, "Hello")

    events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
    audio_start = next(event for event in events if event.type == "audio-start")
    assert (audio_start.data["rate"], audio_start.data["width"], audio_start.data["channels"]) == (24000, 2, 1)
    pcm = b"".join(event.payload for event in events if event.type == "audio-chunk")
    assert pcm == struct.pack("<4h", 0, 16384, -16384, 32767) * 100


@pytest.mark.asyncio
async def test_unplayable_wav_response_fails_with_a_descriptive_error(
    enhanced_handler, mock_info, mock_clients, caplog
):
    """Test a WAV response in a format the proxy cannot play fails the synthesis instead of playing noise."""
    _, tts_client = mock_clients
    voice = mock_info.tts[0].voices[0]
    _mock_speech_response(tts_client, [float_wav([0.0] * 100, tag=0x55)])

    with caplog.at_level(logging.ERROR, logger="wyoming_openai.handler"):
        assert await _stream_whole_request(enhanced_handler, voice, "Hello") is None

    assert "TTS backend returned audio that cannot be played: WAV audio format 0x0055 (MPEG Layer 3)" in caplog.text
    assert "Traceback" not in caplog.text
    assert not [call for call in enhanced_handler.write_event.call_args_list if call.args[0].type == "audio-chunk"]


@pytest.mark.asyncio
async def test_wav_response_to_a_pcm_request_is_played_by_its_header(
    enhanced_handler, mock_info, mock_clients, caplog
):
    """Test a backend that ignores `response_format: pcm` and answers with float WAV still plays correctly."""
    _, tts_client = mock_clients
    voice = mock_info.tts[0].voices[0]
    enhanced_handler._tts_extra_body = {"response_format": "pcm"}
    wav = wav_file(fmt_chunk_body(3, rate=16000), struct.pack("<4f", 0.0, 0.5, -0.5, 1.0) * 100)
    _mock_speech_response(tts_client, [wav[:2], wav[2:]])

    with caplog.at_level(logging.WARNING, logger="wyoming_openai.handler"):
        assert await _stream_whole_request(enhanced_handler, voice, "Hello")

    events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
    audio_start = next(event for event in events if event.type == "audio-start")
    assert (audio_start.data["rate"], audio_start.data["width"], audio_start.data["channels"]) == (16000, 2, 1)
    pcm = b"".join(event.payload for event in events if event.type == "audio-chunk")
    assert pcm == struct.pack("<4h", 0, 16384, -16384, 32767) * 100
    assert caplog.text.count("TTS response is a WAV file although response_format is 'pcm'") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mp3",
    [
        b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\xff\xfb\x90\x64" * 100,
        # As Microsoft Edge TTS sends it for any format when it has no ffmpeg
        bytes.fromhex("fff364c4") * 100,
    ],
    ids=["tagged", "untagged"],
)
@pytest.mark.parametrize("response_format", ["wav", "pcm"])
async def test_mp3_response_fails_naming_it_instead_of_playing_static(
    enhanced_handler, mock_info, mock_clients, caplog, response_format, mp3
):
    """Test a backend answering with MP3, tagged or not and whatever was requested, fails with what it sent."""
    _, tts_client = mock_clients
    voice = mock_info.tts[0].voices[0]
    enhanced_handler._tts_extra_body = {"response_format": response_format}
    _mock_speech_response(tts_client, [mp3])

    with caplog.at_level(logging.ERROR, logger="wyoming_openai.handler"):
        assert await _stream_whole_request(enhanced_handler, voice, "Hello") is None

    assert "TTS backend returned audio that cannot be played: TTS response is MP3" in caplog.text
    assert not [call for call in enhanced_handler.write_event.call_args_list if call.args[0].type == "audio-chunk"]


@pytest.mark.asyncio
async def test_buffered_sentence_is_played_in_stream_sized_pieces_that_yield_to_the_event_loop(enhanced_handler):
    """Test a long buffered sentence does not hold the event loop for its whole conversion."""
    pcm = bytes(range(256)) * 40
    wav = _pcm_wav_with_unbounded_sizes(pcm)
    loop_turns = 0

    async def count_loop_turns():
        nonlocal loop_turns
        while True:
            loop_turns += 1
            await asyncio.sleep(0)

    turns_at_writes = []

    async def write(event):
        if event.type == "audio-chunk":
            turns_at_writes.append(loop_turns)

    enhanced_handler.write_event = AsyncMock(side_effect=write)
    counter = asyncio.create_task(count_loop_turns())
    try:
        assert await _play_buffered_sentence(enhanced_handler, wav, True, 0, WAV_AUDIO_FORMAT) is not None
    finally:
        counter.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await counter

    events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
    assert b"".join(event.payload for event in events if event.type == "audio-chunk") == pcm
    assert len(turns_at_writes) == -(-len(wav) // TTS_CHUNK_SIZE)
    # Other tasks ran between every two pieces
    assert turns_at_writes == sorted(set(turns_at_writes))


@pytest.mark.asyncio
async def test_buffered_float_wav_sentence_is_played_as_pcm16(enhanced_handler):
    """Test a buffered sentence of IEEE float WAV is converted like a streamed one."""
    wav = float_wav([0.0, 0.5, -0.5, 1.0] * 1000)

    assert await _play_buffered_sentence(enhanced_handler, wav, True, 0, WAV_AUDIO_FORMAT) is not None

    events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
    assert (events[0].type, events[0].data["width"]) == ("audio-start", 2)
    pcm = b"".join(event.payload for event in events if event.type == "audio-chunk")
    assert pcm == struct.pack("<4h", 0, 16384, -16384, 32767) * 1000


@pytest.mark.asyncio
async def test_container_warning_is_logged_when_the_response_then_fails(
    enhanced_handler, mock_info, mock_clients, caplog
):
    """Test a response played as raw PCM still says so when it fails midway, since that may explain the static."""
    _, tts_client = mock_clients
    voice = mock_info.tts[0].voices[0]

    async def respond(text):
        # Raw PCM, although a WAV file was requested
        yield b"\x00\x01" * 1024
        raise RuntimeError("read timed out")

    _mock_speech_response(tts_client, respond)
    with caplog.at_level(logging.WARNING, logger="wyoming_openai.handler"):
        assert await _stream_whole_request(enhanced_handler, voice, "Hello") is None

    assert "Error streaming TTS audio: read timed out" in caplog.text
    assert caplog.text.count("TTS response is not a WAV file; playing it as raw PCM") == 1


@pytest.mark.asyncio
async def test_container_warning_is_logged_as_soon_as_the_response_shows_it(
    enhanced_handler, mock_info, mock_clients, caplog
):
    """Test the warning that explains static comes while the response is still playing, not after it."""
    _, tts_client = mock_clients
    voice = mock_info.tts[0].voices[0]
    first_chunk_played = asyncio.Event()
    finish = asyncio.Event()

    async def respond(text):
        yield b"\x00\x01" * 1024
        # Asked for the next bytes only once the first ones were played
        first_chunk_played.set()
        await finish.wait()
        yield b"\x00\x01" * 1024

    _mock_speech_response(tts_client, respond)
    warning = "TTS response is not a WAV file; playing it as raw PCM"
    with caplog.at_level(logging.WARNING, logger="wyoming_openai.handler"):
        synthesis = asyncio.create_task(_stream_whole_request(enhanced_handler, voice, "Hello"))
        async with asyncio.timeout(1):
            await first_chunk_played.wait()
        assert caplog.text.count(warning) == 1
        finish.set()
        assert await synthesis is not None

    assert caplog.text.count(warning) == 1


@pytest.mark.asyncio
async def test_long_metadata_before_the_audio_plays_buffered_or_streamed_alike(
    enhanced_handler, mock_info, mock_clients
):
    """Test metadata longer than the 64 KiB the header was once held to no longer stops either path."""
    _, tts_client = mock_clients
    voice = mock_info.tts[0].voices[0]
    pcm = b"\x00\x01" * 240
    wav = wav_file(fmt_chunk_body(1, bits=16), pcm, chunks=riff_chunk(b"LIST", b"\x00" * 70000))

    def played_audio():
        events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
        enhanced_handler.write_event.reset_mock()
        return b"".join(event.payload for event in events if event.type == "audio-chunk")

    assert await _play_buffered_sentence(enhanced_handler, wav, True, 0, WAV_AUDIO_FORMAT) is not None
    assert played_audio() == pcm
    # As httpx reads a live response
    _mock_speech_response(tts_client, [wav[i : i + TTS_CHUNK_SIZE] for i in range(0, len(wav), TTS_CHUNK_SIZE)])
    assert await _stream_whole_request(enhanced_handler, voice, "Hello") is not None
    assert played_audio() == pcm


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("unplayable", "error"),
    [
        (wav_file(fmt_chunk_body(7, bits=8), b"\xff" * 240), "WAV audio format 0x0007 (mu-law)"),
        (_pcm_wav_with_unbounded_sizes(b"\x00\x01" * 240)[:30], "TTS response ended after 30 bytes, before its WAV"),
    ],
    ids=["mu-law", "header-cut-off"],
)
async def test_unplayable_wav_sentence_ends_a_streaming_synthesis(
    enhanced_handler, mock_info, mock_clients, caplog, unplayable, error
):
    """Test a sentence whose WAV cannot be played stops the audio once, after what played."""
    _, tts_client = mock_clients
    mock_info.tts[0].supports_synthesize_streaming = True
    spoken = _pcm_wav_with_unbounded_sizes(b"\x00\x01" * 240)
    _mock_speech_response(tts_client, lambda text: [unplayable if text.startswith("Second") else spoken])
    assert await enhanced_handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="alloy")).event())

    with caplog.at_level(logging.ERROR, logger="wyoming_openai.handler"):
        assert not await enhanced_handler._process_ready_sentences(["First one.", "Second one.", "Third one."])

    assert caplog.text.count(f"TTS backend returned audio that cannot be played: {error}") == 1
    event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
    assert event_types == ["audio-start", "audio-chunk", "error", "audio-stop", "synthesize-stopped"]
    await asyncio.sleep(0)
    assert not enhanced_handler._synthesis_tasks
    await enhanced_handler._drain_background_tasks()


@pytest.mark.asyncio
async def test_a_request_whose_wav_ends_inside_its_header_fails_without_audio(
    enhanced_handler, mock_info, mock_clients, caplog
):
    """Test a whole request fails, naming the cause, when its WAV response ends inside its header."""
    _, tts_client = mock_clients
    voice = mock_info.tts[0].voices[0]
    _mock_speech_response(tts_client, [_pcm_wav_with_unbounded_sizes(b"\x00\x01" * 240)[:30]])

    with caplog.at_level(logging.ERROR, logger="wyoming_openai.handler"):
        assert await _stream_whole_request(enhanced_handler, voice, "Hello") is None

    assert "TTS backend returned audio that cannot be played: TTS response ended after 30 bytes" in caplog.text
    assert not [call for call in enhanced_handler.write_event.call_args_list if call.args[0].type == "audio-chunk"]
