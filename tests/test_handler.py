import asyncio
import base64
import builtins
import io
import logging
import struct
import wave
from collections.abc import AsyncIterator, Callable
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import pytest_asyncio
from openai import omit
from wyoming.asr import Transcript, TranscriptChunk
from wyoming.event import Event
from wyoming.info import Attribution, TtsVoice
from wyoming.tts import SynthesizeChunk, SynthesizeStart, SynthesizeVoice

from wyoming_openai.compatibility import (
    OpenAIBackend,
    TtsVoiceModel,
    create_asr_programs,
    create_info,
    create_tts_programs,
    create_tts_voices,
)
from wyoming_openai.handler import (
    OpenAIEventHandler,
    TtsAudioFormat,
    TtsStreamError,
    _has_speakable_content,
)


def _riff_chunk(chunk_id: bytes, payload: bytes) -> bytes:
    """Build a word-aligned RIFF chunk."""
    padding = b"\x00" if len(payload) % 2 else b""
    return chunk_id + struct.pack("<I", len(payload)) + payload + padding


def _pcm_wav_with_trailing_metadata(pcm: bytes) -> bytes:
    """Build a valid PCM WAV with LIST/INFO and C2PA chunks after data."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(pcm)

    wav_data = wav_buffer.getvalue()
    metadata = _riff_chunk(b"LIST", b"INFOISFT\x08\x00\x00\x00CrispASR") + _riff_chunk(b"C2PA", b"manifest")
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
    return MagicMock(name="reader"), MagicMock(name="writer")


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

    handler = OpenAIEventHandler(
        dummy_reader_writer[0],
        dummy_reader_writer[1],
        info=dummy_info,
        stt_client=stt_client,
        tts_client=tts_client,
    )
    handler.write_event = AsyncMock()

    await handler.stop()

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
    assert handler._is_synthesizing is False
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
    assert handler._is_synthesizing is False
    assert handler._synthesis_buffer == []
    assert handler._text_accumulator == ""


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
        result = await handler.handle_event(SynthesizeChunk(text="Test sentence. Another one.").event())

    assert result is False
    event_types = [call.args[0].type for call in handler.write_event.call_args_list]
    assert "synthesize-stopped" in event_types
    assert handler._is_synthesizing is False
    # Verify state was fully reset
    assert handler._audio_started is False
    assert handler._current_timestamp == 0
    assert handler._resolved_synthesis_voice is None
    assert handler._synthesis_language is None


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
        assert handler._is_synthesizing is False

        # Try to start again - should work
        await handler.handle_event(SynthesizeStart(voice=start_voice).event())
        assert handler._is_synthesizing is True

        # Second failure
        result2 = await handler.handle_event(SynthesizeChunk(text="Another chunk. And another.").event())
        assert result2 is False

        # Verify state is consistently reset
        assert handler._is_synthesizing is False
        assert handler._synthesis_buffer == []
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
    async def test_handle_audio_stop_event(self, enhanced_handler):
        """Test handling of AudioStop event."""
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
    async def test_handle_realtime_transcription_flow(self, enhanced_handler, mock_info, mock_clients):
        """Test realtime STT emits deltas, final transcript, and cleanup without audio transcriptions API."""
        stt_client, _ = mock_clients
        mock_info.asr[0].models[0].name = "gpt-realtime-whisper"
        enhanced_handler._stt_realtime_models = {"gpt-realtime-whisper"}

        connection = _FakeRealtimeConnection(
            [
                _FakeRealtimeServerEvent("conversation.item.input_audio_transcription.delta", delta="Hello"),
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
        assert event_types == ["transcript-start", "transcript-chunk", "transcript", "transcript-stop"]
        chunk_event = next(
            call.args[0]
            for call in enhanced_handler.write_event.call_args_list
            if call.args[0].type == "transcript-chunk"
        )
        assert TranscriptChunk.from_event(chunk_event).text == "Hello"
        transcript_event = next(
            call.args[0] for call in enhanced_handler.write_event.call_args_list if call.args[0].type == "transcript"
        )
        assert Transcript.from_event(transcript_event).text == "Hello world"
        # The closing handshake runs in the background so it cannot hold up the Wyoming client
        await enhanced_handler._drain_background_tasks()
        assert manager.exited is True
        assert connection.closed is True
        assert enhanced_handler._realtime_receive_task is None

    @pytest.mark.asyncio
    async def test_handle_realtime_transcription_error_cleans_up(self, enhanced_handler, mock_info, mock_clients):
        """Test realtime STT errors emit final Transcript and clean up the websocket task."""
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
        await enhanced_handler.handle_event(Event(type="audio-stop"))

        stt_client.realtime.connect.assert_called_once_with(extra_query={"intent": "transcription"})
        stt_client.audio.transcriptions.create.assert_not_called()
        event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
        assert event_types == ["transcript-start", "transcript", "transcript-stop"]
        transcript_event = next(
            call.args[0] for call in enhanced_handler.write_event.call_args_list if call.args[0].type == "transcript"
        )
        assert Transcript.from_event(transcript_event).text == ""
        # The closing handshake runs in the background so it cannot hold up the Wyoming client
        await enhanced_handler._drain_background_tasks()
        assert manager.exited is True
        assert connection.closed is True
        assert enhanced_handler._realtime_receive_task is None

    @pytest.mark.asyncio
    async def test_handle_realtime_audio_stop_missing_future_emits_final_transcript(self, enhanced_handler):
        """Test realtime STT stop emits final Transcript when future state is missing."""
        connection = Mock()
        connection.input_audio_buffer.commit = AsyncMock()
        connection.close = AsyncMock()
        enhanced_handler._is_recording = True
        enhanced_handler._realtime_connection = connection
        enhanced_handler._realtime_transcript_future = None
        enhanced_handler.write_event = AsyncMock()

        await enhanced_handler._handle_realtime_audio_stop()

        event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
        assert event_types == ["transcript", "transcript-stop"]
        transcript_event = enhanced_handler.write_event.call_args_list[0].args[0]
        assert Transcript.from_event(transcript_event).text == ""
        connection.input_audio_buffer.commit.assert_not_called()

    def test_convert_audio_to_realtime_pcm_resamples_to_24khz(self, enhanced_handler):
        """Test Wyoming PCM is converted to OpenAI Realtime's 24 kHz PCM16 format."""
        source_audio = b"\x00\x00\xe8\x03"

        converted = enhanced_handler._convert_audio_to_realtime_pcm(
            source_audio, sample_rate=16000, audio_width=2, audio_channels=1
        )

        assert len(converted) == 6

    @pytest.mark.asyncio
    async def test_realtime_transcription_session_includes_configured_prompt(self, enhanced_handler, mock_info):
        """Test realtime STT preserves configured prompt in the session payload."""
        mock_info.asr[0].models[0].name = "gpt-4o-transcribe"
        enhanced_handler._stt_realtime_models = {"gpt-4o-transcribe"}

        assert await enhanced_handler.handle_event(
            Event(type="transcribe", data={"language": "en", "name": "gpt-4o-transcribe"})
        )

        session = enhanced_handler._get_realtime_transcription_session()

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

        timestamp = await enhanced_handler._stream_audio_to_wyoming(
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

        timestamp = await enhanced_handler._stream_audio_to_wyoming(
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
        wav_params = enhanced_handler._parse_wav_header(wav_data)
        assert wav_params is not None
        data_offset = wav_params[3]
        truncated_wav = wav_data[: data_offset + len(pcm) - 10]

        timestamp = await enhanced_handler._stream_audio_to_wyoming(
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
        assert "ended with 10 declared PCM bytes missing" in caplog.text

    @pytest.mark.asyncio
    async def test_stream_audio_to_wyoming_accepts_unbounded_wav_data(self, enhanced_handler, caplog):
        """Test buffered WAV conversion accepts an unbounded data chunk size."""
        pcm = b"\x00\x01" * 240
        wav_data = _pcm_wav_with_unbounded_sizes(pcm)

        wav_params = enhanced_handler._parse_wav_header(wav_data)
        assert wav_params == (24000, 1, 2, 44, None)

        timestamp = await enhanced_handler._stream_audio_to_wyoming(
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
        assert "declared PCM bytes missing" not in caplog.text
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

        timestamp = await enhanced_handler._stream_tts_audio(voice, "Hello world", send_audio_start=True)

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
        timestamp = await enhanced_handler._stream_tts_audio(voice, "Hello world", send_audio_start=True)

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
        wav_params = enhanced_handler._parse_wav_header(wav_data)
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

            timestamp = await enhanced_handler._stream_tts_audio(voice, "Hello world", send_audio_start=True)

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
        """Test incremental WAV conversion reports missing declared PCM bytes."""
        _, tts_client = mock_clients
        pcm = b"\x00\x01" * 240
        wav_data = _pcm_wav_with_trailing_metadata(pcm)
        wav_params = enhanced_handler._parse_wav_header(wav_data)
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

        timestamp = await enhanced_handler._stream_tts_audio(voice, "Hello world", send_audio_start=True)

        audio_chunk_events = [
            call.args[0] for call in enhanced_handler.write_event.call_args_list if call.args[0].type == "audio-chunk"
        ]
        assert b"".join(event.payload for event in audio_chunk_events) == pcm[:-10]
        assert timestamp == pytest.approx(470 / 2 / 24000 * 1000)
        assert "ended with 10 declared PCM bytes missing" in caplog.text

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

        timestamp = await enhanced_handler._stream_tts_audio(voice, "Hello world", send_audio_start=True)

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
        assert "declared PCM bytes missing" not in caplog.text

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
        handler._process_ready_sentences = AsyncMock(return_value=True)

        assert await handler.handle_event(Event(type="synthesize-start", data={"voice": {"name": "voice1"}})) is True
        assert await handler.handle_event(Event(type="synthesize-chunk", data={"text": "First sentence. "})) is True
        assert await handler.handle_event(Event(type="synthesize-chunk", data={"text": "Todo "})) is True
        # yasbd keeps the inter-sentence space but attributes it to the retained tail
        assert handler._text_accumulator == " Todo "

        assert await handler.handle_event(Event(type="synthesize-chunk", data={"text": "listo."})) is True
        assert handler._text_accumulator == " Todo listo."
        handler._process_ready_sentences.assert_awaited_once_with(["First sentence."], None)

    @pytest.mark.asyncio
    async def test_stream_tts_audio_incremental_includes_configured_tts_extra_body(
        self, enhanced_handler, mock_clients
    ):
        """Test incremental TTS requests include configured extra_body."""
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

        await enhanced_handler._stream_tts_audio_incremental("Hello world", voice)

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
    assert enhanced_handler._synthesis_text_format == "ssml"

    result = await enhanced_handler.handle_event(
        Event(type="synthesize-chunk", data={"text": "Hello <break/>world"})
    )

    assert result is True
    assert enhanced_handler._text_accumulator == "Hello world"
    assert enhanced_handler._synthesis_buffer == ["Hello world"]


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

    assert "".join(enhanced_handler._synthesis_buffer) == "Hello world"
    assert enhanced_handler._text_accumulator == "Hello world"


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

    assert enhanced_handler._text_accumulator == "Hello world"
    assert "".join(enhanced_handler._synthesis_buffer) == "Hello world"


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

    assert enhanced_handler._text_accumulator == "Hello world"
    assert "".join(enhanced_handler._synthesis_buffer) == "Hello world"


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

    assert enhanced_handler._text_accumulator == "Tom & Jerry"
    assert "".join(enhanced_handler._synthesis_buffer) == "Tom & Jerry"


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
    assert enhanced_handler._text_accumulator == "Rules"

    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": "regulations"}))

    assert enhanced_handler._text_accumulator == "Rules &regulations"
    assert "".join(enhanced_handler._synthesis_buffer) == "Rules &regulations"


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

    assert enhanced_handler._text_accumulator == "Foo &ampregulations"
    assert "".join(enhanced_handler._synthesis_buffer) == "Foo &ampregulations"


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

    assert enhanced_handler._text_accumulator == "Tom & Jerry"
    assert "".join(enhanced_handler._synthesis_buffer) == "Tom & Jerry"


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

    assert enhanced_handler._text_accumulator == "Foo& bar"
    assert "".join(enhanced_handler._synthesis_buffer) == "Foo& bar"


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

    assert enhanced_handler._text_accumulator == "Foo & bar"
    assert "".join(enhanced_handler._synthesis_buffer) == "Foo & bar"


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
    assert enhanced_handler._text_accumulator == "Foo"

    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": ";<vendor"}))
    assert enhanced_handler._text_accumulator == "Foo &"

    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": ">bar"}))

    assert enhanced_handler._text_accumulator == "Foo & bar"
    assert "".join(enhanced_handler._synthesis_buffer) == "Foo & bar"


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

    assert enhanced_handler._text_accumulator == "Hello world"
    assert "".join(enhanced_handler._synthesis_buffer) == "Hello world"


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

    assert enhanced_handler._text_accumulator == "Hello world"
    assert "".join(enhanced_handler._synthesis_buffer) == "Hello world"


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

    assert enhanced_handler._text_accumulator == "Hello world"
    assert "".join(enhanced_handler._synthesis_buffer) == "Hello world"


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

    assert enhanced_handler._text_accumulator == "Hello world"
    assert "".join(enhanced_handler._synthesis_buffer) == "Hello world"


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
    enhanced_handler._segmenters["en"] = segmenter
    enhanced_handler._process_ready_sentences = AsyncMock(return_value=True)

    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": "First. Second. "}))
    assert enhanced_handler._text_accumulator == "Second."

    await enhanced_handler.handle_event(Event(type="synthesize-chunk", data={"text": " Next"}))

    assert "".join(enhanced_handler._synthesis_buffer) == "First. Second. Next"
    assert segmenter.segment.call_args_list[1].args == ("Second. Next",)
    assert enhanced_handler._text_accumulator == "Second. Next"


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

    assert enhanced_handler._text_accumulator == "Hello world"
    assert "".join(enhanced_handler._synthesis_buffer) == "Hello world"


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

    assert enhanced_handler._text_accumulator == "world"
    assert "".join(enhanced_handler._synthesis_buffer) == "world"


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

    assert enhanced_handler._text_accumulator == "spoken"
    assert "".join(enhanced_handler._synthesis_buffer) == "spoken"


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
    assert "<" + flooded in "".join(enhanced_handler._synthesis_buffer)

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
    assert "&#123" + flooded in "".join(enhanced_handler._synthesis_buffer)


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
    enhanced_handler._audio_started = True  # take the incremental early-exit path
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

    assert enhanced_handler._text_accumulator == "&" + "<" + flood
    assert "".join(enhanced_handler._synthesis_buffer) == "&" + "<" + flood


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
    enhanced_handler._audio_started = True  # take the incremental early-exit path
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
    assert enhanced_handler._text_accumulator == "Foo"

    # A trailing entity-like fragment never completed into an entity is literal
    # text; flushing must not decode it ("&amp" must not become "&")
    enhanced_handler._process_ready_sentences = AsyncMock(return_value=True)
    enhanced_handler._audio_started = True  # take the incremental early-exit path
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

    session = enhanced_handler._get_realtime_transcription_session()

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
    tts_client.realtime.connect.assert_called_once_with(model=REALTIME_TTS_MODEL)
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
        # A failure before any audio emits nothing
        ([_FakeRealtimeServerEvent("error", error={"message": "invalid voice"})], []),
        # Audio that was already started is terminated
        (_realtime_tts_events(status="failed"), ["audio-start", "audio-chunk", "audio-stop"]),
        # A completed response that never produced audio is a failure, not an empty success
        ([_FakeRealtimeServerEvent("response.done", response=Mock(status="completed"))], []),
    ],
)
async def test_realtime_tts_failure_closes_connection(enhanced_handler, realtime_tts, events, expected_event_types):
    """Test Realtime TTS errors fail the synthesis and still close the websocket."""
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
    """Test a Realtime response failing after audio closes the stream exactly once when streaming."""
    realtime_tts(_realtime_tts_events(status="failed"))
    voice = SynthesizeVoice(name="alloy")

    await enhanced_handler.handle_event(SynthesizeStart(voice=voice).event())
    result = await enhanced_handler.handle_event(SynthesizeChunk(text="First sentence. Second one.").event())

    assert result is False
    event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
    assert event_types == ["audio-start", "audio-chunk", "audio-stop", "synthesize-stopped"]


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
    assert event_types == ["synthesize-stopped"]


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
    assert event_types == ["audio-start", "audio-chunk", "audio-stop"]
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
async def test_realtime_tts_fidelity_joins_transcript_parts(enhanced_handler, realtime_tts, caplog):
    """Test a response spoken in several transcript parts is compared as a whole."""
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

    with patch.object(enhanced_handler, "_parse_wav_header") as parse_wav_header:
        await enhanced_handler._stream_audio_to_wyoming(result.audio, True, 0, audio_format=audio_format)

    parse_wav_header.assert_not_called()
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

    session = enhanced_handler._get_realtime_transcription_session()

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
    assert event_types == ["audio-start", "audio-stop"]

    assert await enhanced_handler._stream_tts_audio(voice, "Hello", send_audio_start=False) is None

    result = await enhanced_handler._get_tts_audio_stream("Hello", voice)
    assert await enhanced_handler._stream_audio_to_wyoming(result.audio, False, 0, WAV_AUDIO_FORMAT, "Hello") is None


@pytest.mark.asyncio
async def test_incremental_tts_sentence_without_audio_aborts_instead_of_going_missing(
    enhanced_handler, mock_info, mock_clients
):
    """Test a sentence that comes back as a header-only WAV fails the synthesis rather than being dropped."""
    _, tts_client = mock_clients
    mock_info.tts[0].supports_synthesize_streaming = True
    spoken = _pcm_wav_with_unbounded_sizes(b"\x00\x01" * 240)
    _mock_speech_response(tts_client, lambda text: [_unbounded_header_only_wav() if "Second" in text else spoken])

    assert await enhanced_handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="alloy")).event())
    assert not await enhanced_handler._process_ready_sentences(["First one.", "Second one.", "Third one."])

    event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
    assert event_types == ["audio-start", "audio-chunk", "audio-stop", "synthesize-stopped"]


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
    assert await enhanced_handler._stream_tts_audio(voice, "...", send_audio_start=False, start_timestamp=5) == 5

    _mock_speech_response(tts_client, [header_only_wav()])
    result = await enhanced_handler._get_tts_audio_stream("...", voice)
    assert await enhanced_handler._stream_audio_to_wyoming(result.audio, False, 5, WAV_AUDIO_FORMAT) == 5


@pytest.mark.asyncio
async def test_buffered_http_pcm_is_forwarded_without_wav_sniffing(enhanced_handler, mock_info, mock_clients):
    """Test the buffered path reads the format the direct path uses, so `response_format: pcm` is not sniffed."""
    _, tts_client = mock_clients
    voice = mock_info.tts[0].voices[0]
    assert enhanced_handler._get_tts_audio_format(voice) == TtsAudioFormat(headerless=False, rate=24000)

    enhanced_handler._tts_extra_body = {"response_format": "pcm"}
    audio_format = enhanced_handler._get_tts_audio_format(voice)
    assert audio_format == TtsAudioFormat(headerless=True, rate=24000)

    pcm = b"\x00\x01" * 240
    _mock_speech_response(tts_client, [pcm])
    result = await enhanced_handler._get_tts_audio_stream("Hello", voice)
    with patch.object(enhanced_handler, "_parse_wav_header") as parse_wav_header:
        assert await enhanced_handler._stream_audio_to_wyoming(result.audio, True, 0, audio_format=audio_format) == 10

    parse_wav_header.assert_not_called()
    assert enhanced_handler.write_event.call_args.args[0].payload == pcm


@pytest.mark.asyncio
async def test_http_tts_zero_size_wav_header_followed_by_audio_is_played(enhanced_handler, mock_info, mock_clients):
    """Test audio after a header that declares no samples is played, as servers streaming unknown lengths send it."""
    _, tts_client = mock_clients
    voice = mock_info.tts[0].voices[0]
    pcm = b"\x00\x01" * 240

    _mock_speech_response(tts_client, [_header_only_wav(), pcm])
    assert await enhanced_handler._stream_tts_audio(voice, "Hello", send_audio_start=True) == 10
    events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
    assert [event.type for event in events] == ["audio-start", "audio-chunk"]
    assert events[1].payload == pcm

    enhanced_handler.write_event.reset_mock()
    wav = _header_only_wav() + pcm
    assert await enhanced_handler._stream_audio_to_wyoming(wav, False, 0, WAV_AUDIO_FORMAT) == 10
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
    enhanced_handler.write_event.assert_not_called()

    _mock_speech_response(tts_client, [])
    assert await enhanced_handler._stream_tts_audio(voice, "Hello", send_audio_start=False) is None

    # A buffered sentence fails when it is played, by the same rule
    _mock_speech_response(tts_client, [])
    result = await enhanced_handler._get_tts_audio_stream("Hello", voice)
    assert await enhanced_handler._stream_audio_to_wyoming(result.audio, True, 0, WAV_AUDIO_FORMAT, "Hello") is None
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
async def test_http_tts_truncated_wav_without_samples_fails(enhanced_handler, mock_clients):
    """Test a WAV that declares PCM data but delivers only its header fails and closes the opened stream."""
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
    assert event_types == ["audio-start", "audio-stop"]

    # As on the buffered path, text with nothing to say does not excuse the missing samples
    _mock_speech_response(tts_client, [header_only])
    assert not await enhanced_handler.handle_event(
        Event(type="synthesize", data={"text": "...", "voice": {"name": "alloy"}})
    )


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
@pytest.mark.parametrize(("configured", "expected_peak"), [(None, 3), (1, 1), (2, 2), (5, 5)])
async def test_tts_requests_overlap_up_to_configured_limit(
    dummy_info, dummy_clients, dummy_reader_writer, configured, expected_peak
):
    """Three sentences are synthesized at once by default; tts_concurrent_requests changes that limit."""
    stt_client, tts_client = dummy_clients
    reader, writer = dummy_reader_writer
    speech = _OverlapRecordingSpeech()
    tts_client.audio.speech.with_streaming_response = speech
    extra_kwargs = {} if configured is None else {"tts_concurrent_requests": configured}
    handler = OpenAIEventHandler(
        reader, writer, info=dummy_info, stt_client=stt_client, tts_client=tts_client, **extra_kwargs
    )
    voice = handler._get_voice("voice1")
    assert voice is not None

    results = await asyncio.gather(*(handler._get_tts_audio_stream(f"Sentence {i}.", voice) for i in range(6)))

    assert all(result.audio for result in results)
    assert speech.peak == expected_peak


@pytest.mark.parametrize("value", [0, -1])
def test_init_rejects_non_positive_tts_concurrent_requests(dummy_info, dummy_clients, dummy_reader_writer, value):
    stt_client, tts_client = dummy_clients
    reader, writer = dummy_reader_writer

    with pytest.raises(ValueError, match="tts_concurrent_requests must be at least 1"):
        OpenAIEventHandler(
            reader,
            writer,
            info=dummy_info,
            stt_client=stt_client,
            tts_client=tts_client,
            tts_concurrent_requests=value,
        )


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

    transcription = enhanced_handler._get_realtime_transcription_session()["audio"]["input"]["transcription"]
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

    transcription = enhanced_handler._get_realtime_transcription_session()["audio"]["input"]["transcription"]
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


def test_nameless_voice_request_prefers_the_speech_api(enhanced_handler, mock_info):
    """Test a request that names no voice does not default onto a Realtime model listed first."""
    speech_voice, realtime_voice = _add_realtime_voice_first(mock_info, enhanced_handler)

    assert enhanced_handler._validate_tts_voice_and_language(None, None) is speech_voice

    # Realtime is still the default when it is the only transport
    mock_info.tts[0].voices.remove(speech_voice)
    assert enhanced_handler._validate_tts_voice_and_language(None, None) is realtime_voice


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
    assert await enhanced_handler._stream_audio_to_wyoming(result.audio, True, 5, audio_format, "...") == 5
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
    assert enhanced_handler._audio_started is False

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

    assert await enhanced_handler._stream_audio_to_wyoming(header_only, True, 0, WAV_AUDIO_FORMAT) is None
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
    assert event_types == ["audio-start", "audio-chunk", "audio-stop"]
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
        assert await enhanced_handler._stream_tts_audio(voice, "...", send_audio_start=True) == 0
        assert [call.args[0].type for call in enhanced_handler.write_event.call_args_list] == ["audio-start"]

    # A buffered sentence without samples leaves the stream for the next sentence to open
    enhanced_handler.write_event.reset_mock()
    enhanced_handler._audio_started = False  # The direct calls above opened it and were never stopped
    assert await enhanced_handler._stream_audio_to_wyoming(wav, True, 0, WAV_AUDIO_FORMAT) == 0
    enhanced_handler.write_event.assert_not_called()
    assert enhanced_handler._audio_started is False


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
    assert await enhanced_handler._stream_tts_audio(voice, "Hello", send_audio_start=True) is not None
    events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
    assert b"".join(event.payload for event in events if event.type == "audio-chunk") == pcm

    enhanced_handler.write_event.reset_mock()
    assert await enhanced_handler._stream_audio_to_wyoming(_header_only_wav() + pcm, True, 0, WAV_AUDIO_FORMAT)
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
    assert enhanced_handler._resolved_synthesis_voice is None
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
    assert await enhanced_handler._stream_audio_to_wyoming(result.audio, False, 5, audio_format, "...") == 5

    # Text with something to say still has to produce audio
    result = await enhanced_handler._get_tts_audio_stream("Hello", voice)
    assert await enhanced_handler._stream_audio_to_wyoming(result.audio, False, 5, audio_format, "Hello") is None


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
        enhanced_handler._audio_started = False
        enhanced_handler._current_timestamp = 0
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
    assert await enhanced_handler._stream_audio_to_wyoming(wav, True, 0, WAV_AUDIO_FORMAT) == 0
    enhanced_handler.write_event.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("riff_size", [0x7FFFFFFF, 0xFFFFFFFF])
async def test_zero_size_wav_with_a_placeholder_riff_size_is_played(enhanced_handler, riff_size):
    """Test a RIFF size that stands in for an unknown length marks streamed audio, which is played."""
    pcm = b"\x00\x01" * 240
    header = _header_only_wav()
    wav = header[:4] + struct.pack("<I", riff_size) + header[8:] + pcm

    assert await enhanced_handler._stream_audio_to_wyoming(wav, True, 0, WAV_AUDIO_FORMAT) == 10
    assert enhanced_handler.write_event.call_args.args[0].payload == pcm


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
        ("Hello world", "Sure, I can help with that.", True),
    ],
)
def test_realtime_tts_fidelity_normalizes_unicode(enhanced_handler, caplog, requested, spoken, expect_warning):
    """Test the fidelity check does not warn about text that only differs in Unicode form or case folding."""
    with caplog.at_level(logging.WARNING, logger="wyoming_openai.handler"):
        enhanced_handler._check_realtime_tts_fidelity(requested, spoken)

    assert ("differs from the request" in caplog.text) is expect_warning


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
    assert event_types == ["audio-start", "audio-chunk", "audio-stop", "synthesize-stopped"]
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
    """Test sentences that produced no audio are not sent to the backend again by the synthesize-stop fallback."""
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
    assert enhanced_handler._synthesized_incrementally is False
    assert enhanced_handler._audio_started is False


@pytest.mark.asyncio
@pytest.mark.parametrize("payload_size", [8192, 60000, 70000])
async def test_zero_size_wav_does_not_play_a_large_trailing_riff_chunk(
    enhanced_handler, mock_info, mock_clients, payload_size
):
    """Test metadata far larger than a stream chunk is not played, whether it arrives whole or split up."""
    _, tts_client = mock_clients
    voice = mock_info.tts[0].voices[0]
    wav = _empty_wav_with_trailer(_riff_chunk(b"JUNK", b"\x01" * payload_size))

    assert await enhanced_handler._stream_audio_to_wyoming(wav, True, 0, WAV_AUDIO_FORMAT) == 0
    enhanced_handler.write_event.assert_not_called()

    # The speech response is read in pieces of TTS_CHUNK_SIZE, so the chunk is never seen whole at once
    _mock_speech_response(tts_client, [wav[offset : offset + 2048] for offset in range(0, len(wav), 2048)])
    assert await enhanced_handler._stream_tts_audio(voice, "...", send_audio_start=True) == 0
    assert [call.args[0].type for call in enhanced_handler.write_event.call_args_list] == ["audio-start"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wav",
    [
        _empty_wav_with_trailer(_riff_chunk(b"LIST", b"\x01" * 40000) + _riff_chunk(b"C2PA", b"\x02" * 40000)),
        # A response cut off inside a chunk the RIFF size covers
        _empty_wav_with_trailer(_riff_chunk(b"LIST", b"\x01" * 12) + _riff_chunk(b"C2PA", b"\x02" * 100))[:-90],
        # A finite RIFF size is taken at its word however much metadata it covers
        _empty_wav_with_trailer(_riff_chunk(b"JUNK", b"\x01" * 64000) * 17),
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

    assert await enhanced_handler._stream_audio_to_wyoming(wav, True, 0, WAV_AUDIO_FORMAT) == 0
    enhanced_handler.write_event.assert_not_called()

    _mock_speech_response(tts_client, [wav[offset : offset + 2048] for offset in range(0, len(wav), 2048)])
    assert await enhanced_handler._stream_tts_audio(voice, "...", send_audio_start=True) == 0
    assert [call.args[0].type for call in enhanced_handler.write_event.call_args_list] == ["audio-start"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("batches", "expected_event_types", "expected_stop_timestamp"),
    [
        # The cancelled sentence streams directly, into a stream the batch before it opened; the 10 ms it
        # sent before it was cancelled count towards the stop
        ([["Hello there."], ["Stalls here."]], ["audio-start", "audio-chunk", "audio-chunk", "audio-stop"], 20),
        # The cancelled sentence is still being buffered behind the one that opened the stream
        ([["Hello there.", "Stalls here."]], ["audio-start", "audio-chunk", "audio-stop"], 10),
    ],
)
async def test_stop_closes_the_audio_an_earlier_sentence_opened(
    enhanced_handler, mock_info, mock_clients, batches, expected_event_types, expected_stop_timestamp
):
    """Test a sentence cancelled by stop() does not leave a stream an earlier sentence opened without audio-stop."""
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
        while len(enhanced_handler.write_event.call_args_list) < len(expected_event_types) - 1:
            await asyncio.sleep(0)

    with patch("wyoming.server.AsyncEventHandler.stop", new=AsyncMock()):
        await enhanced_handler.stop()
    with pytest.raises(asyncio.CancelledError):
        await task

    events = [call.args[0] for call in enhanced_handler.write_event.call_args_list]
    assert [event.type for event in events] == expected_event_types
    assert events[-1].data["timestamp"] == expected_stop_timestamp
    assert enhanced_handler._audio_started is False


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
        "audio-stop",
        "synthesize-stopped",
    ]
    assert events[3].data["timestamp"] == 20


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("send_audio_start", "expected_event_types"),
    [
        (True, ["audio-start", "audio-chunk", "audio-stop"]),
        # A stream this call did not open is left to whoever opened it
        (False, ["audio-chunk"]),
    ],
)
async def test_cancelled_tts_stream_closes_the_audio_it_opened(
    enhanced_handler, mock_info, mock_clients, send_audio_start, expected_event_types
):
    """Test a sentence cancelled after its audio started still ends the stream for a client that is connected."""
    _, tts_client = mock_clients
    voice = mock_info.tts[0].voices[0]
    enhanced_handler._tts_extra_body = {"response_format": "pcm"}
    first_chunk_written = asyncio.Event()

    async def respond(text):
        yield b"\x00\x01" * 240
        first_chunk_written.set()
        await asyncio.Event().wait()

    _mock_speech_response(tts_client, respond)

    task = asyncio.create_task(enhanced_handler._stream_tts_audio(voice, "Hello", send_audio_start=send_audio_start))
    await first_chunk_written.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    event_types = [call.args[0].type for call in enhanced_handler.write_event.call_args_list]
    assert event_types == expected_event_types


@pytest.mark.asyncio
async def test_headerless_wav_response_is_reported_by_the_path_that_saw_it(
    enhanced_handler, mock_info, mock_clients, caplog
):
    """Test raw PCM answering a WAV request warns when it is streamed, and not for every buffered sentence."""
    _, tts_client = mock_clients
    voice = mock_info.tts[0].voices[0]
    pcm = b"\x00\x01" * 40000  # More than is buffered while waiting for a header

    with caplog.at_level(logging.WARNING, logger="wyoming_openai.handler"):
        assert await enhanced_handler._stream_audio_to_wyoming(pcm, True, 0, WAV_AUDIO_FORMAT)
    assert "Could not parse WAV header" not in caplog.text
    assert enhanced_handler.write_event.call_args.args[0].payload == pcm

    _mock_speech_response(tts_client, [pcm])
    with caplog.at_level(logging.WARNING, logger="wyoming_openai.handler"):
        assert await enhanced_handler._stream_tts_audio(voice, "Hello", send_audio_start=True)
    assert caplog.text.count(f"Could not parse WAV header after buffering {len(pcm)} bytes") == 1
