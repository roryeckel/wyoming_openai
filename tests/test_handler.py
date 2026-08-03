import asyncio
import base64
import builtins
import io
import struct
import wave
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from openai import omit
from wyoming.asr import Transcript, TranscriptChunk
from wyoming.event import Event
from wyoming.info import Attribution, TtsVoice
from wyoming.tts import SynthesizeChunk, SynthesizeStart, SynthesizeVoice

from wyoming_openai.compatibility import OpenAIBackend
from wyoming_openai.handler import (
    OpenAIEventHandler,
    TtsStreamError,
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


class _FakeRealtimeConnection:
    def __init__(self, commit_events):
        self.commit_events = commit_events
        self.events = asyncio.Queue()
        self.session = _FakeRealtimeSession()
        self.input_audio_buffer = _FakeRealtimeInputAudioBuffer(self)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        event = await self.events.get()
        if event is None:
            raise StopAsyncIteration
        return event

    async def close(self):
        self.closed = True
        await self.events.put(None)


class _FakeRealtimeConnectionManager:
    def __init__(self, connection):
        self.connection = connection
        self.exited = False

    async def enter(self):
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
    assert handler._synthesis_voice is None


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
        )

        audio_chunk_events = [
            call.args[0] for call in enhanced_handler.write_event.call_args_list if call.args[0].type == "audio-chunk"
        ]
        assert b"".join(event.payload for event in audio_chunk_events) == pcm[:-10]
        assert timestamp == pytest.approx(470 / 2 / 24000 * 1000)
        assert "ended after 470 of 480 declared PCM bytes" in caplog.text

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
