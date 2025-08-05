import asyncio
import io
import wave
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from wyoming.asr import Transcript
from wyoming.event import Event

from wyoming_openai.handler import (
    OpenAIEventHandler,
)


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
    tts_client = MagicMock()
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
        client_lock=asyncio.Lock(),
    )

@pytest.mark.asyncio
async def test_init_and_stop(dummy_info, dummy_clients, dummy_reader_writer, handler):
    stt_client, tts_client = dummy_clients
    await handler.stop()
    stt_client.close.assert_called_once()
    tts_client.close.assert_called_once()

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
        client_lock=asyncio.Lock(),
        stt_temperature=0.5,
        stt_prompt="Test prompt",
        tts_speed=1.0,
        tts_instructions="Test instructions"
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
        event = Event(
            type="audio-start",
            data={
                "rate": 16000,
                "width": 2,
                "channels": 1
            }
        )

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
        chunk_event = Event(
            type="audio-chunk",
            data={
                "rate": 16000,
                "width": 2,
                "channels": 1
            },
            payload=audio_data
        )

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
        transcribe_event = Event(
            type="transcribe",
            data={
                "language": "en",
                "name": "whisper-1"
            }
        )
        result = await enhanced_handler.handle_event(transcribe_event)
        assert result is True

        # Now record some audio
        start_event = Event(
            type="audio-start",
            data={"rate": 16000, "width": 2, "channels": 1}
        )
        await enhanced_handler.handle_event(start_event)

        # Add audio data
        chunk_event = Event(
            type="audio-chunk",
            data={"rate": 16000, "width": 2, "channels": 1},
            payload=b"\x00\x01" * 1000
        )
        await enhanced_handler.handle_event(chunk_event)

        # Stop recording - this triggers transcription
        # Patch the isinstance check in the handler to accept our mock
        with patch('wyoming_openai.handler.isinstance') as mock_isinstance:
            def isinstance_side_effect(obj, class_or_tuple):
                if obj is mock_transcription:
                    from openai.resources.audio.transcriptions import TranscriptionCreateResponse
                    return class_or_tuple is TranscriptionCreateResponse
                return isinstance.__wrapped__(obj, class_or_tuple)
            mock_isinstance.side_effect = isinstance_side_effect

            stop_event = Event(type="audio-stop")
            await enhanced_handler.handle_event(stop_event)

        # Verify transcription was called
        stt_client.audio.transcriptions.create.assert_called_once()

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
    async def test_handle_synthesize_event(self, enhanced_handler, mock_clients):
        """Test handling of Synthesize event."""
        _, tts_client = mock_clients

        # Create proper WAV data with header
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
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
                self.chunks = [data[i:i+2048] for i in range(0, len(data), 2048)]
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
            type="synthesize",
            data={
                "text": "Hello world",
                "voice": {"name": "alloy"},
                "raw_text": "Hello world"
            }
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
    async def test_handle_transcribe_with_streaming(self, enhanced_handler, mock_clients, mock_info):
        """Test handling of Transcribe event with streaming model."""
        stt_client, _ = mock_clients

        # Make model support streaming
        mock_info.asr[0].supports_transcript_streaming = True

        # For this test, just verify that the streaming path is attempted
        # by checking that create is called with stream=True
        stt_client.audio.transcriptions.create = AsyncMock(side_effect=Exception("Streaming test - expected"))

        # First send the transcribe event to set the model
        transcribe_event = Event(
            type="transcribe",
            data={
                "language": "en",
                "name": "whisper-1"
            }
        )
        result = await enhanced_handler.handle_event(transcribe_event)
        assert result is True

        # Start recording
        start_event = Event(
            type="audio-start",
            data={"rate": 16000, "width": 2, "channels": 1}
        )
        await enhanced_handler.handle_event(start_event)

        # Add some audio
        audio_data = b"\x00\x01" * 100
        chunk_event = Event(
            type="audio-chunk",
            data={"rate": 16000, "width": 2, "channels": 1},
            payload=audio_data
        )
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
        event = Event(
            type="transcribe",
            data={
                "language": "en",
                "name": "invalid-model"
            }
        )

        result = await enhanced_handler.handle_event(event)

        assert result is False

    @pytest.mark.asyncio
    async def test_handle_unsupported_language(self, enhanced_handler):
        """Test handling of Transcribe event with unsupported language."""
        event = Event(
            type="transcribe",
            data={
                "language": "zh",  # Not in supported languages
                "name": "whisper-1"
            }
        )

        result = await enhanced_handler.handle_event(event)

        assert result is False

    @pytest.mark.asyncio
    async def test_audio_recording_workflow(self, enhanced_handler, mock_clients):
        """Test complete audio recording workflow."""
        stt_client, _ = mock_clients

        # Mock transcription response
        mock_transcription = Mock()
        mock_transcription.text = "Recorded audio transcription"
        stt_client.audio.transcriptions.create = AsyncMock(return_value=mock_transcription)

        # First set up transcription model
        transcribe_event = Event(
            type="transcribe",
            data={
                "language": "en",
                "name": "whisper-1"
            }
        )
        await enhanced_handler.handle_event(transcribe_event)

        # Start recording
        start_event = Event(
            type="audio-start",
            data={"rate": 16000, "width": 2, "channels": 1}
        )
        await enhanced_handler.handle_event(start_event)

        assert enhanced_handler._is_recording is True
        assert enhanced_handler._wav_buffer is not None

        # Send multiple audio chunks
        for i in range(5):
            chunk_data = bytes([i % 256] * 200)
            chunk_event = Event(
                type="audio-chunk",
                data={"rate": 16000, "width": 2, "channels": 1},
                payload=chunk_data
            )
            await enhanced_handler.handle_event(chunk_event)

        # Stop recording - this triggers transcription
        with patch('wyoming_openai.handler.isinstance') as mock_isinstance:
            def isinstance_side_effect(obj, class_or_tuple):
                if obj is mock_transcription:
                    from openai.resources.audio.transcriptions import TranscriptionCreateResponse
                    return class_or_tuple is TranscriptionCreateResponse
                return isinstance.__wrapped__(obj, class_or_tuple)
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
