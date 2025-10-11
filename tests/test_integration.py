import io
import wave
from unittest.mock import AsyncMock, Mock, patch

import pytest
from wyoming.event import Event

from wyoming_openai.compatibility import (
    CustomAsyncOpenAI,
    OpenAIBackend,
    TtsVoiceModel,
    create_asr_programs,
    create_info,
    create_tts_programs,
    create_tts_voices,
)
from wyoming_openai.handler import OpenAIEventHandler


def create_test_wav_data(frame_count: int = 2000, sample_rate: int = 24000) -> bytes:
    """Create test WAV data with specified parameters."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x01" * frame_count)
    wav_buffer.seek(0)
    return wav_buffer.read()


class TestIntegration:
    """Integration tests that test component interactions."""

    @pytest.mark.asyncio
    async def test_end_to_end_transcription_workflow(self):
        """Test complete transcription workflow from audio to text."""
        # Create real-like info structure
        asr_models = ["whisper-1"]
        streaming_models = []
        languages = ["en", "es"]
        base_url = "https://api.openai.com/v1"

        asr_programs = create_asr_programs(asr_models, streaming_models, base_url, languages)
        tts_programs = create_tts_programs([])  # No TTS for this test
        info = create_info(asr_programs, tts_programs)

        # Create mock clients
        stt_client = AsyncMock()
        tts_client = AsyncMock()

        # Mock transcription response
        mock_transcription = Mock()
        mock_transcription.text = "Integration test transcription"
        stt_client.audio.transcriptions.create = AsyncMock(return_value=mock_transcription)

        # Create handler
        reader = AsyncMock()
        writer = AsyncMock()
        handler = OpenAIEventHandler(
            reader, writer, info=info, stt_client=stt_client, tts_client=tts_client,
        )
        handler.write_event = AsyncMock()

        # Execute workflow
        # 1. Set transcription model
        transcribe_event = Event(type="transcribe", data={"language": "en", "name": "whisper-1"})
        result = await handler.handle_event(transcribe_event)
        assert result is True

        # 2. Start audio recording
        start_event = Event(type="audio-start", data={"rate": 16000, "width": 2, "channels": 1})
        await handler.handle_event(start_event)
        assert handler._is_recording is True

        # 3. Send audio data
        test_audio_data = b"\x00\x01" * 500
        chunk_event = Event(
            type="audio-chunk",
            data={"rate": 16000, "width": 2, "channels": 1},
            payload=test_audio_data
        )
        await handler.handle_event(chunk_event)

        # 4. Stop recording and trigger transcription
        with patch('wyoming_openai.handler.isinstance') as mock_isinstance:
            def isinstance_side_effect(obj, class_or_tuple):
                if obj is mock_transcription:
                    from openai.resources.audio.transcriptions import TranscriptionCreateResponse
                    return class_or_tuple is TranscriptionCreateResponse
                return isinstance.__wrapped__(obj, class_or_tuple)
            mock_isinstance.side_effect = isinstance_side_effect

            stop_event = Event(type="audio-stop")
            await handler.handle_event(stop_event)

        # Verify the complete workflow
        assert handler._is_recording is False
        stt_client.audio.transcriptions.create.assert_called_once()

        # Verify transcript event was written
        transcript_events = [
            call[0][0] for call in handler.write_event.call_args_list
            if call[0][0].type == "transcript"
        ]
        assert len(transcript_events) > 0

    @pytest.mark.asyncio
    async def test_end_to_end_synthesis_workflow(self):
        """Test complete synthesis workflow from text to audio."""
        # Create real-like info structure
        tts_models = ["tts-1"]
        tts_voice_names = ["alloy", "echo"]
        languages = ["en"]
        base_url = "https://api.openai.com/v1"

        tts_voices = create_tts_voices(tts_models, [], tts_voice_names, base_url, languages)
        tts_programs = create_tts_programs(tts_voices)
        asr_programs = create_asr_programs([], [], base_url, languages)  # No ASR for this test
        info = create_info(asr_programs, tts_programs)

        # Create mock clients
        stt_client = AsyncMock()
        tts_client = AsyncMock()

        # Create proper WAV data for TTS response
        mock_audio_data = create_test_wav_data(2000)

        # Mock streaming response
        class MockAsyncIterator:
            def __init__(self, data):
                self.chunks = [data[i:i+1024] for i in range(0, len(data), 1024)]
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

        # Create handler
        reader = AsyncMock()
        writer = AsyncMock()
        handler = OpenAIEventHandler(
            reader, writer, info=info, stt_client=stt_client, tts_client=tts_client,
        )
        handler.write_event = AsyncMock()

        # Execute synthesis workflow
        synthesize_event = Event(
            type="synthesize",
            data={
                "text": "Integration test synthesis",
                "voice": {"name": "alloy"},
                "raw_text": "Integration test synthesis"
            }
        )

        result = await handler.handle_event(synthesize_event)

        # Verify the complete workflow
        assert result is True
        tts_client.audio.speech.with_streaming_response.create.assert_called_once()

        # Verify audio events were written
        event_types = [call[0][0].type for call in handler.write_event.call_args_list]
        assert "audio-start" in event_types
        assert "audio-stop" in event_types
        assert "audio-chunk" in event_types

    @pytest.mark.asyncio
    async def test_backend_autodetection_integration(self):
        """Test backend autodetection with different base URLs."""
        test_cases = [
            ("https://api.openai.com/v1", OpenAIBackend.OPENAI),
            ("http://localhost:8080/v1", OpenAIBackend.LOCALAI),
            ("https://api.speaches.org/v1/", OpenAIBackend.SPEACHES),
        ]

        factory = CustomAsyncOpenAI.create_autodetected_factory()

        for base_url, expected_backend in test_cases:
            # Mock the detection methods based on expected backend
            with patch.object(CustomAsyncOpenAI, '_is_localai', return_value=(expected_backend == OpenAIBackend.LOCALAI)):
                with patch.object(CustomAsyncOpenAI, '_is_speaches', return_value=(expected_backend == OpenAIBackend.SPEACHES)):
                    with patch.object(CustomAsyncOpenAI, '_is_kokoro_fastapi', return_value=(expected_backend == OpenAIBackend.KOKORO_FASTAPI)):
                        client = await factory(api_key="test-key", base_url=base_url)
                        assert client.backend == expected_backend

    @pytest.mark.asyncio
    async def test_multi_event_workflow(self):
        """Test handling multiple events in sequence."""
        # Setup comprehensive info
        asr_programs = create_asr_programs(["whisper-1"], [], "https://api.openai.com/v1", ["en"])
        tts_voices = create_tts_voices(["tts-1"], [], ["alloy"], "https://api.openai.com/v1", ["en"])
        tts_programs = create_tts_programs(tts_voices)
        info = create_info(asr_programs, tts_programs)

        # Create mock clients
        stt_client = AsyncMock()
        tts_client = AsyncMock()

        # Create handler
        reader = AsyncMock()
        writer = AsyncMock()
        handler = OpenAIEventHandler(
            reader, writer, info=info, stt_client=stt_client, tts_client=tts_client,
        )
        handler.write_event = AsyncMock()

        # Test sequence of events
        events = [
            Event(type="describe"),
            Event(type="transcribe", data={"language": "en", "name": "whisper-1"}),
            Event(type="synthesize", data={"text": "Hello", "voice": {"name": "alloy"}, "raw_text": "Hello"}),
        ]

        # Mock TTS response for synthesis
        test_wav_data = create_test_wav_data(100)

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
        mock_response.iter_bytes = Mock(return_value=MockAsyncIterator(test_wav_data))

        mock_stream_response = AsyncMock()
        mock_stream_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_response.__aexit__ = AsyncMock(return_value=None)

        tts_client.audio.speech.with_streaming_response.create = Mock(return_value=mock_stream_response)

        # Execute all events
        results = []
        expected_results = [True, True, True]  # describe returns True, transcribe returns True, synthesize returns True
        for event, expected in zip(events, expected_results, strict=False):
            result = await handler.handle_event(event)
            results.append(result)
            assert result == expected

        # Verify info event was written for describe
        info_events = [
            call[0][0] for call in handler.write_event.call_args_list
            if call[0][0].type == "info"
        ]
        assert len(info_events) > 0

        # Verify TTS was called for synthesis
        tts_client.audio.speech.with_streaming_response.create.assert_called_once()

    def test_info_structure_integration(self):
        """Test that the info structure is properly constructed with all components."""
        # Create comprehensive configuration
        asr_models = ["whisper-1", "whisper-large"]
        streaming_models = ["whisper-large"]
        tts_models = ["tts-1", "tts-1-hd"]
        tts_voices = ["alloy", "echo", "fable"]
        languages = ["en", "es", "fr"]
        base_url = "https://api.openai.com/v1"

        # Build info structure
        asr_programs = create_asr_programs(asr_models, streaming_models, base_url, languages)
        tts_voice_models = create_tts_voices(tts_models, [], tts_voices, base_url, languages)
        tts_programs = create_tts_programs(tts_voice_models)
        info = create_info(asr_programs, tts_programs)

        # Verify structure integrity
        assert len(info.asr) == 2  # One streaming, one non-streaming
        assert len(info.tts) == 1  # One TTS program

        # Verify ASR programs
        streaming_program = next(p for p in info.asr if p.supports_transcript_streaming)
        non_streaming_program = next(p for p in info.asr if not p.supports_transcript_streaming)

        assert len(streaming_program.models) == 1
        assert streaming_program.models[0].name == "whisper-large"
        assert len(non_streaming_program.models) == 1
        assert non_streaming_program.models[0].name == "whisper-1"

        # Verify TTS program
        tts_program = info.tts[0]
        assert len(tts_program.voices) == 6  # 2 models * 3 voices

        # Verify all voices have correct attributes
        for voice in tts_program.voices:
            assert isinstance(voice, TtsVoiceModel)
            assert voice.name in tts_voices
            assert voice.model_name in tts_models
            assert voice.languages == languages

    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """Test error handling across components."""
        # Create minimal setup
        info = create_info([], [])  # No models/voices

        stt_client = AsyncMock()
        tts_client = AsyncMock()

        reader = AsyncMock()
        writer = AsyncMock()
        handler = OpenAIEventHandler(
            reader, writer, info=info, stt_client=stt_client, tts_client=tts_client,
        )
        handler.write_event = AsyncMock()

        # Test handling events with no available models
        invalid_events = [
            Event(type="transcribe", data={"language": "en", "name": "nonexistent-model"}),
            Event(type="synthesize", data={"text": "Hello", "voice": {"name": "nonexistent-voice"}}),
        ]

        for event in invalid_events:
            result = await handler.handle_event(event)
            assert result is False  # Should fail gracefully

        # Verify no API calls were made
        stt_client.audio.transcriptions.create.assert_not_called()
        tts_client.audio.speech.with_streaming_response.create.assert_not_called()
