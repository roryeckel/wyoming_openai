"""
Tests to prevent duplicate audio synthesis regression.

The issue: When streaming synthesis is used, both the `synthesize` event
and the `synthesize-stop` event can trigger audio synthesis, leading to
duplicate audio output.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from wyoming.info import Attribution, Info, TtsProgram
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeVoice

from wyoming_openai.compatibility import CustomAsyncOpenAI, TtsVoiceModel
from wyoming_openai.handler import OpenAIEventHandler


class TestDuplicateAudioFix:

    @pytest.fixture
    def mock_tts_client(self):
        """Mock TTS client that tracks how many times synthesis is called."""
        client = AsyncMock(spec=CustomAsyncOpenAI)

        # Mock the streaming response
        mock_response = AsyncMock()
        async def async_iter_bytes(*args, **kwargs):
            for chunk in [b"fake_audio_chunk_1", b"fake_audio_chunk_2", b"fake_audio_chunk_3"]:
                yield chunk
        mock_response.iter_bytes = async_iter_bytes

        # Mock the async context manager properly
        async_context_manager = AsyncMock()
        async_context_manager.__aenter__ = AsyncMock(return_value=mock_response)
        async_context_manager.__aexit__ = AsyncMock(return_value=None)

        # Track synthesis calls
        client.audio.speech.with_streaming_response.create = Mock(return_value=async_context_manager)

        return client

    @pytest.fixture
    def mock_info(self):
        """Mock Wyoming info with TTS voice."""
        voice = TtsVoiceModel(
            model_name="test-model",
            name="test-voice",
            description="Test Voice",
            attribution=Attribution(name="Test", url="http://test.com"),
            installed=True,
            languages=["en"],
            version="1.0"
        )

        tts_program = TtsProgram(
            name="test-tts",
            description="Test TTS",
            attribution=Attribution(name="Test", url="http://test.com"),
            installed=True,
            version="1.0",
            voices=[voice],
            supports_synthesize_streaming=False  # Non-streaming voice
        )

        return Info(tts=[tts_program])

    @pytest.fixture
    def handler(self, mock_info, mock_tts_client):
        """Create handler with mocked dependencies."""
        # Mock reader and writer required by AsyncEventHandler
        reader = AsyncMock()
        writer = AsyncMock()

        handler = OpenAIEventHandler(
            reader,
            writer,
            info=mock_info,
            stt_client=AsyncMock(spec=CustomAsyncOpenAI),
            tts_client=mock_tts_client,
            client_lock=asyncio.Lock()
        )
        handler.write_event = AsyncMock()
        return handler

    @pytest.mark.asyncio
    async def test_no_duplicate_audio_with_streaming_events(self, handler, mock_tts_client):
        """
        Test that when streaming synthesis events are used, audio is NOT synthesized twice.

        This test reproduces the bug scenario:
        1. synthesize-start
        2. synthesize-chunk
        3. synthesize (THIS should not trigger synthesis if streaming is active)
        4. synthesize-stop (This should handle the synthesis)
        """
        text = "Hello! How can I assist you today?"
        voice = SynthesizeVoice(name="test-voice", language=None)

        # Start streaming synthesis
        start_event = SynthesizeStart(voice=voice)
        result = await handler.handle_event(start_event.event())
        assert result is True

        # Add text chunk
        chunk_event = SynthesizeChunk(text=text)
        result = await handler.handle_event(chunk_event.event())
        assert result is True

        # THIS IS THE PROBLEM: A standalone synthesize event during streaming
        synthesize_event = Synthesize(text=text, voice=voice)
        result = await handler.handle_event(synthesize_event.event())

        # Stop streaming synthesis
        stop_event = SynthesizeStop()
        result = await handler.handle_event(stop_event.event())
        assert result is True

        # ASSERTION: TTS synthesis should only be called ONCE, not twice
        synthesis_calls = mock_tts_client.audio.speech.with_streaming_response.create.call_count
        assert synthesis_calls == 1, f"Expected 1 synthesis call, but got {synthesis_calls}. This indicates duplicate audio synthesis!"

    @pytest.mark.asyncio
    async def test_standalone_synthesize_works_normally(self, handler, mock_tts_client):
        """
        Test that standalone synthesize events (without streaming) work normally.
        """
        text = "Hello world"
        voice = SynthesizeVoice(name="test-voice", language=None)

        # Single synthesize event (not part of streaming)
        synthesize_event = Synthesize(text=text, voice=voice)
        result = await handler.handle_event(synthesize_event.event())
        assert result is True

        # Should have exactly 1 synthesis call
        synthesis_calls = mock_tts_client.audio.speech.with_streaming_response.create.call_count
        assert synthesis_calls == 1, f"Expected 1 synthesis call, got {synthesis_calls}"

    @pytest.mark.asyncio
    async def test_streaming_only_synthesis(self, handler, mock_tts_client):
        """
        Test streaming-only synthesis (start/chunk/stop without standalone synthesize).
        """
        text = "Test streaming"
        voice = SynthesizeVoice(name="test-voice", language=None)

        # Streaming synthesis flow
        await handler.handle_event(SynthesizeStart(voice=voice).event())
        await handler.handle_event(SynthesizeChunk(text=text).event())
        await handler.handle_event(SynthesizeStop().event())

        # Should have exactly 1 synthesis call
        synthesis_calls = mock_tts_client.audio.speech.with_streaming_response.create.call_count
        assert synthesis_calls == 1, f"Expected 1 synthesis call, got {synthesis_calls}"
