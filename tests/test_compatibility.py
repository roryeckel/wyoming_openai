from unittest.mock import AsyncMock, Mock, patch

import pytest
from wyoming.info import AsrModel, AsrProgram, Attribution, Info, TtsProgram, TtsVoice

from wyoming_openai.compatibility import (
    CustomAsyncOpenAI,
    OpenAIBackend,
    TtsVoiceModel,
    asr_model_to_string,
    create_asr_programs,
    create_info,
    create_tts_programs,
    create_tts_voices,
    tts_voice_to_string,
)
from wyoming_openai.const import (
    ATTRIBUTION_NAME_PROGRAM,
    ATTRIBUTION_NAME_PROGRAM_STREAMING,
)


def test_tts_voice_model_inherits_ttsvoice():
    v = TtsVoiceModel("model-x", name="voice1", description="desc", attribution=Attribution(name="n", url="u"), installed=True, languages=["en"], version="1.0")
    assert isinstance(v, TtsVoice)
    assert v.model_name == "model-x"

def test_create_asr_programs():
    progs = create_asr_programs(["m1"], ["m2"], "url", ["en"])
    assert isinstance(progs, list)
    assert all(isinstance(p, AsrProgram) for p in progs)

def test_create_tts_voices():
    voices = create_tts_voices(["m"], ["v"], "url", ["en"])
    assert isinstance(voices, list)
    assert all(isinstance(v, TtsVoiceModel) for v in voices)

def test_create_tts_programs():
    voices = create_tts_voices(["m"], ["v"], "url", ["en"])
    progs = create_tts_programs(voices)
    assert isinstance(progs, list)
    assert all(isinstance(p, TtsProgram) for p in progs)

def test_create_info():
    asr = create_asr_programs(["m1"], ["m2"], "url", ["en"])
    tts = create_tts_programs(create_tts_voices(["m"], ["v"], "url", ["en"]))
    info = create_info(asr, tts)
    assert isinstance(info, Info)

def test_asr_model_to_string_and_tts_voice_to_string():
    asr = AsrModel(name="n", description="d", attribution=Attribution(name="n", url="u"), installed=True, languages=["en"], version="1.0")
    tts = TtsVoiceModel("model-x", name="voice1", description="desc", attribution=Attribution(name="n", url="u"), installed=True, languages=["en"], version="1.0")
    assert isinstance(asr_model_to_string(asr, True), str)
    assert isinstance(tts_voice_to_string(tts), str)

def test_openai_backend_enum():
    assert OpenAIBackend.OPENAI.name == "OPENAI"
    assert isinstance(OpenAIBackend.SPEACHES.value, int)

def test_custom_async_openai_init_sets_backend(monkeypatch):
    # Patch AsyncOpenAI to avoid real network
    class DummyAsyncOpenAI:
        def __init__(self, *args, **kwargs):
            pass
    monkeypatch.setattr("wyoming_openai.compatibility.AsyncOpenAI", DummyAsyncOpenAI)
    c = CustomAsyncOpenAI(backend=OpenAIBackend.SPEACHES)
    assert c.backend == OpenAIBackend.SPEACHES


class TestOpenAIBackend:
    """Test OpenAIBackend enum."""

    def test_backend_values(self):
        """Test that all backend values are defined."""
        assert OpenAIBackend.OPENAI.value == 0
        assert OpenAIBackend.SPEACHES.value == 1
        assert OpenAIBackend.KOKORO_FASTAPI.value == 2
        assert OpenAIBackend.LOCALAI.value == 3
        # Check if VOXTRAL exists
        if hasattr(OpenAIBackend, 'VOXTRAL'):
            assert OpenAIBackend.VOXTRAL.value == 4


class TestCustomAsyncOpenAI:
    """Test CustomAsyncOpenAI class."""

    @pytest.mark.asyncio
    async def test_create_backend_factory_openai(self):
        """Test creating OpenAI backend factory."""
        factory = CustomAsyncOpenAI.create_backend_factory(OpenAIBackend.OPENAI)

        client = await factory(api_key="test-key", base_url="https://api.openai.com")

        assert isinstance(client, CustomAsyncOpenAI)
        assert client.backend == OpenAIBackend.OPENAI
        assert client.api_key == "test-key"
        assert client.base_url == "https://api.openai.com"

    @pytest.mark.asyncio
    async def test_create_backend_factory_speaches(self):
        """Test creating Speaches backend factory."""
        factory = CustomAsyncOpenAI.create_backend_factory(OpenAIBackend.SPEACHES)

        client = await factory(api_key="test-key", base_url="https://api.speaches.com")

        assert isinstance(client, CustomAsyncOpenAI)
        assert client.backend == OpenAIBackend.SPEACHES

    @pytest.mark.asyncio
    async def test_create_autodetected_factory(self):
        """Test creating autodetected backend factory."""
        factory = CustomAsyncOpenAI.create_autodetected_factory()

        # Mock the detection methods
        with patch.object(CustomAsyncOpenAI, '_is_localai', return_value=False):
            with patch.object(CustomAsyncOpenAI, '_is_speaches', return_value=False):
                with patch.object(CustomAsyncOpenAI, '_is_kokoro_fastapi', return_value=False):
                    # Test OpenAI detection (default)
                    client = await factory(api_key="test-key", base_url="https://api.openai.com/v1")
                    assert client.backend == OpenAIBackend.OPENAI

        # Test Speaches detection
        with patch.object(CustomAsyncOpenAI, '_is_localai', return_value=False):
            with patch.object(CustomAsyncOpenAI, '_is_speaches', return_value=True):
                with patch.object(CustomAsyncOpenAI, '_is_kokoro_fastapi', return_value=False):
                    client = await factory(api_key="test-key", base_url="https://api.speaches.com")
                    assert client.backend == OpenAIBackend.SPEACHES

        # Test LocalAI detection
        with patch.object(CustomAsyncOpenAI, '_is_localai', return_value=True):
            with patch.object(CustomAsyncOpenAI, '_is_speaches', return_value=False):
                with patch.object(CustomAsyncOpenAI, '_is_kokoro_fastapi', return_value=False):
                    client = await factory(api_key="test-key", base_url="http://localhost:8080")
                    assert client.backend == OpenAIBackend.LOCALAI

    @pytest.mark.asyncio
    async def test_list_supported_voices_openai(self):
        """Test listing supported voices for OpenAI backend."""
        custom_client = CustomAsyncOpenAI(api_key="test-key", backend=OpenAIBackend.OPENAI)

        voices = await custom_client.list_supported_voices(["tts-1", "tts-1-hd"], ["en", "fr"])

        # Should return default OpenAI voices
        assert len(voices) == 18  # 9 voices * 2 models
        assert all(isinstance(v, TtsVoiceModel) for v in voices)
        assert all(v.name in ["alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"] for v in voices)

    @pytest.mark.asyncio
    async def test_list_supported_voices_speaches(self):
        """Test listing supported voices for Speaches backend."""
        custom_client = CustomAsyncOpenAI(api_key="test-key", backend=OpenAIBackend.SPEACHES)

        # Mock the _list_speaches_voices method
        with patch.object(custom_client, '_list_speaches_voices', AsyncMock(return_value=["voice1", "voice2"])):
            voices = await custom_client.list_supported_voices(["tts-1"], ["en"])

            assert len(voices) == 2
            assert voices[0].name == "voice1"
            assert voices[1].name == "voice2"

    @pytest.mark.asyncio
    async def test_list_supported_voices_localai(self):
        """Test listing supported voices for LocalAI backend."""
        custom_client = CustomAsyncOpenAI(api_key="test-key", backend=OpenAIBackend.LOCALAI)

        # Mock the _list_localai_voices method
        with patch.object(custom_client, '_list_localai_voices', AsyncMock(return_value=["tts-model"])):
            voices = await custom_client.list_supported_voices(["tts-1"], ["en"])

            assert len(voices) == 1
            assert voices[0].name == "tts-model"
            assert voices[0].model_name == "tts-1"  # Model name comes from the input


class TestHelperFunctions:
    """Test helper functions with enhanced coverage."""

    def test_asr_model_to_string_detailed(self):
        """Test converting ASR model to string with full details."""
        model = AsrModel(
            name="whisper-1",
            description="OpenAI Whisper",
            attribution=Attribution(name="OpenAI", url="https://openai.com"),
            installed=True,
            version="1.0",
            languages=["en", "fr"]
        )

        result = asr_model_to_string(model, is_streaming=True)

        assert "whisper-1" in result
        assert "OpenAI Whisper" in result
        assert "OpenAI" in result
        assert "https://openai.com" in result
        assert "en, fr" in result
        assert "Yes" in result  # Streaming: Yes

    def test_tts_voice_to_string_detailed(self):
        """Test converting TTS voice to string with full details."""
        voice = TtsVoiceModel(
            name="alloy",
            description="Alloy voice",
            attribution=Attribution(name="OpenAI", url="https://openai.com"),
            installed=True,
            version="1.0",
            languages=["en"],
            model_name="tts-1"
        )

        result = tts_voice_to_string(voice)

        assert "alloy" in result
        assert "Alloy voice" in result
        assert "OpenAI" in result
        assert "https://openai.com" in result
        assert "en" in result
        assert "tts-1" in result

    def test_create_asr_programs_detailed(self):
        """Test creating ASR programs with streaming and non-streaming models."""
        models = ["whisper-1", "whisper-2"]
        streaming_models = ["whisper-2"]
        base_url = "https://api.openai.com"
        languages = ["en", "fr"]

        programs = create_asr_programs(models, streaming_models, base_url, languages)

        assert len(programs) == 2

        # Find streaming and non-streaming programs
        streaming_prog = next(p for p in programs if p.supports_transcript_streaming)
        non_streaming_prog = next(p for p in programs if not p.supports_transcript_streaming)

        # Check streaming program
        assert len(streaming_prog.models) == 1
        assert streaming_prog.models[0].name == "whisper-2"
        assert streaming_prog.supports_transcript_streaming
        assert streaming_prog.attribution.name == ATTRIBUTION_NAME_PROGRAM_STREAMING

        # Check non-streaming program
        assert len(non_streaming_prog.models) == 1
        assert non_streaming_prog.models[0].name == "whisper-1"
        assert not non_streaming_prog.supports_transcript_streaming
        assert non_streaming_prog.attribution.name == ATTRIBUTION_NAME_PROGRAM

    def test_create_asr_programs_empty(self):
        """Test creating ASR programs with empty models."""
        programs = create_asr_programs([], [], "https://api.openai.com", ["en"])
        assert programs == []

    def test_create_tts_voices_detailed(self):
        """Test creating TTS voices with multiple models and voices."""
        models = ["tts-1", "tts-1-hd"]
        voices = ["alloy", "echo"]
        base_url = "https://api.openai.com"
        languages = ["en", "fr"]

        result = create_tts_voices(models, voices, base_url, languages)

        assert len(result) == 4  # 2 models * 2 voices
        assert all(isinstance(v, TtsVoiceModel) for v in result)

        # Check first voice
        first_voice = result[0]
        assert first_voice.name == "alloy"
        assert first_voice.model_name == "tts-1"
        assert first_voice.languages == ["en", "fr"]

    def test_create_tts_programs_detailed(self):
        """Test creating TTS programs with actual voice models."""
        voices = [
            TtsVoiceModel(
                name="alloy",
                description="Alloy voice",
                attribution=Attribution(name="OpenAI", url="https://openai.com"),
                installed=True,
                version="1.0",
                languages=["en"],
                model_name="tts-1"
            )
        ]

        programs = create_tts_programs(voices)

        assert len(programs) == 1
        assert programs[0].name == "openai"
        assert programs[0].voices == voices
        assert programs[0].attribution.name == ATTRIBUTION_NAME_PROGRAM

    def test_create_tts_programs_empty(self):
        """Test creating TTS programs with empty voices."""
        programs = create_tts_programs([])
        assert programs == []

    def test_create_info_detailed(self):
        """Test creating Info object with actual programs."""
        asr_programs = [Mock(spec=AsrProgram)]
        tts_programs = [Mock(spec=TtsProgram)]

        info = create_info(asr_programs, tts_programs)

        assert info.asr == asr_programs
        assert info.tts == tts_programs


class TestBackendSpecificBehavior:
    """Test backend-specific behavior."""

    @pytest.mark.asyncio
    async def test_kokoro_fastapi_voices(self):
        """Test Kokoro FastAPI voice listing."""
        custom_client = CustomAsyncOpenAI(api_key="test-key", backend=OpenAIBackend.KOKORO_FASTAPI)

        # Mock the Kokoro-specific voice listing
        with patch.object(custom_client, '_list_kokoro_fastapi_voices', AsyncMock(return_value=["af_sky", "bf_emma"])):
            voices = await custom_client.list_supported_voices(["kokoro-v0_19"], ["en", "ja"])

            # Should return Kokoro-specific voices
            assert len(voices) == 2
            assert all(v.model_name == "kokoro-v0_19" for v in voices)
            assert all(set(v.languages) == {"en", "ja"} for v in voices)
            assert voices[0].name == "af_sky"
            assert voices[1].name == "bf_emma"
