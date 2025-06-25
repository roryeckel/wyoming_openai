import asyncio
from unittest.mock import MagicMock

import pytest

from wyoming_openai.handler import OpenAIEventHandler


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
