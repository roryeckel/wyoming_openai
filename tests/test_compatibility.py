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
