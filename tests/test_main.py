import sys
from unittest.mock import Mock

import pytest

import wyoming_openai.__main__ as main_module
from wyoming_openai.__main__ import main


@pytest.mark.asyncio
async def test_main_rejects_non_object_stt_extra_body_env(monkeypatch, capsys):
    monkeypatch.setenv("STT_EXTRA_BODY", '["not-an-object"]')
    monkeypatch.setattr(sys, "argv", ["wyoming_openai"])

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 2
    assert "Invalid STT extra body: expected a JSON object" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_main_rejects_invalid_stt_strip_regex_env(monkeypatch, capsys):
    monkeypatch.setenv("STT_STRIP_REGEX", "(unclosed")
    monkeypatch.setattr(sys, "argv", ["wyoming_openai"])

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 2
    assert "Invalid STT strip regex" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_main_rejects_empty_stt_strip_regex_cli(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["wyoming_openai", "--stt-strip-regex", ""])

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 2
    assert "Invalid STT strip regex" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_main_rejects_invalid_tts_extra_body_cli(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["wyoming_openai", "--tts-extra-body", '{"stream":'])

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 2
    assert "Invalid TTS extra body" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_main_rejects_invalid_stt_response_format_before_server_start(monkeypatch, capsys):
    async def fake_factory(*args, **kwargs):
        return _FakeClient()

    monkeypatch.setattr(
        main_module.CustomAsyncOpenAI,
        "create_autodetected_factory",
        staticmethod(lambda: fake_factory),
    )
    monkeypatch.setattr(
        main_module.AsyncServer,
        "from_uri",
        staticmethod(lambda uri: _CapturingServer()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wyoming_openai",
            "--stt-models",
            "whisper-1",
            "--stt-extra-body",
            '{"response_format":"text"}',
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 2
    assert "STT extra_body response_format must be one of 'json'" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_main_rejects_non_boolean_stt_stream_override_before_server_start(monkeypatch, capsys):
    async def fake_factory(*args, **kwargs):
        return _FakeClient()

    monkeypatch.setattr(
        main_module.CustomAsyncOpenAI,
        "create_autodetected_factory",
        staticmethod(lambda: fake_factory),
    )
    monkeypatch.setattr(
        main_module.AsyncServer,
        "from_uri",
        staticmethod(lambda uri: _CapturingServer()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wyoming_openai",
            "--stt-models",
            "whisper-1",
            "--stt-extra-body",
            '{"stream":"yes"}',
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 2
    assert "STT extra_body stream must be a boolean" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_main_rejects_tts_transport_override_before_server_start(monkeypatch, capsys):
    async def fake_factory(*args, **kwargs):
        return _FakeClient()

    monkeypatch.setattr(
        main_module.CustomAsyncOpenAI,
        "create_autodetected_factory",
        staticmethod(lambda: fake_factory),
    )
    monkeypatch.setattr(
        main_module.AsyncServer,
        "from_uri",
        staticmethod(lambda uri: _CapturingServer()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wyoming_openai",
            "--tts-models",
            "tts-1",
            "--tts-voices",
            "alloy",
            "--tts-extra-body",
            '{"stream":true}',
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 2
    assert "TTS extra_body does not support overriding 'stream'" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_main_validates_tts_extra_body_before_client_creation(monkeypatch, capsys):
    def unexpected_factory():
        async def should_not_be_called(*args, **kwargs):
            raise AssertionError("client factory should not be created for invalid extra_body")

        return should_not_be_called

    monkeypatch.setattr(
        main_module.CustomAsyncOpenAI,
        "create_autodetected_factory",
        staticmethod(unexpected_factory),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wyoming_openai",
            "--tts-models",
            "tts-1",
            "--tts-voices",
            "alloy",
            "--tts-extra-body",
            '{"stream":true}',
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 2
    assert "TTS extra_body does not support overriding 'stream'" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_main_allows_invalid_unused_tts_extra_body_when_voice_discovery_returns_none(monkeypatch):
    async def fake_factory(*args, **kwargs):
        return _FakeClient()

    monkeypatch.setattr(
        main_module.CustomAsyncOpenAI,
        "create_autodetected_factory",
        staticmethod(lambda: fake_factory),
    )
    monkeypatch.setattr(
        main_module.AsyncServer,
        "from_uri",
        staticmethod(lambda uri: _CapturingServer()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wyoming_openai",
            "--stt-models",
            "whisper-1",
            "--tts-models",
            "tts-1",
            "--tts-extra-body",
            '{"stream":true}',
        ],
    )

    await main()


class _FakeClient:
    def __init__(self):
        self.backend = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def list_supported_voices(self, *args, **kwargs):
        return []


class _CapturingServer:
    def __init__(self):
        self.handlers = []

    async def run(self, handler_factory):
        self.handlers.append(handler_factory(Mock(name="reader"), Mock(name="writer")))


@pytest.mark.asyncio
async def test_main_allows_unused_tts_response_format_in_stt_only_mode(monkeypatch):
    async def fake_factory(*args, **kwargs):
        return _FakeClient()

    for env_var in ("TTS_MODELS", "TTS_STREAMING_MODELS", "TTS_VOICES"):
        monkeypatch.delenv(env_var, raising=False)

    monkeypatch.setattr(
        main_module.CustomAsyncOpenAI,
        "create_autodetected_factory",
        staticmethod(lambda: fake_factory),
    )
    monkeypatch.setattr(
        main_module.AsyncServer,
        "from_uri",
        staticmethod(lambda uri: _CapturingServer()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wyoming_openai",
            "--stt-models",
            "whisper-1",
            "--tts-extra-body",
            '{"response_format":"mp3"}',
        ],
    )

    await main()


@pytest.mark.asyncio
async def test_main_allows_unused_stt_response_format_in_tts_only_mode(monkeypatch):
    async def fake_factory(*args, **kwargs):
        return _FakeClient()

    for env_var in ("STT_MODELS", "STT_STREAMING_MODELS", "STT_REALTIME_MODELS"):
        monkeypatch.delenv(env_var, raising=False)

    monkeypatch.setattr(
        main_module.CustomAsyncOpenAI,
        "create_autodetected_factory",
        staticmethod(lambda: fake_factory),
    )
    monkeypatch.setattr(
        main_module.AsyncServer,
        "from_uri",
        staticmethod(lambda uri: _CapturingServer()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wyoming_openai",
            "--tts-models",
            "tts-1",
            "--tts-voices",
            "alloy",
            "--stt-extra-body",
            '{"response_format":"text"}',
        ],
    )

    await main()


@pytest.mark.asyncio
async def test_main_skips_tts_client_creation_in_stt_only_mode(monkeypatch):
    server = _CapturingServer()
    factory_calls = []

    async def fake_factory(*args, **kwargs):
        factory_calls.append(kwargs)
        return _FakeClient()

    monkeypatch.setattr(
        main_module.CustomAsyncOpenAI,
        "create_autodetected_factory",
        staticmethod(lambda: fake_factory),
    )
    monkeypatch.setattr(
        main_module.AsyncServer,
        "from_uri",
        staticmethod(lambda uri: server),
    )
    for env_var in ("TTS_MODELS", "TTS_STREAMING_MODELS", "TTS_VOICES"):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wyoming_openai",
            "--stt-models",
            "whisper-1",
        ],
    )

    await main()

    assert len(factory_calls) == 1
    assert len(server.handlers) == 1
    assert server.handlers[0]._stt_client is not None
    assert server.handlers[0]._tts_client is None


@pytest.mark.asyncio
async def test_main_skips_stt_client_creation_in_tts_only_mode(monkeypatch):
    server = _CapturingServer()
    factory_calls = []

    async def fake_factory(*args, **kwargs):
        factory_calls.append(kwargs)
        return _FakeClient()

    monkeypatch.setattr(
        main_module.CustomAsyncOpenAI,
        "create_autodetected_factory",
        staticmethod(lambda: fake_factory),
    )
    monkeypatch.setattr(
        main_module.AsyncServer,
        "from_uri",
        staticmethod(lambda uri: server),
    )
    for env_var in ("STT_MODELS", "STT_STREAMING_MODELS", "STT_REALTIME_MODELS"):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wyoming_openai",
            "--tts-models",
            "tts-1",
            "--tts-voices",
            "alloy",
        ],
    )

    await main()

    assert len(factory_calls) == 1
    assert len(server.handlers) == 1
    assert server.handlers[0]._stt_client is None
    assert server.handlers[0]._tts_client is not None


@pytest.mark.asyncio
async def test_main_configures_realtime_stt_models(monkeypatch):
    server = _CapturingServer()

    async def fake_factory(*args, **kwargs):
        return _FakeClient()

    monkeypatch.setattr(
        main_module.CustomAsyncOpenAI,
        "create_autodetected_factory",
        staticmethod(lambda: fake_factory),
    )
    monkeypatch.setattr(
        main_module.AsyncServer,
        "from_uri",
        staticmethod(lambda uri: server),
    )
    for env_var in ("TTS_MODELS", "TTS_STREAMING_MODELS", "TTS_VOICES"):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wyoming_openai",
            "--stt-realtime-models",
            "gpt-realtime-whisper",
        ],
    )

    await main()

    assert len(server.handlers) == 1
    handler = server.handlers[0]
    assert handler._stt_client is not None
    assert handler._tts_client is None
    assert handler._stt_realtime_models == {"gpt-realtime-whisper"}
