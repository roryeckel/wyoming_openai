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
        self.backend: main_module.OpenAIBackend | None = None
        self.is_official_openai = False

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


@pytest.mark.asyncio
async def test_main_treats_tts_realtime_models_as_tts_models(monkeypatch):
    server = _CapturingServer()
    listed = {}

    class _VoiceListingClient(_FakeClient):
        async def list_supported_voices(self, *args, **kwargs):
            listed["args"] = args
            listed["kwargs"] = kwargs
            return []

    async def fake_factory(*args, **kwargs):
        return _VoiceListingClient()

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
    for env_var in ("STT_MODELS", "STT_STREAMING_MODELS", "STT_REALTIME_MODELS", "TTS_MODELS", "TTS_VOICES"):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wyoming_openai",
            "--tts-realtime-models",
            "gpt-realtime-2.1-mini",
        ],
    )

    await main()

    assert listed["args"][0] == ["gpt-realtime-2.1-mini"]
    assert listed["kwargs"]["realtime_model_names"] == ["gpt-realtime-2.1-mini"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "official", "expected_voices", "expect_backend_warning"),
    [
        (main_module.OpenAIBackend.OPENAI, True, ["alloy"], False),
        # Unrecognized compatible servers are autodetected as OPENAI without being the official API
        (main_module.OpenAIBackend.OPENAI, False, ["alloy", "fable"], True),
        (main_module.OpenAIBackend.SPEACHES, False, ["alloy", "fable"], True),
    ],
)
async def test_main_realtime_tts_voice_filter_and_warnings(
    monkeypatch, caplog, backend, official, expected_voices, expect_backend_warning
):
    server = _CapturingServer()

    async def fake_factory(*args, **kwargs):
        client = _FakeClient()
        client.backend = backend
        client.is_official_openai = official
        return client

    monkeypatch.setattr(
        main_module.CustomAsyncOpenAI,
        "create_autodetected_factory",
        staticmethod(lambda: fake_factory),
    )
    monkeypatch.setattr(main_module.AsyncServer, "from_uri", staticmethod(lambda uri: server))
    monkeypatch.setattr(main_module, "configure_logging", lambda *args, **kwargs: None)
    for env_var in (
        "STT_MODELS",
        "STT_STREAMING_MODELS",
        "STT_REALTIME_MODELS",
        "TTS_MODELS",
        "TTS_STREAMING_MODELS",
        "TTS_VOICES",
        "TTS_BACKEND",
        "TTS_EXTRA_BODY",
        "TTS_REALTIME_EXTRA_BODY",
    ):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wyoming_openai",
            "--tts-realtime-models",
            "gpt-realtime-2.1-mini",
            "--tts-voices",
            "alloy",
            "fable",
            "--tts-speed",
            "3.0",
            "--tts-extra-body",
            '{"lang_code":"en"}',
            "--tts-realtime-extra-body",
            '{"reasoning":{"effort":"low"}}',
        ],
    )

    with caplog.at_level("WARNING"):
        await main()

    handler = server.handlers[0]
    assert [voice.name for voice in handler._wyoming_info.tts[0].voices] == expected_voices
    assert handler._tts_extra_body == {"lang_code": "en"}
    assert handler._tts_realtime_extra_body == {"reasoning": {"effort": "low"}}
    assert "Realtime TTS speed must be between" in caplog.text
    assert ("must implement /v1/realtime" in caplog.text) is expect_backend_warning


@pytest.mark.asyncio
async def test_main_rejects_incompatible_realtime_extra_body(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wyoming_openai",
            "--tts-realtime-models",
            "gpt-realtime-2.1-mini",
            "--tts-realtime-extra-body",
            '{"audio":{"output":{"format":{"type":"audio/pcmu"}}}}',
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 2
    assert "audio.output.format" in capsys.readouterr().err


_MAIN_ENV_VARS = (
    "STT_MODELS",
    "STT_STREAMING_MODELS",
    "STT_REALTIME_MODELS",
    "STT_EXTRA_BODY",
    "STT_REALTIME_EXTRA_BODY",
    "TTS_MODELS",
    "TTS_STREAMING_MODELS",
    "TTS_REALTIME_MODELS",
    "TTS_VOICES",
    "TTS_BACKEND",
    "TTS_EXTRA_BODY",
    "TTS_REALTIME_EXTRA_BODY",
)


@pytest.mark.asyncio
async def test_main_rejects_realtime_tts_model_left_without_voices(monkeypatch, capsys):
    async def fake_factory(*args, **kwargs):
        client = _FakeClient()
        client.backend = main_module.OpenAIBackend.OPENAI
        client.is_official_openai = True
        return client

    monkeypatch.setattr(
        main_module.CustomAsyncOpenAI,
        "create_autodetected_factory",
        staticmethod(lambda: fake_factory),
    )
    monkeypatch.setattr(main_module, "configure_logging", lambda *args, **kwargs: None)
    for env_var in _MAIN_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["wyoming_openai", "--tts-realtime-models", "gpt-realtime-2.1-mini", "--tts-voices", "onyx", "nova"],
    )

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "onyx" in error
    assert "marin" in error


@pytest.mark.asyncio
async def test_main_passes_stt_realtime_extra_body(monkeypatch):
    server = _CapturingServer()

    async def fake_factory(*args, **kwargs):
        return _FakeClient()

    monkeypatch.setattr(
        main_module.CustomAsyncOpenAI,
        "create_autodetected_factory",
        staticmethod(lambda: fake_factory),
    )
    monkeypatch.setattr(main_module.AsyncServer, "from_uri", staticmethod(lambda uri: server))
    monkeypatch.setattr(main_module, "configure_logging", lambda *args, **kwargs: None)
    for env_var in _MAIN_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("STT_REALTIME_EXTRA_BODY", '{"keywords":["Wyoming"]}')
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wyoming_openai",
            "--stt-realtime-models",
            "gpt-live-transcribe",
            "--stt-extra-body",
            '{"temperature":0.2}',
        ],
    )

    await main()

    handler = server.handlers[0]
    assert handler._stt_extra_body == {"temperature": 0.2}
    assert handler._stt_realtime_extra_body == {"keywords": ["Wyoming"]}


async def _run_main_and_capture_handler_kwargs(monkeypatch, *cli_args):
    """Run main() against fake backends and return the keyword arguments the handler factory was configured with."""
    created = []

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
    monkeypatch.setattr(main_module, "OpenAIEventHandler", lambda *args, **kwargs: created.append(kwargs))
    for env_var in ("TTS_MODELS", "TTS_STREAMING_MODELS", "TTS_VOICES"):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(sys, "argv", ["wyoming_openai", "--stt-models", "whisper-1", *cli_args])

    await main()

    assert len(created) == 1
    return created[0]


@pytest.mark.asyncio
async def test_main_defaults_tts_concurrent_requests_to_three(monkeypatch):
    monkeypatch.delenv("TTS_CONCURRENT_REQUESTS", raising=False)

    handler_kwargs = await _run_main_and_capture_handler_kwargs(monkeypatch)

    assert handler_kwargs["tts_concurrent_requests"] == 3


@pytest.mark.asyncio
async def test_main_treats_empty_tts_concurrent_requests_env_as_unset(monkeypatch):
    monkeypatch.setenv("TTS_CONCURRENT_REQUESTS", "")

    handler_kwargs = await _run_main_and_capture_handler_kwargs(monkeypatch)

    assert handler_kwargs["tts_concurrent_requests"] == 3


@pytest.mark.asyncio
async def test_main_reads_tts_concurrent_requests_from_env(monkeypatch):
    monkeypatch.setenv("TTS_CONCURRENT_REQUESTS", "2")

    handler_kwargs = await _run_main_and_capture_handler_kwargs(monkeypatch)

    assert handler_kwargs["tts_concurrent_requests"] == 2


@pytest.mark.asyncio
async def test_main_cli_tts_concurrent_requests_overrides_env(monkeypatch):
    monkeypatch.setenv("TTS_CONCURRENT_REQUESTS", "2")

    handler_kwargs = await _run_main_and_capture_handler_kwargs(monkeypatch, "--tts-concurrent-requests", "4")

    assert handler_kwargs["tts_concurrent_requests"] == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["0", "-1", "two", "1.5"])
async def test_main_rejects_invalid_tts_concurrent_requests_cli(monkeypatch, capsys, value):
    monkeypatch.delenv("TTS_CONCURRENT_REQUESTS", raising=False)
    monkeypatch.setattr(sys, "argv", ["wyoming_openai", "--tts-concurrent-requests", value])

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 2
    assert "--tts-concurrent-requests: expected a whole number of at least 1" in capsys.readouterr().err


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["0", "two"])
async def test_main_rejects_invalid_tts_concurrent_requests_env(monkeypatch, capsys, value):
    monkeypatch.setenv("TTS_CONCURRENT_REQUESTS", value)
    monkeypatch.setattr(sys, "argv", ["wyoming_openai"])

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 2
    assert "--tts-concurrent-requests: expected a whole number of at least 1" in capsys.readouterr().err
