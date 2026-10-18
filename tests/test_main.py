import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from unittest.mock import Mock

import pytest
from conftest import WYOMING_CONFIG_ENV_PREFIXES

import wyoming_openai.__main__ as main_module
from wyoming_openai.__main__ import main
from wyoming_openai.compatibility import create_tts_voices
from wyoming_openai.const import DEFAULT_LOG_BACKUP_COUNT, DEFAULT_LOG_MAX_BYTES


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "env_var,option",
    [
        ("STT_EXTRA_BODY", "--stt-extra-body"),
        ("TTS_EXTRA_BODY", "--tts-extra-body"),
        ("STT_REALTIME_EXTRA_BODY", "--stt-realtime-extra-body"),
        ("TTS_REALTIME_EXTRA_BODY", "--tts-realtime-extra-body"),
    ],
)
async def test_main_cli_extra_body_overrides_invalid_env(monkeypatch, env_var, option):
    monkeypatch.setenv(env_var, "{bad")
    kwargs = await _run_main_and_capture_handler_kwargs(monkeypatch, option, '{"custom":true}')
    assert kwargs[env_var.lower()] == {"custom": True}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "env_var,option",
    [
        ("STT_EXTRA_BODY", "--stt-extra-body"),
        ("TTS_EXTRA_BODY", "--tts-extra-body"),
        ("STT_REALTIME_EXTRA_BODY", "--stt-realtime-extra-body"),
        ("TTS_REALTIME_EXTRA_BODY", "--tts-realtime-extra-body"),
    ],
)
async def test_main_invalid_extra_body_env_reports_option_and_full_usage(monkeypatch, capsys, env_var, option):
    monkeypatch.setenv(env_var, "{bad")
    monkeypatch.setattr(sys, "argv", ["wyoming_openai"])
    with pytest.raises(SystemExit) as exc_info:
        await main()
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert option in err
    assert "--stt-models" in err


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "env_var", ["STT_EXTRA_BODY", "TTS_EXTRA_BODY", "STT_REALTIME_EXTRA_BODY", "TTS_REALTIME_EXTRA_BODY"]
)
async def test_main_help_ignores_invalid_extra_body_env(monkeypatch, capsys, env_var):
    monkeypatch.setenv(env_var, "{bad")
    monkeypatch.setattr(sys, "argv", ["wyoming_openai", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        await main()
    assert exc_info.value.code == 0
    assert "--stt-extra-body" in capsys.readouterr().out


INVALID_ENV_VALUES = [
    ("STT_BACKEND", "bogus", "--stt-backend"),
    ("TTS_BACKEND", "bogus", "--tts-backend"),
    ("STT_TEMPERATURE", "hot", "--stt-temperature"),
    ("TTS_SPEED", "fast", "--tts-speed"),
    ("TTS_SPEED", "inf", "--tts-speed"),
    ("TTS_STREAMING_MIN_WORDS", "two", "--tts-streaming-min-words"),
    ("TTS_STREAMING_MAX_CHARS", "many", "--tts-streaming-max-chars"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("env_var", "value", "option"), INVALID_ENV_VALUES)
async def test_main_help_ignores_invalid_env(monkeypatch, capsys, env_var, value, option):
    """Test --help works whatever the environment holds."""
    monkeypatch.setenv(env_var, value)
    monkeypatch.setattr(sys, "argv", ["wyoming_openai", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        await main()
    assert exc_info.value.code == 0
    assert option in capsys.readouterr().out


@pytest.mark.asyncio
@pytest.mark.parametrize(("env_var", "value", "option"), INVALID_ENV_VALUES)
async def test_main_reports_invalid_env_as_a_usage_error(monkeypatch, capsys, env_var, value, option):
    """Test an invalid environment value names its option instead of ending in a traceback."""
    monkeypatch.setenv(env_var, value)
    monkeypatch.setattr(sys, "argv", ["wyoming_openai", "--stt-models", "whisper-1"])
    with pytest.raises(SystemExit) as exc_info:
        await main()
    assert exc_info.value.code == 2
    assert option in capsys.readouterr().err


@pytest.mark.asyncio
async def test_main_cli_overrides_invalid_env(monkeypatch):
    """Test an explicit option wins over an invalid environment value for the same setting."""
    monkeypatch.setenv("TTS_SPEED", "fast")
    monkeypatch.setenv("TTS_STREAMING_MIN_WORDS", "two")

    kwargs = await _run_main_and_capture_handler_kwargs(
        monkeypatch, "--tts-speed", "1.25", "--tts-streaming-min-words", "3"
    )

    assert kwargs["tts_speed"] == 1.25
    assert kwargs["tts_streaming_min_words"] == 3


@pytest.mark.asyncio
async def test_main_reads_numeric_settings_from_env(monkeypatch):
    """Test numeric environment values still reach the handler, converted like their options."""
    monkeypatch.setenv("TTS_SPEED", "1.25")
    monkeypatch.setenv("STT_TEMPERATURE", "0.5")
    monkeypatch.setenv("TTS_STREAMING_MAX_CHARS", "120")

    kwargs = await _run_main_and_capture_handler_kwargs(monkeypatch)

    assert (kwargs["tts_speed"], kwargs["stt_temperature"], kwargs["tts_streaming_max_chars"]) == (1.25, 0.5, 120)


@pytest.mark.asyncio
async def test_main_rejects_a_non_finite_speed_option(monkeypatch, capsys):
    """Test a speed no request can carry fails at startup rather than at every synthesis."""
    monkeypatch.setattr(sys, "argv", ["wyoming_openai", "--tts-models", "gpt-4o-mini-tts", "--tts-speed", "nan"])
    with pytest.raises(SystemExit) as exc_info:
        await main()
    assert exc_info.value.code == 2
    assert "expected a finite number" in capsys.readouterr().err


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "env_var", ["STT_EXTRA_BODY", "TTS_EXTRA_BODY", "STT_REALTIME_EXTRA_BODY", "TTS_REALTIME_EXTRA_BODY"]
)
@pytest.mark.parametrize("value", ["", " ", "\n"])
async def test_main_whitespace_extra_body_env_is_unset(monkeypatch, env_var, value):
    monkeypatch.setenv(env_var, value)
    kwargs = await _run_main_and_capture_handler_kwargs(monkeypatch)
    assert kwargs[env_var.lower()] is None


@pytest.fixture(autouse=True)
def _clean_config_env(monkeypatch):
    """Keep proxy settings exported in the developer's shell out of main(); tests set the ones they need."""
    for env_var in list(os.environ):
        if env_var.startswith(WYOMING_CONFIG_ENV_PREFIXES):
            monkeypatch.delenv(env_var)


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


async def _advertised_tts_voices(monkeypatch, client, *cli_args):
    """Run main() with a fake TTS backend and return (name, description) of every voice the server advertises."""
    server = _CapturingServer()

    async def fake_factory(*args, **kwargs):
        return client

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
    for env_var in (
        "STT_MODELS",
        "STT_STREAMING_MODELS",
        "STT_REALTIME_MODELS",
        "STT_BACKEND",
        "STT_EXTRA_BODY",
        "TTS_MODELS",
        "TTS_STREAMING_MODELS",
        "TTS_VOICES",
        "TTS_BACKEND",
        "TTS_EXTRA_BODY",
        "TTS_CONCURRENT_REQUESTS",
    ):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(sys, "argv", ["wyoming_openai", "--tts-models", "tts-1", *cli_args])

    await main()

    assert len(server.handlers) == 1
    info = server.handlers[0]._wyoming_info
    return [(voice.name, voice.description) for program in info.tts for voice in program.voices]


@pytest.mark.asyncio
async def test_main_keeps_voice_names_as_descriptions_by_default(monkeypatch):
    monkeypatch.delenv("TTS_VOICE_LABELS", raising=False)

    voices = await _advertised_tts_voices(monkeypatch, _FakeClient(), "--tts-voices", "alloy", "echo")

    assert voices == [("alloy", "alloy"), ("echo", "echo")]


@pytest.mark.asyncio
async def test_main_applies_tts_voice_labels_from_cli(monkeypatch):
    monkeypatch.delenv("TTS_VOICE_LABELS", raising=False)

    voices = await _advertised_tts_voices(
        monkeypatch, _FakeClient(), "--tts-voices", "alloy", "echo", "--tts-voice-labels", '{"alloy": "Allie"}'
    )

    assert voices == [("alloy", "Allie"), ("echo", "echo")]


@pytest.mark.asyncio
async def test_main_applies_tts_voice_labels_from_env(monkeypatch):
    monkeypatch.setenv("TTS_VOICE_LABELS", '{"echo": "Echo (low)"}')

    voices = await _advertised_tts_voices(monkeypatch, _FakeClient(), "--tts-voices", "alloy", "echo")

    assert voices == [("alloy", "alloy"), ("echo", "Echo (low)")]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["", " ", "\n"])
async def test_main_ignores_blank_tts_voice_labels_env(monkeypatch, value):
    monkeypatch.setenv("TTS_VOICE_LABELS", value)

    voices = await _advertised_tts_voices(monkeypatch, _FakeClient(), "--tts-voices", "alloy")

    assert voices == [("alloy", "alloy")]


@pytest.mark.asyncio
async def test_main_applies_tts_voice_labels_to_voices_listed_by_the_backend(monkeypatch):
    class VoiceListingClient(_FakeClient):
        async def list_supported_voices(
            self, model_names, streaming_model_names, languages, *, realtime_model_names=None
        ):
            voices = ["alloy", "echo"]
            return create_tts_voices(model_names, streaming_model_names, voices, "http://tts.test", languages)

    monkeypatch.delenv("TTS_VOICE_LABELS", raising=False)

    voices = await _advertised_tts_voices(
        monkeypatch, VoiceListingClient(), "--tts-voice-labels", '{"alloy": "Allie"}'
    )

    assert voices == [("alloy", "Allie"), ("echo", "echo")]


@pytest.mark.asyncio
async def test_main_rejects_invalid_tts_voice_labels(monkeypatch, capsys):
    monkeypatch.delenv("TTS_VOICE_LABELS", raising=False)
    monkeypatch.setattr(sys, "argv", ["wyoming_openai", "--tts-voice-labels", '{"alloy": 3}'])

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 2
    assert "label for 'alloy' must be a non-empty string" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_main_rejects_invalid_tts_voice_labels_env(monkeypatch, capsys):
    monkeypatch.setenv("TTS_VOICE_LABELS", "not json")
    monkeypatch.setattr(sys, "argv", ["wyoming_openai"])

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 2
    assert "Invalid TTS voice labels" in capsys.readouterr().err


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
        # A proxy in front of OpenAI is classed as OPENAI without being the official API
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
    # Every TTS model is a Realtime one, so the speech API body goes unused
    assert "TTS extra body is set but unused" in caplog.text


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


@pytest.mark.asyncio
async def test_main_allows_unused_tts_realtime_extra_body_in_stt_only_mode(monkeypatch, caplog):
    """Test a Realtime TTS body is not validated without TTS, like the other bodies, only reported as unused."""

    async def fake_factory(*args, **kwargs):
        return _FakeClient()

    for env_var in ("TTS_MODELS", "TTS_STREAMING_MODELS", "TTS_REALTIME_MODELS", "TTS_VOICES"):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(
        main_module.CustomAsyncOpenAI, "create_autodetected_factory", staticmethod(lambda: fake_factory)
    )
    monkeypatch.setattr(main_module.AsyncServer, "from_uri", staticmethod(lambda uri: _CapturingServer()))
    monkeypatch.setattr(main_module, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys, "argv", ["wyoming_openai", "--stt-models", "whisper-1", "--tts-realtime-extra-body", '{"model":"x"}']
    )

    with caplog.at_level("WARNING"):
        await main()

    assert "TTS Realtime extra body is set but unused" in caplog.text


@pytest.mark.asyncio
async def test_main_rejects_incompatible_realtime_extra_body_without_realtime_models(monkeypatch, capsys):
    """Test a Realtime extra body is validated even when no Realtime models would use it."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["wyoming_openai", "--tts-models", "gpt-4o-mini-tts", "--tts-realtime-extra-body", '{"model":"x"}'],
    )

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 2
    assert "'model'" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_main_warns_about_unused_realtime_extra_bodies(monkeypatch, caplog):
    """Test Realtime extra bodies without Realtime models are reported instead of silently ignored."""
    server = _CapturingServer()

    async def fake_factory(*args, **kwargs):
        return _FakeClient()

    monkeypatch.setattr(
        main_module.CustomAsyncOpenAI, "create_autodetected_factory", staticmethod(lambda: fake_factory)
    )
    monkeypatch.setattr(main_module.AsyncServer, "from_uri", staticmethod(lambda uri: server))
    monkeypatch.setattr(main_module, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wyoming_openai",
            "--stt-models",
            "whisper-1",
            "--stt-realtime-extra-body",
            '{"keywords":["Wyoming"]}',
            "--tts-models",
            "gpt-4o-mini-tts",
            "--tts-voices",
            "alloy",
            "--tts-extra-body",
            '{"lang_code":"en"}',
            "--tts-realtime-extra-body",
            '{"reasoning":{"effort":"low"}}',
        ],
    )

    with caplog.at_level("WARNING"):
        await main()

    assert "TTS Realtime extra body is set but unused" in caplog.text
    assert "STT Realtime extra body is set but unused" in caplog.text
    # A speech API model is configured, so its extra body is in use
    assert "TTS extra body is set but unused" not in caplog.text
    assert "STT extra body is set but unused" not in caplog.text


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
async def test_main_keeps_configured_voices_for_realtime_models_off_the_official_api(monkeypatch, caplog):
    """Test a server that only counts as OPENAI by default is not blocked over voice names it may offer itself."""
    server = _CapturingServer()

    async def fake_factory(*args, **kwargs):
        client = _FakeClient()
        client.backend = main_module.OpenAIBackend.OPENAI
        client.is_official_openai = False
        return client

    monkeypatch.setattr(
        main_module.CustomAsyncOpenAI,
        "create_autodetected_factory",
        staticmethod(lambda: fake_factory),
    )
    monkeypatch.setattr(main_module.AsyncServer, "from_uri", staticmethod(lambda uri: server))
    monkeypatch.setattr(main_module, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["wyoming_openai", "--tts-realtime-models", "gpt-realtime-2.1-mini", "--tts-voices", "onyx", "nova"],
    )

    with caplog.at_level("WARNING"):
        await main()

    handler = server.handlers[0]
    assert [voice.name for voice in handler._wyoming_info.tts[0].voices] == ["onyx", "nova"]
    assert "is not the official OpenAI API" in caplog.text


@pytest.mark.asyncio
async def test_main_passes_stt_realtime_extra_body(monkeypatch, caplog):
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

    with caplog.at_level("WARNING"):
        await main()

    handler = server.handlers[0]
    assert handler._stt_extra_body == {"temperature": 0.2}
    assert handler._stt_realtime_extra_body == {"keywords": ["Wyoming"]}
    # The only STT model is a Realtime one, so the /v1/audio/transcriptions body goes unused
    assert "STT extra body is set but unused" in caplog.text


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


# Logging configuration


def _capture_basic_config(monkeypatch):
    """Intercept logging.basicConfig so a test never rewrites the root logger's real handlers."""
    configured = {}
    monkeypatch.setattr(main_module.logging, "basicConfig", lambda **kwargs: configured.update(kwargs))
    return configured


def _close_captured_handlers(configured):
    """Release the file handles of handlers that never reached the real root logger."""
    for handler in configured.get("handlers", []):
        handler.close()


def test_configure_logging_rotates_the_log_file(tmp_path, monkeypatch):
    configured = _capture_basic_config(monkeypatch)
    log_path = tmp_path / "logs" / "wyoming_openai.log"

    main_module.configure_logging("DEBUG", str(log_path), max_bytes=256, backup_count=1)

    assert configured["level"] == logging.DEBUG
    file_handler = next(handler for handler in configured["handlers"] if isinstance(handler, RotatingFileHandler))
    assert file_handler.maxBytes == 256
    assert file_handler.backupCount == 1

    logger = logging.getLogger("test_configure_logging_rotates_the_log_file")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    try:
        for index in range(20):
            logger.info("entry %02d %s", index, "x" * 40)
    finally:
        logger.removeHandler(file_handler)
        _close_captured_handlers(configured)

    assert (tmp_path / "logs" / "wyoming_openai.log.1").exists()
    assert log_path.stat().st_size <= 256


def test_configure_logging_defaults_to_rotating_the_log_file(tmp_path, monkeypatch):
    configured = _capture_basic_config(monkeypatch)

    try:
        main_module.configure_logging("INFO", str(tmp_path / "wyoming_openai.log"))

        file_handler = next(handler for handler in configured["handlers"] if isinstance(handler, RotatingFileHandler))
        assert file_handler.maxBytes == DEFAULT_LOG_MAX_BYTES
        assert file_handler.backupCount == DEFAULT_LOG_BACKUP_COUNT
    finally:
        _close_captured_handlers(configured)


def test_configure_logging_prunes_backups_above_the_configured_count(tmp_path, monkeypatch):
    configured = _capture_basic_config(monkeypatch)
    log_path = tmp_path / "wyoming_openai.log"
    for index in range(1, 6):
        log_path.with_name(f"{log_path.name}.{index}").write_text("stale", encoding="utf-8")
    unrelated = log_path.with_name(f"{log_path.name}.keep")
    unrelated.write_text("not a numbered backup", encoding="utf-8")

    try:
        main_module.configure_logging("INFO", str(log_path), max_bytes=1024, backup_count=2)
    finally:
        _close_captured_handlers(configured)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "wyoming_openai.log",
        "wyoming_openai.log.1",
        "wyoming_openai.log.2",
        "wyoming_openai.log.keep",
    ]


def test_configure_logging_keeps_old_backups_when_rotation_is_disabled(tmp_path, monkeypatch):
    configured = _capture_basic_config(monkeypatch)
    log_path = tmp_path / "wyoming_openai.log"
    stale = log_path.with_name(f"{log_path.name}.3")
    stale.write_text("kept", encoding="utf-8")

    try:
        main_module.configure_logging("INFO", str(log_path), max_bytes=0, backup_count=1)
    finally:
        _close_captured_handlers(configured)

    assert stale.exists()


def test_configure_logging_without_log_file_only_logs_to_the_stream(monkeypatch):
    configured = _capture_basic_config(monkeypatch)

    main_module.configure_logging("INFO")

    assert [type(handler) for handler in configured["handlers"]] == [logging.StreamHandler]


def test_configure_logging_rejects_an_unknown_level():
    with pytest.raises(ValueError, match="Invalid log level"):
        main_module.configure_logging("VERBOSE")


@pytest.mark.asyncio
async def test_main_passes_default_log_rotation_options(monkeypatch):
    calls = []
    monkeypatch.setattr(main_module, "configure_logging", lambda *args, **kwargs: calls.append((args, kwargs)))

    await _run_main_and_capture_handler_kwargs(monkeypatch)

    assert calls == [
        (("INFO", None), {"max_bytes": DEFAULT_LOG_MAX_BYTES, "backup_count": DEFAULT_LOG_BACKUP_COUNT})
    ]


@pytest.mark.asyncio
async def test_main_reads_log_rotation_from_env(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(main_module, "configure_logging", lambda *args, **kwargs: calls.append((args, kwargs)))
    log_path = tmp_path / "wyoming_openai.log"
    monkeypatch.setenv("WYOMING_LOG_FILE", str(log_path))
    monkeypatch.setenv("WYOMING_LOG_MAX_BYTES", "4096")
    monkeypatch.setenv("WYOMING_LOG_BACKUP_COUNT", "2")

    await _run_main_and_capture_handler_kwargs(monkeypatch)

    assert calls == [(("INFO", str(log_path)), {"max_bytes": 4096, "backup_count": 2})]


@pytest.mark.asyncio
async def test_main_cli_log_rotation_overrides_env(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(main_module, "configure_logging", lambda *args, **kwargs: calls.append((args, kwargs)))
    log_path = tmp_path / "wyoming_openai.log"
    monkeypatch.setenv("WYOMING_LOG_MAX_BYTES", "4096")
    monkeypatch.setenv("WYOMING_LOG_BACKUP_COUNT", "2")

    await _run_main_and_capture_handler_kwargs(
        monkeypatch,
        "--log-file",
        str(log_path),
        "--log-max-bytes",
        "2048",
        "--log-backup-count",
        "3",
    )

    assert calls == [(("INFO", str(log_path)), {"max_bytes": 2048, "backup_count": 3})]


@pytest.mark.asyncio
async def test_main_allows_disabling_log_rotation(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(main_module, "configure_logging", lambda *args, **kwargs: calls.append((args, kwargs)))

    await _run_main_and_capture_handler_kwargs(
        monkeypatch, "--log-file", str(tmp_path / "wyoming_openai.log"), "--log-max-bytes", "0"
    )

    assert calls[0][1]["max_bytes"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["-1", "two", "1.5"])
async def test_main_rejects_invalid_log_max_bytes(monkeypatch, capsys, value):
    monkeypatch.setattr(sys, "argv", ["wyoming_openai", "--log-max-bytes", value])

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 2
    assert "--log-max-bytes: expected a whole number of at least 0" in capsys.readouterr().err


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["0", "-1", "two"])
async def test_main_rejects_invalid_log_backup_count(monkeypatch, capsys, value):
    monkeypatch.setattr(sys, "argv", ["wyoming_openai", "--log-backup-count", value])

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 2
    assert "--log-backup-count: expected a whole number of at least 1" in capsys.readouterr().err


def test_configure_logging_ignores_non_decimal_backup_names(tmp_path, monkeypatch):
    configured = _capture_basic_config(monkeypatch)
    log_path = tmp_path / "wyoming_openai.log"
    # "²".isdigit() is true, but it is not a backup number that int() accepts
    unicode_digit = log_path.with_name(f"{log_path.name}.\u00b2")
    unicode_digit.write_text("unrelated", encoding="utf-8")

    try:
        main_module.configure_logging("INFO", str(log_path), max_bytes=1024, backup_count=1)
    finally:
        _close_captured_handlers(configured)

    assert unicode_digit.exists()


def test_configure_logging_starts_when_the_log_directory_cannot_be_listed(monkeypatch, tmp_path):
    configured = _capture_basic_config(monkeypatch)
    log_path = tmp_path / "wyoming_openai.log"

    def deny_listing(directory):
        raise PermissionError("listing denied")

    monkeypatch.setattr(main_module.os, "listdir", deny_listing)

    try:
        main_module.configure_logging("INFO", str(log_path), max_bytes=1024, backup_count=1)
        file_handler = next(handler for handler in configured["handlers"] if isinstance(handler, RotatingFileHandler))
        assert file_handler.baseFilename == str(log_path)
    finally:
        _close_captured_handlers(configured)
