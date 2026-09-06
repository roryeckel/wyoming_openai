import argparse
import asyncio
import logging
import os
from contextlib import AsyncExitStack
from functools import partial

from wyoming.server import AsyncServer

from .compatibility import (
    CustomAsyncOpenAI,
    OpenAIBackend,
    asr_model_to_string,
    create_asr_programs,
    create_info,
    create_tts_programs,
    create_tts_voices,
    tts_voice_to_string,
)
from .const import DEFAULT_OPENAI_BASE_URL, __version__
from .handler import OpenAIEventHandler
from .utilities import (
    create_enum_parser,
    create_json_object_parser,
    validate_stt_extra_body,
    validate_tts_extra_body,
)


def configure_logging(level, log_file=None):
    """Configure logging based on a string level, optionally also logging to a file."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=numeric_level, handlers=handlers, force=True)


async def main():
    """Main entry point for the Wyoming OpenAI server."""
    parser = argparse.ArgumentParser()

    # Create reusable enum parser for backend arguments
    backend_parser = create_enum_parser(OpenAIBackend)
    stt_extra_body_parser = create_json_object_parser("STT extra body")
    tts_extra_body_parser = create_json_object_parser("TTS extra body")

    stt_backend_env = os.getenv("STT_BACKEND")
    stt_backend_default = None
    if stt_backend_env:
        try:
            stt_backend_default = backend_parser(stt_backend_env)
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))

    tts_backend_env = os.getenv("TTS_BACKEND")
    tts_backend_default = None
    if tts_backend_env:
        try:
            tts_backend_default = backend_parser(tts_backend_env)
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))

    stt_extra_body_env = os.getenv("STT_EXTRA_BODY")
    stt_extra_body_default = None
    if stt_extra_body_env:
        try:
            stt_extra_body_default = stt_extra_body_parser(stt_extra_body_env)
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))

    tts_extra_body_env = os.getenv("TTS_EXTRA_BODY")
    tts_extra_body_default = None
    if tts_extra_body_env:
        try:
            tts_extra_body_default = tts_extra_body_parser(tts_extra_body_env)
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))

    # General configuration
    parser.add_argument(
        "--uri", default=os.getenv("WYOMING_URI", "tcp://0.0.0.0:10300"), help="This Wyoming Server URI"
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("WYOMING_LOG_LEVEL", "INFO"),
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument(
        "--log-file",
        default=os.getenv("WYOMING_LOG_FILE", None),
        help="Optional file path to additionally log to (created if missing)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=os.getenv("WYOMING_LANGUAGES", "en").split(),
        help="List of languages supported by BOTH STT AND TTS (example: en, fr)",
    )

    # STT configuration
    parser.add_argument(
        "--stt-openai-key",
        required=False,
        default=os.getenv("STT_OPENAI_KEY", None),
        help="OpenAI API key for speech-to-text",
    )
    parser.add_argument(
        "--stt-openai-url",
        default=os.getenv("STT_OPENAI_URL", DEFAULT_OPENAI_BASE_URL),
        help="Custom OpenAI API base URL for STT",
    )
    parser.add_argument(
        "--stt-models",
        nargs="+",  # Use nargs to accept multiple values
        default=os.getenv("STT_MODELS", "").split(),
        help="List of STT model identifiers",
    )
    parser.add_argument(
        "--stt-backend",
        type=backend_parser,
        required=False,
        choices=list(OpenAIBackend),
        default=stt_backend_default,
        help="Backend for speech-to-text (OPENAI, SPEACHES, KOKORO_FASTAPI, LOCALAI, or None)",
    )
    parser.add_argument(
        "--stt-temperature",
        type=float,
        default=float(_v) if (_v := os.getenv("STT_TEMPERATURE")) else None,
        help="Sampling temperature for speech-to-text (0.0 to 1.0, default is None for OpenAI default)",
    )
    parser.add_argument("--stt-prompt", default=os.getenv("STT_PROMPT", None), help="Optional prompt for STT requests")
    parser.add_argument(
        "--stt-extra-body",
        type=stt_extra_body_parser,
        default=stt_extra_body_default,
        help=(
            "Optional JSON object merged into the STT request body via extra_body; "
            "overlapping keys override top-level request fields. "
            "'response_format' must remain 'json' and 'stream' must be a boolean. "
            "Incompatible values cause a startup error"
        ),
    )
    parser.add_argument(
        "--stt-streaming-models",
        nargs="+",
        default=os.getenv("STT_STREAMING_MODELS", "").split(),
        help="Space-separated list of STT model names that support streaming (e.g. gpt-4o-transcribe)",
    )
    parser.add_argument(
        "--stt-realtime-models",
        nargs="+",
        default=os.getenv("STT_REALTIME_MODELS", "").split(),
        help="Space-separated list of STT model names that use Realtime transcription (e.g. gpt-realtime-whisper)",
    )

    # TTS configuration
    parser.add_argument(
        "--tts-openai-key",
        required=False,
        default=os.getenv("TTS_OPENAI_KEY", None),
        help="OpenAI API key for text-to-speech",
    )
    parser.add_argument(
        "--tts-openai-url",
        default=os.getenv("TTS_OPENAI_URL", DEFAULT_OPENAI_BASE_URL),
        help="Custom OpenAI API base URL for TTS",
    )
    parser.add_argument(
        "--tts-models", nargs="+", default=os.getenv("TTS_MODELS", "").split(), help="List of TTS model identifiers"
    )
    parser.add_argument(
        "--tts-voices",
        nargs="+",
        default=os.getenv("TTS_VOICES", "").split(),
        required=False,
        help="List of available TTS voices",
    )
    parser.add_argument(
        "--tts-backend",
        type=backend_parser,
        required=False,
        choices=list(OpenAIBackend),
        default=tts_backend_default,
        help="Backend for text-to-speech (OPENAI, SPEACHES, KOKORO_FASTAPI, LOCALAI, or None)",
    )
    parser.add_argument(
        "--tts-speed",
        type=float,
        default=float(_v) if (_v := os.getenv("TTS_SPEED")) else None,
        help="Speed of the TTS output (0.25 to 4.0, default is None for OpenAI default)",
    )
    parser.add_argument(
        "--tts-instructions", default=os.getenv("TTS_INSTRUCTIONS", None), help="Optional instructions for TTS requests"
    )
    parser.add_argument(
        "--tts-extra-body",
        type=tts_extra_body_parser,
        default=tts_extra_body_default,
        help=(
            "Optional JSON object merged into the TTS request body via extra_body; "
            "overlapping keys override top-level request fields. "
            "'stream' and 'stream_format' are not allowed; "
            "'response_format' is limited to 'pcm' or 'wav'. "
            "Incompatible values cause a startup error"
        ),
    )
    parser.add_argument(
        "--tts-streaming-models",
        nargs="+",
        default=os.getenv("TTS_STREAMING_MODELS", "").split(),
        help="Space-separated list of TTS model names that support streaming synthesis (e.g. tts-1)",
    )
    parser.add_argument(
        "--tts-streaming-min-words",
        type=int,
        default=int(_v) if (_v := os.getenv("TTS_STREAMING_MIN_WORDS")) else None,
        help="Minimum words per chunk for streaming TTS (optional)",
    )
    parser.add_argument(
        "--tts-streaming-max-chars",
        type=int,
        default=int(_v) if (_v := os.getenv("TTS_STREAMING_MAX_CHARS")) else None,
        help="Maximum characters per chunk for streaming TTS (optional)",
    )

    args = parser.parse_args()

    stt_requested = bool(args.stt_models or args.stt_streaming_models or args.stt_realtime_models)
    tts_requested = bool(args.tts_models or args.tts_streaming_models)
    tts_validation_deferred = tts_requested and not args.tts_voices

    try:
        if stt_requested:
            validate_stt_extra_body(args.stt_extra_body)
        if tts_requested and args.tts_voices:
            validate_tts_extra_body(args.tts_extra_body)
    except ValueError as exc:
        parser.error(str(exc))

    configure_logging(args.log_level, args.log_file)
    _logger = logging.getLogger(__name__)

    _logger.info("Starting Wyoming OpenAI %s", __version__)

    if not stt_requested and not tts_requested:
        _logger.error("No STT or TTS models specified. Exiting.")
        return

    stt_client: CustomAsyncOpenAI | None = None
    if stt_requested:
        if args.stt_backend is None:
            _logger.debug("STT backend is None, autodetecting...")
            stt_factory = CustomAsyncOpenAI.create_autodetected_factory()
        else:
            stt_factory = CustomAsyncOpenAI.create_backend_factory(args.stt_backend)

        stt_client = await stt_factory(api_key=args.stt_openai_key, base_url=args.stt_openai_url)
        _logger.debug("Detected STT backend: %s", stt_client.backend)

    tts_client: CustomAsyncOpenAI | None = None
    if tts_requested:
        if args.tts_backend is None:
            _logger.debug("TTS backend is None, autodetecting...")
            tts_factory = CustomAsyncOpenAI.create_autodetected_factory()
        else:
            tts_factory = CustomAsyncOpenAI.create_backend_factory(args.tts_backend)

        tts_client = await tts_factory(api_key=args.tts_openai_key, base_url=args.tts_openai_url)
        _logger.debug("Detected TTS backend: %s", tts_client.backend)

    # Use only configured clients in async context managers
    async with AsyncExitStack() as exit_stack:
        if stt_client is not None:
            stt_client = await exit_stack.enter_async_context(stt_client)
        if tts_client is not None:
            tts_client = await exit_stack.enter_async_context(tts_client)

        asr_programs = (
            create_asr_programs(
                args.stt_models,
                args.stt_streaming_models,
                args.stt_openai_url,
                args.languages,
                stt_realtime_models=args.stt_realtime_models,
            )
            if stt_requested
            else []
        )

        if not tts_requested:
            tts_voices = []
        elif args.tts_voices:
            # If TTS_VOICES is set, use that
            tts_voices = create_tts_voices(
                args.tts_models, args.tts_streaming_models, args.tts_voices, args.tts_openai_url, args.languages
            )
        else:
            # Otherwise, list supported voices via backend (with streaming fallback)
            assert tts_client is not None
            tts_voices = await tts_client.list_supported_voices(
                args.tts_models, args.tts_streaming_models, args.languages
            )

        tts_programs = create_tts_programs(tts_voices, tts_streaming_models=args.tts_streaming_models)

        if not asr_programs and not tts_programs:
            _logger.error("No STT or TTS models specified. Exiting.")
            return

        try:
            if tts_validation_deferred and tts_programs:
                validate_tts_extra_body(args.tts_extra_body)
        except ValueError as exc:
            parser.error(str(exc))

        info = create_info(asr_programs, tts_programs)

        # Log the model configurations
        if asr_programs:
            streaming_asr_models_for_logging = []
            non_streaming_asr_models_for_logging = []

            for prog in asr_programs:
                for model in prog.models:
                    if prog.supports_transcript_streaming:
                        streaming_asr_models_for_logging.append(model)
                    else:
                        non_streaming_asr_models_for_logging.append(model)

            if streaming_asr_models_for_logging:
                _logger.info(
                    "*** Streaming ASR Models ***\n%s",
                    "\n".join(asr_model_to_string(x, is_streaming=True) for x in streaming_asr_models_for_logging),
                )
            else:
                _logger.info("No Streaming ASR models specified")

            if non_streaming_asr_models_for_logging:
                _logger.info(
                    "*** Non-Streaming ASR Models ***\n%s",
                    "\n".join(asr_model_to_string(x, is_streaming=False) for x in non_streaming_asr_models_for_logging),
                )
            else:
                _logger.info("No Non-Streaming ASR models specified")
        else:
            _logger.warning("No ASR models specified")

        if tts_programs:
            streaming_tts_voices_for_logging = []
            non_streaming_tts_voices_for_logging = []

            for prog in tts_programs:
                for voice in prog.voices:
                    if getattr(prog, "supports_synthesize_streaming", False):
                        streaming_tts_voices_for_logging.append(voice)
                    else:
                        non_streaming_tts_voices_for_logging.append(voice)

            if streaming_tts_voices_for_logging:
                _logger.info(
                    "*** Streaming TTS Voices ***\n%s",
                    "\n".join(tts_voice_to_string(x) for x in streaming_tts_voices_for_logging),
                )
            else:
                _logger.info("No Streaming TTS voices specified")

            if non_streaming_tts_voices_for_logging:
                _logger.info(
                    "*** Non-Streaming TTS Voices ***\n%s",
                    "\n".join(tts_voice_to_string(x) for x in non_streaming_tts_voices_for_logging),
                )
            else:
                _logger.info("No Non-Streaming TTS voices specified")
        else:
            _logger.warning("No TTS models specified")

        # Create Wyoming server
        server = AsyncServer.from_uri(args.uri)

        # Run Wyoming server
        _logger.info("Starting server at %s", args.uri)
        await server.run(
            partial(
                OpenAIEventHandler,
                info=info,
                stt_client=stt_client,
                tts_client=tts_client,
                stt_temperature=args.stt_temperature,
                tts_speed=args.tts_speed,
                tts_instructions=args.tts_instructions,
                stt_prompt=args.stt_prompt,
                stt_extra_body=args.stt_extra_body,
                stt_realtime_models=args.stt_realtime_models,
                tts_extra_body=args.tts_extra_body,
                tts_streaming_min_words=args.tts_streaming_min_words,
                tts_streaming_max_chars=args.tts_streaming_max_chars,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
