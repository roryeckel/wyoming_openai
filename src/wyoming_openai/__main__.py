import argparse
import asyncio
import logging
import os
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
from .utilities import create_enum_parser, create_json_object_parser, validate_stt_extra_body, validate_tts_extra_body


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _env_float(name: str) -> float | None:
    v = os.getenv(name)
    return float(v) if v else None


def _env_int(name: str) -> int | None:
    v = os.getenv(name)
    return int(v) if v else None


def _try_parse(parser_fn, env_val, argparser):
    """Parse an env var with a parser function, calling parser.error on failure."""
    if not env_val:
        return None
    try:
        return parser_fn(env_val)
    except argparse.ArgumentTypeError as exc:
        argparser.error(str(exc))


async def main():
    parser = argparse.ArgumentParser()
    backend_parser = create_enum_parser(OpenAIBackend)
    stt_extra_parser = create_json_object_parser("STT extra body")
    tts_extra_parser = create_json_object_parser("TTS extra body")

    # Pre-parse env vars that need validation
    stt_backend_default = _try_parse(backend_parser, _env("STT_BACKEND"), parser)
    tts_backend_default = _try_parse(backend_parser, _env("TTS_BACKEND"), parser)
    stt_extra_default = _try_parse(stt_extra_parser, _env("STT_EXTRA_BODY"), parser)
    tts_extra_default = _try_parse(tts_extra_parser, _env("TTS_EXTRA_BODY"), parser)

    # General
    parser.add_argument("--uri", default=_env("WYOMING_URI", "tcp://0.0.0.0:10300"))
    parser.add_argument("--log-level", default=_env("WYOMING_LOG_LEVEL", "INFO"))
    parser.add_argument("--languages", nargs="+", default=_env("WYOMING_LANGUAGES", "en").split())

    # STT
    parser.add_argument("--stt-openai-key", default=_env("STT_OPENAI_KEY") or None)
    parser.add_argument("--stt-openai-url", default=_env("STT_OPENAI_URL", DEFAULT_OPENAI_BASE_URL))
    parser.add_argument("--stt-models", nargs="+", default=_env("STT_MODELS").split())
    parser.add_argument("--stt-streaming-models", nargs="+", default=_env("STT_STREAMING_MODELS").split())
    parser.add_argument("--stt-backend", type=backend_parser, choices=list(OpenAIBackend), default=stt_backend_default)
    parser.add_argument("--stt-temperature", type=float, default=_env_float("STT_TEMPERATURE"))
    parser.add_argument("--stt-prompt", default=_env("STT_PROMPT") or None)
    parser.add_argument(
        "--stt-extra-body", type=stt_extra_parser, default=stt_extra_default,
        help="JSON object merged into STT requests. 'response_format' must be 'json', 'stream' must be bool.",
    )

    # TTS
    parser.add_argument("--tts-openai-key", default=_env("TTS_OPENAI_KEY") or None)
    parser.add_argument("--tts-openai-url", default=_env("TTS_OPENAI_URL", DEFAULT_OPENAI_BASE_URL))
    parser.add_argument("--tts-models", nargs="+", default=_env("TTS_MODELS").split())
    parser.add_argument("--tts-streaming-models", nargs="+", default=_env("TTS_STREAMING_MODELS").split())
    parser.add_argument("--tts-voices", nargs="+", default=_env("TTS_VOICES").split())
    parser.add_argument("--tts-backend", type=backend_parser, choices=list(OpenAIBackend), default=tts_backend_default)
    parser.add_argument("--tts-speed", type=float, default=_env_float("TTS_SPEED"))
    parser.add_argument("--tts-instructions", default=_env("TTS_INSTRUCTIONS") or None)
    parser.add_argument(
        "--tts-extra-body", type=tts_extra_parser, default=tts_extra_default,
        help="JSON object merged into TTS requests. 'stream'/'stream_format' disallowed. 'response_format': pcm|wav.",
    )
    parser.add_argument("--tts-streaming-min-words", type=int, default=_env_int("TTS_STREAMING_MIN_WORDS"))
    parser.add_argument("--tts-streaming-max-chars", type=int, default=_env_int("TTS_STREAMING_MAX_CHARS"))

    args = parser.parse_args()

    stt_requested = bool(args.stt_models or args.stt_streaming_models)
    tts_requested = bool(args.tts_models or args.tts_streaming_models)
    tts_validation_deferred = tts_requested and not args.tts_voices

    try:
        if stt_requested:
            validate_stt_extra_body(args.stt_extra_body)
        if tts_requested and args.tts_voices:
            validate_tts_extra_body(args.tts_extra_body)
    except ValueError as exc:
        parser.error(str(exc))

    # Logging
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, force=True)
    log = logging.getLogger(__name__)
    log.info("Wyoming OpenAI %s", __version__)

    # Create clients
    async def make_client(backend, key, url):
        if backend is None:
            factory = CustomAsyncOpenAI.create_autodetected_factory()
        else:
            factory = CustomAsyncOpenAI.create_backend_factory(backend)
        return await factory(api_key=key, base_url=url)

    stt_client = await make_client(args.stt_backend, args.stt_openai_key, args.stt_openai_url)
    tts_client = await make_client(args.tts_backend, args.tts_openai_key, args.tts_openai_url)
    log.debug("STT backend: %s | TTS backend: %s", stt_client.backend, tts_client.backend)

    async with stt_client, tts_client:
        asr_programs = create_asr_programs(args.stt_models, args.stt_streaming_models, args.stt_openai_url, args.languages)

        if args.tts_voices:
            tts_voices = create_tts_voices(
                args.tts_models, args.tts_streaming_models, args.tts_voices, args.tts_openai_url, args.languages
            )
        else:
            tts_voices = await tts_client.list_supported_voices(args.tts_models, args.tts_streaming_models, args.languages)

        tts_programs = create_tts_programs(tts_voices, tts_streaming_models=args.tts_streaming_models)

        if not asr_programs and not tts_programs:
            log.error("No STT or TTS models specified. Exiting.")
            return

        try:
            if tts_validation_deferred and tts_programs:
                validate_tts_extra_body(args.tts_extra_body)
        except ValueError as exc:
            parser.error(str(exc))

        info = create_info(asr_programs, tts_programs)

        # Log models/voices
        for prog in asr_programs:
            for m in prog.models:
                log.info(asr_model_to_string(m, is_streaming=prog.supports_transcript_streaming))
        for prog in tts_programs:
            for v in prog.voices:
                log.info(tts_voice_to_string(v))
        if not asr_programs:
            log.warning("No ASR models configured")
        if not tts_programs:
            log.warning("No TTS voices configured")

        server = AsyncServer.from_uri(args.uri)
        log.info("Starting server at %s", args.uri)
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
                tts_extra_body=args.tts_extra_body,
                tts_streaming_min_words=args.tts_streaming_min_words,
                tts_streaming_max_chars=args.tts_streaming_max_chars,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
