import argparse
import asyncio
import logging
import os
from functools import partial

from wyoming.server import AsyncServer

from . import __version__
from .compatibility import (
    CustomAsyncOpenAI,
    OpenAIBackend,
    asr_model_to_string,
    create_asr_models,
    create_tts_voices,
    tts_voice_to_string,
)
from .handler import OpenAIEventHandler


def configure_logging(level):
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    logging.basicConfig(level=numeric_level)

async def main():
    env_stt_backend = os.getenv("STT_BACKEND")
    env_tts_backend = os.getenv("TTS_BACKEND")
    parser = argparse.ArgumentParser()

    # General configuration
    parser.add_argument(
        "--uri",
        default=os.getenv("WYOMING_URI","tcp://0.0.0.0:10300"),
        help="This Wyoming Server URI"
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("WYOMING_LOG_LEVEL", "INFO"),
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=os.getenv("WYOMING_LANGUAGES", "en").split(),
        help="List of languages supported by BOTH STT AND TTS (example: en, fr)"
    )

    # STT configuration
    parser.add_argument(
        "--stt-openai-key",
        required=False,
        default=os.getenv("STT_OPENAI_KEY", None),
        help="OpenAI API key for speech-to-text"
    )
    parser.add_argument(
        "--stt-openai-url",
        default=os.getenv("STT_OPENAI_URL", "https://api.openai.com/v1"),
        help="Custom OpenAI API base URL for STT"
    )
    parser.add_argument(
        "--stt-models",
        nargs='+',  # Use nargs to accept multiple values
        default=os.getenv("STT_MODELS", 'whisper-1 gpt-4o-mini-transcribe gpt-4o-transcribe').split(),
        help="List of STT model identifiers"
    )
    parser.add_argument(
        "--stt-backend",
        type=OpenAIBackend,
        required=False,
        choices=list(OpenAIBackend),
        default=OpenAIBackend[env_stt_backend] if env_stt_backend else None,
        help="Backend for speech-to-text (OPENAI, SPEACHES, KOKORO_FASTAPI, or None)"
    )
    parser.add_argument(
        "--stt-temperature",
        type=float,
        default=float(os.getenv("STT_TEMPERATURE")) if os.getenv("STT_TEMPERATURE") else None,
        help="Sampling temperature for speech-to-text (0.0 to 1.0, default is None for OpenAI default)"
    )

    # TTS configuration
    parser.add_argument(
        "--tts-openai-key",
        required=False,
        default=os.getenv("TTS_OPENAI_KEY", None),
        help="OpenAI API key for text-to-speech"
    )
    parser.add_argument(
        "--tts-openai-url",
        default=os.getenv("TTS_OPENAI_URL", "https://api.openai.com/v1"),
        help="Custom OpenAI API base URL for TTS"
    )
    parser.add_argument(
        "--tts-models",
        nargs='+',
        default=os.getenv("TTS_MODELS", 'tts-1 tts-1-hd gpt-4o-mini-tts').split(),
        help="List of TTS model identifiers"
    )
    parser.add_argument(
        "--tts-voices",
        nargs='+',
        default=os.getenv("TTS_VOICES", '').split(),
        required=False,
        help="List of available TTS voices"
    )
    parser.add_argument(
        "--tts-backend",
        type=OpenAIBackend,
        required=False,
        choices=list(OpenAIBackend),
        default=OpenAIBackend[env_tts_backend] if env_tts_backend else None,
        help="Backend for text-to-speech (OPENAI, SPEACHES, KOKORO_FASTAPI, or None)"
    )
    parser.add_argument(
        "--tts-speed",
        type=float,
        default=float(os.getenv("TTS_SPEED")) if os.getenv("TTS_SPEED") else None,
        help="Speed of the TTS output (0.25 to 4.0, default is None for OpenAI default)"
    )

    args = parser.parse_args()

    configure_logging(args.log_level)
    _logger = logging.getLogger(__name__)

    _logger.info("Starting Wyoming OpenAI %s", __version__)

    # Create factories and clients
    if args.stt_backend is None:
        _logger.debug("STT backend is None, autodetecting...")
        stt_factory = CustomAsyncOpenAI.create_autodetected_factory()
    else:
        stt_factory = CustomAsyncOpenAI.create_backend_factory(args.stt_backend)
    stt_client = await stt_factory(api_key=args.stt_openai_key, base_url=args.stt_openai_url)
    _logger.debug("Detected STT backend: %s", stt_client.backend)

    if args.tts_backend is None:
        _logger.debug("TTS backend is None, autodetecting...")
        tts_factory = CustomAsyncOpenAI.create_autodetected_factory()
    else:
        tts_factory = CustomAsyncOpenAI.create_backend_factory(args.tts_backend)
    tts_client = await tts_factory(api_key=args.tts_openai_key, base_url=args.tts_openai_url)
    _logger.debug("Detected TTS backend: %s", tts_client.backend)

    asr_models = create_asr_models(args.stt_models, args.stt_openai_url, args.languages)

    if args.tts_voices:
        # If TTS_VOICES is set, use that
        tts_voices = create_tts_voices(args.tts_models, args.tts_voices, args.tts_openai_url, args.languages)
    else:
        # Otherwise, list supported voices via defaults
        tts_voices = await tts_client.list_supported_voices(args.tts_models, args.languages)

    # Log everything available
    if asr_models:
        _logger.info("*** ASR Models ***\n%s", "\n".join(asr_model_to_string(x) for x in asr_models))
    else:
        _logger.warning("No ASR models specified")

    if tts_voices:
        _logger.info("*** TTS Voices ***\n%s", "\n".join(tts_voice_to_string(x) for x in tts_voices))
    else:
        _logger.warning("No TTS models specified")

    # Create server
    server = AsyncServer.from_uri(args.uri)

    # Run server
    _logger.info("Starting server at %s", args.uri)
    await server.run(
        partial(
            OpenAIEventHandler,
            stt_client=stt_client,
            tts_client=tts_client,
            client_lock=asyncio.Lock(),
            asr_models=asr_models,
            stt_temperature=args.stt_temperature,
            tts_voices=tts_voices,
            tts_speed=args.tts_speed
        )
    )

asyncio.run(main())
