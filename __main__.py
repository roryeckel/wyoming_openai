import os
import argparse
import asyncio
import logging
from functools import partial
from wyoming.server import AsyncServer

from . import __version__
from .handler import OpenAIEventHandler
from .compatibility import CustomAsyncOpenAI, create_asr_models, create_tts_voices


def configure_logging(level):
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    logging.basicConfig(level=numeric_level)

async def main():
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
        default=os.getenv("STT_MODELS", 'whisper-1').split(),
        help="List of STT model identifiers"
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
        default=os.getenv("TTS_MODELS", 'tts-1 tts-1-hd').split(),
        help="List of TTS model identifiers"
    )
    parser.add_argument(
        "--tts-voices",
        nargs='+',
        default=os.getenv("TTS_VOICES", 'alloy echo fable onyx nova shimmer').split(),
        help="List of available TTS voices"
    )

    args = parser.parse_args()

    configure_logging(args.log_level)
    _LOGGER = logging.getLogger(__name__)

    asr_models = create_asr_models(args.stt_models, args.stt_openai_url, args.languages)
    tts_voices = create_tts_voices(args.tts_models, args.tts_voices, args.tts_openai_url, args.languages)

    if not asr_models:
        _LOGGER.warning("No ASR models specified")
    if not tts_voices:
        _LOGGER.warning("No TTS models models specified")

    # Create server
    server = AsyncServer.from_uri(args.uri)

    # Create clients
    stt_client = CustomAsyncOpenAI(api_key=args.stt_openai_key, base_url=args.stt_openai_url)
    tts_client = CustomAsyncOpenAI(api_key=args.tts_openai_key, base_url=args.tts_openai_url)
    client_lock = asyncio.Lock()

    # Run server
    _LOGGER.info("Starting server at %s", args.uri)
    await server.run(
        partial(
            OpenAIEventHandler,
            stt_client=stt_client,
            tts_client=tts_client,
            client_lock=client_lock,
            asr_models=asr_models,
            tts_voices=tts_voices
        )
    )

asyncio.run(main())