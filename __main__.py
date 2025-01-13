import argparse
import asyncio
import json
import logging
from functools import partial
from wyoming.server import AsyncServer

from . import __version__
from .handler import OpenAIEventHandler
from .utilities import get_openai_models

def configure_logging(level):
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    logging.basicConfig(level=numeric_level)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="tcp://0.0.0.0:10300", help="This Wyoming Server URI")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")

    # STT configuration
    parser.add_argument("--stt-openai-key", required=True, help="OpenAI API key for speech-to-text")
    parser.add_argument("--stt-openai-url", default="https://api.openai.com/v1", help="Custom OpenAI API base URL for STT")
    parser.add_argument(
        "--stt-patterns", 
        type=json.loads,
        default='["^whisper-", "transcribe", "stt", "speech.*text"]',
        help="JSON list of regex patterns to identify STT models"
    )

    # TTS configuration
    parser.add_argument("--tts-openai-key", required=True, help="OpenAI API key for text-to-speech") 
    parser.add_argument("--tts-openai-url", default="https://api.openai.com/v1", help="Custom OpenAI API base URL for TTS")
    parser.add_argument(
        "--tts-patterns",
        type=json.loads,
        default='["^tts-", "speech.*synthesis", "text.*speech"]',
        help="JSON list of regex patterns to identify TTS models"
    )
    parser.add_argument(
        "--tts-voices",
        type=json.loads,
        default='["alloy", "echo", "fable", "onyx", "nova", "shimmer"]',
        help="JSON list of available TTS voices"
    )

    args = parser.parse_args()

    configure_logging(args.log_level)
    
    asr_models, tts_voices = await get_openai_models(
        args.stt_openai_key,
        args.stt_openai_url,
        args.stt_patterns,
        args.tts_patterns,
        args.tts_voices
    )

    if not asr_models:
        _LOGGER.warning("No ASR models found matching patterns: %s", args.stt_patterns)
        _LOGGER.warning("Will use default 'whisper-1'")
    if not tts_voices:
        _LOGGER.warning("No TTS models found matching patterns: %s", args.tts_patterns)
        _LOGGER.warning("Will use defaults")

    # Create server
    server = AsyncServer.from_uri(args.uri)

    # Run server
    _LOGGER = logging.getLogger(__name__)
    _LOGGER.info("Starting server at %s", args.uri)
    await server.run(
        partial(
            OpenAIEventHandler,
            stt_api_key=args.stt_openai_key,
            stt_base_url=args.stt_openai_url,
            tts_api_key=args.tts_openai_key,
            tts_base_url=args.tts_openai_url,
            client_lock=asyncio.Lock(),
            asr_models=asr_models,
            tts_voices=tts_voices
        )
    )

asyncio.run(main())