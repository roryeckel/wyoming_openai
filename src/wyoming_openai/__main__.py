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
from .const import __version__
from .handler import OpenAIEventHandler


def configure_logging(level):
    """Configure logging based on a string level."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    logging.basicConfig(level=numeric_level)

async def main():
    """Main entry point for the Wyoming OpenAI server."""
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
        default=os.getenv("STT_MODELS", '').split(),
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
    parser.add_argument(
        "--stt-prompt",
        default=os.getenv("STT_PROMPT", None),
        help="Optional prompt for STT requests"
    )
    parser.add_argument(
        "--stt-streaming-models",
        nargs="+",
        default=os.getenv("STT_STREAMING_MODELS", '').split(),
        help="Space-separated list of STT model names that support streaming (e.g. gpt-4o-transcribe)"
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
        default=os.getenv("TTS_MODELS", '').split(),
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
    parser.add_argument(
        "--tts-instructions",
        default=os.getenv("TTS_INSTRUCTIONS", None),
        help="Optional instructions for TTS requests"
    )

    args = parser.parse_args()

    configure_logging(args.log_level)
    _logger = logging.getLogger(__name__)

    _logger.info("Starting Wyoming OpenAI %s", __version__)

    # Create STT factory and client
    if args.stt_backend is None:
        _logger.debug("STT backend is None, autodetecting...")
        stt_factory = CustomAsyncOpenAI.create_autodetected_factory()
    else:
        stt_factory = CustomAsyncOpenAI.create_backend_factory(args.stt_backend)

    stt_client = await stt_factory(api_key=args.stt_openai_key, base_url=args.stt_openai_url)
    _logger.debug("Detected STT backend: %s", stt_client.backend)

    # Create TTS factory and client
    if args.tts_backend is None:
        _logger.debug("TTS backend is None, autodetecting...")
        tts_factory = CustomAsyncOpenAI.create_autodetected_factory()
    else:
        tts_factory = CustomAsyncOpenAI.create_backend_factory(args.tts_backend)

    tts_client = await tts_factory(api_key=args.tts_openai_key, base_url=args.tts_openai_url)
    _logger.debug("Detected TTS backend: %s", tts_client.backend)

    # Use clients in async context managers
    async with stt_client, tts_client:

        asr_programs = create_asr_programs(
            args.stt_models,
            args.stt_streaming_models,
            args.stt_openai_url,
            args.languages
        )

        if args.tts_voices:
            # If TTS_VOICES is set, use that
            tts_voices = create_tts_voices(args.tts_models, args.tts_voices, args.tts_openai_url, args.languages)
        else:
            # Otherwise, list supported voices via defaults
            tts_voices = await tts_client.list_supported_voices(args.tts_models, args.languages)

        tts_programs = create_tts_programs(tts_voices)

        # Ensure at least one model is specified
        if not asr_programs and not tts_programs:
            _logger.error("No STT or TTS models specified. Exiting.")
            return

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
                _logger.info("*** Streaming ASR Models ***\n%s", "\n".join(asr_model_to_string(x, is_streaming=True) for x in streaming_asr_models_for_logging))
            else:
                _logger.info("No Streaming ASR models specified")

            if non_streaming_asr_models_for_logging:
                _logger.info("*** Non-Streaming ASR Models ***\n%s", "\n".join(asr_model_to_string(x, is_streaming=False) for x in non_streaming_asr_models_for_logging))
            else:
                _logger.info("No Non-Streaming ASR models specified")
        else:
            _logger.warning("No ASR models specified")

        if tts_programs:
            all_tts_voices = [voice for prog in tts_programs for voice in prog.voices]
            _logger.info("*** TTS Voices ***\n%s", "\n".join(tts_voice_to_string(x) for x in all_tts_voices))
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
                client_lock=asyncio.Lock(),
                stt_temperature=args.stt_temperature,
                tts_speed=args.tts_speed,
                tts_instructions=args.tts_instructions,
                stt_prompt=args.stt_prompt
            )
        )

asyncio.run(main())
