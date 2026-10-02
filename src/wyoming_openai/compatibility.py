import logging
from collections import Counter
from collections.abc import Mapping
from enum import Enum
from urllib.parse import urlparse

from openai import AsyncOpenAI, omit
from wyoming.info import AsrModel, AsrProgram, Attribution, Info, TtsProgram, TtsVoice

from .const import (
    ATTRIBUTION_NAME_MODEL,
    ATTRIBUTION_NAME_PROGRAM,
    ATTRIBUTION_NAME_PROGRAM_STREAMING,
    ATTRIBUTION_URL,
    DEFAULT_OPENAI_BASE_URL,
    __version__,
)

_LOGGER = logging.getLogger(__name__)


class TtsVoiceModel(TtsVoice):
    """
    A subclass of TtsVoice from the Wyoming Protocol representing a text-to-speech voice with an associated model name.

    Attributes:
        model_name (str): The name of the underlying text-to-speech model.
        backend_voice_name (str): The raw voice identifier expected by the backend.
    """

    def __init__(self, model_name: str, *args, **kwargs):
        """
        Initializes a TtsVoiceModel instance.

        Args:
            model_name (str): The name of the text-to-speech model.
            *args: Variable length argument list for superclass initialization.
            **kwargs: Arbitrary keyword arguments for superclass initialization.
        """
        backend_voice_name = kwargs.pop("backend_voice_name", None)
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.backend_voice_name = self.name if backend_voice_name is None else backend_voice_name


def _get_ordered_unique_model_names(*model_groups: list[str]) -> list[str]:
    """Return unique model names while preserving group and input order."""
    seen = set()
    ordered_models = []

    for model_group in model_groups:
        for model_name in model_group:
            if model_name not in seen:
                ordered_models.append(model_name)
                seen.add(model_name)

    return ordered_models


def _get_ordered_unique_models(models: list[str], streaming_models: list[str]) -> list[str]:
    """Return streaming-first model names while preserving input order."""
    return _get_ordered_unique_model_names(streaming_models, models)


def _create_tts_voice_models(
    model_voice_pairs: list[tuple[str, str]],
    tts_url: str,
    languages: list[str],
) -> list[TtsVoiceModel]:
    """Create Wyoming TTS voice models with collision-aware public names."""
    if not model_voice_pairs:
        return []

    raw_name_counts = Counter(raw_voice_name for _, raw_voice_name in model_voice_pairs)
    base_public_names = [
        raw_voice_name if raw_name_counts[raw_voice_name] == 1 else f"{raw_voice_name} ({model_name})"
        for model_name, raw_voice_name in model_voice_pairs
    ]
    public_name_counts = Counter(base_public_names)
    duplicate_public_name_counts: Counter[str] = Counter()

    voices = []
    for (model_name, raw_voice_name), public_name in zip(model_voice_pairs, base_public_names, strict=False):
        if public_name_counts[public_name] > 1:
            duplicate_public_name_counts[public_name] += 1
            public_name = f"{public_name} [{duplicate_public_name_counts[public_name]}]"

        voices.append(
            TtsVoiceModel(
                name=public_name,
                description=public_name,
                model_name=model_name,
                backend_voice_name=raw_voice_name,
                attribution=Attribution(name=ATTRIBUTION_NAME_MODEL, url=tts_url),
                installed=True,
                languages=languages,
                version=None,
            )
        )

    return voices


def _extract_voice_names(result: object) -> list[str]:
    """Extract backend voice identifiers from common voice-list response shapes."""
    voices = result.get("voices", []) if isinstance(result, dict) else result

    if not isinstance(voices, list):
        _LOGGER.warning("Unexpected voices response shape: %s", type(voices).__name__)
        return []

    voice_names = []
    for voice in voices:
        if isinstance(voice, str):
            voice_names.append(voice)
            continue

        if isinstance(voice, dict):
            voice_name = voice.get("id") or voice.get("name")
            if isinstance(voice_name, str):
                voice_names.append(voice_name)
                continue

        _LOGGER.warning("Skipping unexpected voice entry: %r", voice)

    return voice_names


def create_asr_programs(
    stt_models: list[str],
    stt_streaming_models: list[str],
    stt_url: str,
    languages: list[str],
    stt_realtime_models: list[str] | None = None,
) -> list[AsrProgram]:
    """
    Creates a list of ASR programs, separating models based on streaming support.

    Args:
        stt_models (list[str]): List of STT model identifiers.
        stt_streaming_models (list[str]): List of STT models identifiers that support streaming.
        stt_url (str): The URL for the STT service attribution.
        languages (list[str]): A list of supported languages.
        stt_realtime_models (list[str] | None): List of STT model identifiers that use Realtime transcription.

    Returns:
        list[AsrProgram]: A list of Wyoming ASR programs.
    """
    if stt_realtime_models is None:
        stt_realtime_models = []

    # Create ordered list: realtime models first, then response-streaming models, then non-streaming models.
    ordered_models = _get_ordered_unique_model_names(stt_realtime_models, stt_streaming_models, stt_models)

    all_asr_models = []
    for model_name in ordered_models:
        all_asr_models.append(
            AsrModel(
                name=model_name,
                description=model_name,
                attribution=Attribution(name=ATTRIBUTION_NAME_MODEL, url=stt_url),
                installed=True,
                languages=languages,
                version=None,
            )
        )

    realtime_asr_models = []
    streaming_asr_models = []
    non_streaming_asr_models = []

    for model in all_asr_models:
        if model.name in stt_realtime_models:
            realtime_asr_models.append(model)
        elif model.name in stt_streaming_models:
            streaming_asr_models.append(model)
        else:
            non_streaming_asr_models.append(model)

    asr_programs = []

    if realtime_asr_models:
        asr_programs.append(
            AsrProgram(
                name="openai-realtime",
                description="OpenAI (Realtime)",
                attribution=Attribution(name=ATTRIBUTION_NAME_PROGRAM_STREAMING, url=stt_url),
                installed=True,
                version=__version__,
                models=realtime_asr_models,
                supports_transcript_streaming=True,
            )
        )

    if streaming_asr_models:
        asr_programs.append(
            AsrProgram(
                name="openai-streaming",
                description="OpenAI (Streaming)",
                attribution=Attribution(name=ATTRIBUTION_NAME_PROGRAM_STREAMING, url=stt_url),
                installed=True,
                version=__version__,
                models=streaming_asr_models,
                supports_transcript_streaming=True,
            )
        )

    if non_streaming_asr_models:
        asr_programs.append(
            AsrProgram(
                name="openai",
                description="OpenAI (Non-Streaming)",
                attribution=Attribution(name=ATTRIBUTION_NAME_PROGRAM, url=stt_url),
                installed=True,
                version=__version__,
                models=non_streaming_asr_models,
                supports_transcript_streaming=False,
            )
        )

    return asr_programs


def create_tts_voices(
    tts_models: list[str], tts_streaming_models: list[str], tts_voices: list[str], tts_url: str, languages: list[str]
) -> list[TtsVoiceModel]:
    """
    Creates a list of TTS (Text-to-Speech) voice models in the Wyoming Protocol format.
    Uses streaming models as fallback if regular models not specified (consistent with ASR behavior).

    Args:
        tts_models (list[str]): A list of TTS model identifiers.
        tts_streaming_models (list[str]): A list of TTS streaming model identifiers.
        tts_voices (list[str]): A list of voice identifiers.
        tts_url (str): The URL for the TTS service attribution.
        languages (list[str]): A list of supported languages.

    Returns:
        list[TtsVoiceModel]: A list of Wyoming TtsVoiceModel instances.
    """
    ordered_models = _get_ordered_unique_models(tts_models, tts_streaming_models)
    model_voice_pairs = [(model_name, voice_name) for model_name in ordered_models for voice_name in tts_voices]
    return _create_tts_voice_models(model_voice_pairs, tts_url, languages)


def apply_tts_voice_labels(tts_voices: list[TtsVoiceModel], voice_labels: Mapping[str, str]) -> None:
    """
    Set the description of every TTS voice that has a configured display name.

    Wyoming clients such as Home Assistant show a voice's description instead of its name in their voice pickers.
    Only the description changes; the voice name that clients send back and the backend voice name stay as they are.

    A label is looked up by the public voice name first (for example "alloy (tts-1)") and then by the raw backend
    voice name (for example "alloy"). One label can therefore cover a voice offered by several models while a
    more specific label can still tell them apart. Labels that match no voice are logged and ignored.

    Args:
        tts_voices (list[TtsVoiceModel]): Voices to update in place.
        voice_labels (Mapping[str, str]): Display names keyed by voice name.
    """
    known_names: set[str] = set()
    for voice in tts_voices:
        known_names.update((voice.name, voice.backend_voice_name))
        for key in (voice.name, voice.backend_voice_name):
            if key in voice_labels:
                voice.description = voice_labels[key]
                break

    for key in voice_labels:
        if key not in known_names:
            _LOGGER.warning("Ignoring TTS voice label for unknown voice '%s'", key)


def create_tts_programs(
    tts_voices: list[TtsVoiceModel],
    tts_streaming_models: list[str] | None = None,
) -> list[TtsProgram]:
    """
    Create TTS programs from a list of voices, separating voices based on streaming model support.

    Args:
        tts_voices (list[TtsVoiceModel]): A list of TTS voice models.
        tts_streaming_models (list[str]): List of TTS model names that support streaming.

    Returns:
        list[TtsProgram]: A list of Wyoming TTS programs.
    """
    if not tts_voices:
        return []

    if tts_streaming_models is None:
        tts_streaming_models = []

    # Separate streaming and non-streaming voices based on their models
    streaming_tts_voices = []
    non_streaming_tts_voices = []

    for voice in tts_voices:
        if voice.model_name in tts_streaming_models:
            streaming_tts_voices.append(voice)
        else:
            non_streaming_tts_voices.append(voice)

    programs = []

    if streaming_tts_voices:
        programs.append(
            TtsProgram(
                name="openai-streaming",
                description="OpenAI (Streaming)",
                attribution=Attribution(
                    name=ATTRIBUTION_NAME_PROGRAM_STREAMING,
                    url=ATTRIBUTION_URL,
                ),
                installed=True,
                version=__version__,
                voices=streaming_tts_voices,
                supports_synthesize_streaming=True,
            )
        )

    if non_streaming_tts_voices:
        programs.append(
            TtsProgram(
                name="openai",
                description="OpenAI (Non-Streaming)",
                attribution=Attribution(
                    name=ATTRIBUTION_NAME_PROGRAM,
                    url=ATTRIBUTION_URL,
                ),
                installed=True,
                version=__version__,
                voices=non_streaming_tts_voices,
                supports_synthesize_streaming=False,
            )
        )

    return programs


def create_info(asr_programs: list[AsrProgram], tts_programs: list[TtsProgram]) -> Info:
    """
    Create Wyoming info object.

    Args:
        asr_programs (list[AsrProgram]): A list of ASR programs.
        tts_programs (list[TtsProgram]): A list of TTS programs.

    Returns:
        Info: A Wyoming info object.
    """
    return Info(asr=asr_programs, tts=tts_programs)


def asr_model_to_string(asr_model: AsrModel, is_streaming: bool = False) -> str:
    """
    Converts an AsrModel instance to a human-readable string representation.

    Args:
        asr_model (AsrModel): The ASR model instance to convert.
        is_streaming (bool): Indicates whether the model is streaming.

    Returns:
        str: A human-readable string representation of the ASR model.
    """
    return (
        f"ASR Model:\n"
        f"  Name: {asr_model.name}\n"
        f"  Description: {asr_model.description}\n"
        f"  Attribution: {asr_model.attribution.name} - {asr_model.attribution.url}\n"
        f"  Languages: {', '.join(asr_model.languages)}\n"
        f"  Supports Streaming: {is_streaming}\n"
        f"  Installed: {'Yes' if asr_model.installed else 'No'}\n"
        f"  Version: {asr_model.version}"
    )


def tts_voice_to_string(tts_voice_model: TtsVoiceModel) -> str:
    """
    Converts a TtsVoiceModel instance to a human-readable string representation.

    Args:
        tts_voice_model (TtsVoiceModel): The TTS voice model instance to convert.

    Returns:
        str: A human-readable string representation of the TTS voice model.
    """
    return (
        f"TTS Voice Model:\n"
        f"  Name: {tts_voice_model.name}\n"
        f"  Description: {tts_voice_model.description}\n"
        f"  Model Name: {tts_voice_model.model_name}\n"
        f"  Attribution: {tts_voice_model.attribution.name} - {tts_voice_model.attribution.url}\n"
        f"  Installed: {'Yes' if tts_voice_model.installed else 'No'}\n"
        f"  Languages: {', '.join(tts_voice_model.languages)}\n"
        f"  Version: {tts_voice_model.version}"
    )


# https://github.com/speaches-ai/speaches/issues/266
# async def get_openai_models(
#     api_key: str,
#     base_urls: Set[str]
# ):
# """
# Asynchronously fetches OpenAI models from given base URLs.

# Args:
#     api_key (str): The API key for accessing OpenAI services.
#     base_urls (Set[str]): A set of base URLs to fetch the models from.
# """
#     logger = logging.getLogger(__name__)
#     logger.debug("Fetching OpenAI models...")
#
#     for base_url in base_urls:
#         async with AsyncOpenAI(api_key=api_key, base_url=base_url) as client:
#             try:
#                 models_response = await client.models.list()
#
#                 for model in models_response.data:
#                     logger.info("Found model: %s", model.id)
#
#             except Exception as e:
#                 logger.error("Failed to fetch OpenAI models: %s", e)


class OpenAIBackend(Enum):
    """Enum for different unofficial backends."""

    OPENAI = 0  # "Official"
    SPEACHES = 1
    KOKORO_FASTAPI = 2
    LOCALAI = 3


class CustomAsyncOpenAI(AsyncOpenAI):
    """
    Custom implementation of OpenAI's AsyncOpenAI class to handle API key authentication being optional.
    """

    def __init__(self, *args, **kwargs):
        if "api_key" not in kwargs or not kwargs["api_key"]:
            kwargs["api_key"] = ""
            kwargs.setdefault("_enforce_credentials", False)
        if not kwargs.get("base_url"):
            kwargs["base_url"] = DEFAULT_OPENAI_BASE_URL
        self.backend: OpenAIBackend = kwargs.pop("backend", OpenAIBackend.OPENAI)
        try:
            super().__init__(*args, **kwargs)
        except TypeError as exc:
            if "_enforce_credentials" not in kwargs or "_enforce_credentials" not in str(exc):
                raise

            kwargs.pop("_enforce_credentials", None)
            super().__init__(*args, **kwargs)

    async def _prepare_options(self, options):
        # Local keyless backends (Speaches, LocalAI, Kokoro) don't require auth.
        # Without an api_key the SDK refuses to send the request unless Authorization
        # is explicitly omitted via the Omit sentinel on the per-request headers.
        options = await super()._prepare_options(options)
        if not self.api_key:
            headers = dict(options.headers or {})
            headers.setdefault("Authorization", omit)
            options.headers = headers
        return options

    # OpenAI

    async def list_openai_voices(self) -> list[str]:
        """
        Not official implemented by OpenAI, hard-coded.
        https://platform.openai.com/docs/guides/text-to-speech/voice-options
        """
        return ["alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"]

    # Kokoro-FastAPI

    async def _is_kokoro_fastapi(self) -> bool:
        """
        Checks if the backend is Kokoro-FastAPI by sending a request to /test
        Example Response: { "status": "ok" }
        """
        try:
            response = await self._client.get("/test")
            response.raise_for_status()
            return response.json().get("status", None) == "ok"
        except Exception:
            return False

    async def _list_kokoro_fastapi_voices(self) -> list[str]:
        """
        Fetches the available audio voices from the Kokoro-FastAPI /audio/voices endpoint.
        Caution: This is not a part of official OpenAI spec.
        Example legacy response: ["af_sky", "af_bella", ...]
        Example old response: {"voices": ["af_sky", "af_bella", ...]}
        Example current response: [{"id": "af_sky", "name": "Sky"}, ...]
        """
        if self.backend != OpenAIBackend.KOKORO_FASTAPI:
            _LOGGER.debug("Skipping /audio/voices request because backend is not KOKORO_FASTAPI")
            return []

        try:
            response = await self._client.get("/audio/voices")
            response.raise_for_status()
            return _extract_voice_names(response.json())
        except Exception:
            _LOGGER.exception("Failed to fetch /audio/voices")
            raise

    # LocalAI

    async def _is_localai(self) -> bool:
        """
        Checks if the backend is LocalAI by sending a request to /readyz
        LocalAI returns a 200 OK status on /readyz when ready
        """
        try:
            response = await self._client.get("/readyz")
            response.raise_for_status()
            return True
        except Exception:
            return False

    async def _list_localai_voices(self, model_name: str) -> list[str]:
        """
        LocalAI doesn't require voice specification - the voice is optional.
        Returns the model name as the voice name for compatibility.
        """
        return [model_name]

    # Speaches

    async def _is_speaches(self) -> bool:
        """
        Checks if the backend is Speaches by sending a request to /health
        Example Response: OK
        """
        try:
            response = await self._client.get("../../health")
            response.raise_for_status()
            return response.text == "OK"
        except Exception:
            return False

    async def _list_speaches_voices(self, model_name: str) -> list[str]:
        """
        Fetches the available voices from the Speaches /models/{model_name}
        and optionally falls back to the older /audio/speech/voices endpoint.
        Caution: This is not a part of official OpenAI spec.
        """
        if self.backend != OpenAIBackend.SPEACHES:
            _LOGGER.debug("Skipping /models/%s request because backend is not SPEACHES", model_name)
            return []

        # NEW Endpoint
        # Example: {
        #   "id": "speaches-ai/Kokoro-82M-v1.0-ONNX",
        #   "created": 1749005993,
        #   "object": "model",
        #   "owned_by": "speaches-ai",
        #   "language": [
        #     "multilingual"
        #   ],
        #   "task": "text-to-speech",
        #   "sample_rate": 24000,
        #   "voices": [
        #     {
        #       "name": "af_heart",
        #       "language": "en-us",
        #       "gender": "female"
        #     }
        #   ]
        # }
        try:
            response = await self._client.get(f"/models/{model_name}")
            response.raise_for_status()
            result = response.json()
            if "voices" in result:
                return [voice["name"] for voice in result.get("voices", [])]
        except Exception:
            _LOGGER.exception("Failed to fetch /models/%s, checking legacy endpoint...", model_name)

        # LEGACY Endpoint
        # Example: [{"model_id": "hexgrad/Kokoro-82M", "voice_id": "af_sky"}]
        try:
            response = await self._client.get("/audio/speech/voices", params={"model_id": model_name})
            response.raise_for_status()
            result = response.json()
            return [voice["voice_id"] for voice in result]
        except Exception:
            _LOGGER.exception("Failed to fetch /audio/speech/voices")
            raise

    # Unified API

    async def list_supported_voices(
        self, model_names: list[str], streaming_model_names: list[str], languages: list[str]
    ) -> list[TtsVoiceModel]:
        """
        Fetches the available voices via unofficial specs with streaming model fallback (consistent with ASR behavior).
        Uses streaming models if regular models not specified.
        Note: this is not the list of CONFIGURED voices.
        """
        ordered_models = _get_ordered_unique_models(model_names, streaming_model_names)

        model_voice_pairs: list[tuple[str, str]] = []
        for model_name in ordered_models:
            if self.backend == OpenAIBackend.OPENAI:
                tts_voices = await self.list_openai_voices()
            elif self.backend == OpenAIBackend.SPEACHES:
                tts_voices = await self._list_speaches_voices(model_name)
            elif self.backend == OpenAIBackend.KOKORO_FASTAPI:
                tts_voices = await self._list_kokoro_fastapi_voices()
            elif self.backend == OpenAIBackend.LOCALAI:
                tts_voices = await self._list_localai_voices(model_name)
            else:
                _LOGGER.warning("Unknown backend: %s", self.backend)
                continue

            model_voice_pairs.extend((model_name, voice_name) for voice_name in tts_voices)

        return _create_tts_voice_models(model_voice_pairs, str(self.base_url), languages)

    _OPENAI_HOSTNAME = urlparse(DEFAULT_OPENAI_BASE_URL).hostname

    @classmethod
    def _is_openai_domain(cls, base_url: str | None) -> bool:
        """Check if the base URL points to the official OpenAI API."""
        if not base_url:
            return False
        try:
            return urlparse(str(base_url)).hostname == cls._OPENAI_HOSTNAME
        except Exception:
            return False

    @classmethod
    def create_autodetected_factory(cls):
        """
        Create a factory that autodetects the backend type.
        This factory will initialize the client and set the backend based on the detected type.
        Skips detection probes when the URL is the official OpenAI API domain.
        """

        async def factory(*args, **kwargs):
            client = cls(*args, **kwargs)
            resolved_base_url = str(getattr(client, "base_url", ""))
            if cls._is_openai_domain(resolved_base_url):
                _LOGGER.debug("OpenAI domain detected, skipping backend autodetection")
                client.backend = OpenAIBackend.OPENAI
                return client

            if await client._is_localai():
                client.backend = OpenAIBackend.LOCALAI
            elif await client._is_speaches():
                client.backend = OpenAIBackend.SPEACHES
            elif await client._is_kokoro_fastapi():
                client.backend = OpenAIBackend.KOKORO_FASTAPI
            else:
                client.backend = OpenAIBackend.OPENAI
            return client

        return factory

    @classmethod
    def create_backend_factory(cls, backend: OpenAIBackend):
        """
        Create a factory for a specific backend type.
        """

        async def factory(*args, **kwargs):
            return cls(*args, **kwargs, backend=backend)

        return factory
