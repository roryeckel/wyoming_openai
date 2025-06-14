import logging
from enum import Enum
from typing import override

from openai import AsyncOpenAI
from wyoming.info import AsrModel, Attribution, TtsVoice

_LOGGER = logging.getLogger(__name__)


class TtsVoiceModel(TtsVoice):
    """
    A subclass of TtsVoice from the Wyoming Protocol representing a text-to-speech voice with an associated model name.

    Attributes:
        model_name (str): The name of the underlying text-to-speech model.
    """
    def __init__(self, model_name: str, *args, **kwargs):
        """
        Initializes a TtsVoiceModel instance.

        Args:
            model_name (str): The name of the text-to-speech model.
            *args: Variable length argument list for superclass initialization.
            **kwargs: Arbitrary keyword arguments for superclass initialization.
        """
        super().__init__(*args, **kwargs)
        self.model_name = model_name

def create_asr_models(stt_models: list[str], stt_url: str, languages: list[str]) -> list[AsrModel]:
    """
    Creates a list of ASR (Automatic Speech Recognition) models in the Wyoming Protocol format.

    Args:
        stt_models (List[str]): A list of STT model names.
        stt_url (str): The URL for the STT service attribution.
        languages (List[str]): A list of supported languages.

    Returns:
        List[AsrModel]: A list of AsrModel instances.
    """
    asr_models = []
    for model_name in stt_models:
        asr_models.append(AsrModel(
            name=model_name,
            description=model_name,
            attribution=Attribution(
                name="OpenAI Wyoming Proxy",
                url=stt_url
            ),
            installed=True,
            languages=languages,
            version=None
        ))
    return asr_models

def create_tts_voices(tts_models: list[str], tts_voices: list[str], tts_url: str, languages: list[str]) -> list[TtsVoiceModel]:
    """
    Creates a list of TTS (Text-to-Speech) voice models in the Wyoming Protocol format.

    Args:
        tts_models (List[str]): A list of TTS model names.
        tts_voices (List[str]): A list of voice identifiers.
        tts_url (str): The URL for the TTS service attribution.
        languages (List[str]): A list of supported languages.

    Returns:
        List[TtsVoiceModel]: A list of TtsVoiceModel instances.
    """
    voices = []
    for model_name in tts_models:
        for voice in tts_voices:
            voices.append(TtsVoiceModel(
                name=voice,
                description=f"{voice} ({model_name})",
                model_name=model_name,
                attribution=Attribution(
                    name="OpenAI Wyoming Proxy",
                    url=tts_url
                ),
                installed=True,
                languages=languages,
                version=None
            ))
    return voices

def asr_model_to_string(asr_model: AsrModel) -> str:
    """
    Converts an AsrModel instance to a human-readable string representation.

    Args:
        asr_model (AsrModel): The ASR model instance to convert.

    Returns:
        str: A human-readable string representation of the ASR model.
    """
    return (
        f"ASR Model:\n"
        f"  Name: {asr_model.name}\n"
        f"  Description: {asr_model.description}\n"
        f"  Attribution: {asr_model.attribution.name} - {asr_model.attribution.url}\n"
        f"  Installed: {'Yes' if asr_model.installed else 'No'}\n"
        f"  Languages: {', '.join(asr_model.languages)}\n"
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
    OPENAI = 0 # "Official"
    SPEACHES = 1
    KOKORO_FASTAPI = 2

class CustomAsyncOpenAI(AsyncOpenAI):
    """
    Custom implementation of OpenAI's AsyncOpenAI class to handle API key authentication being optional.
    """
    def __init__(self, *args, **kwargs):
        if "api_key" not in kwargs or not kwargs["api_key"]:
            kwargs["api_key"] = ""
        self.backend: OpenAIBackend = kwargs.pop("backend", OpenAIBackend.OPENAI)
        super().__init__(*args, **kwargs)

    @property
    @override
    def auth_headers(self) -> dict[str, str]:
        """
        Override the auth_headers property to remove the Authorization header if no API key is provided.
        """
        super_headers = super().auth_headers
        if not self.api_key:
            del super_headers["Authorization"]
        return super_headers

    # OpenAI

    async def list_openai_voices(self) -> list[str]:
        """
        Not official implemented by OpenAI, hard-coded.
        https://platform.openai.com/docs/guides/text-to-speech/voice-options
        """
        return ['alloy', 'ash', 'coral', 'echo', 'fable', 'onyx', 'nova', 'sage', 'shimmer']

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
        Example: ["af_sky"]
        """
        if self.backend != OpenAIBackend.KOKORO_FASTAPI:
            _LOGGER.debug("Skipping /audio/voices request because backend is not KOKORO_FASTAPI")
            return []

        try:
            response = await self._client.get("/audio/voices")
            response.raise_for_status()
            return response.json().get("voices", [])
        except Exception as e:
            _LOGGER.exception(e, "Failed to fetch /audio/voices")
            raise

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
            _LOGGER.debug("Skipping /models/{model_name} request because backend is not SPEACHES", model_name=model_name)
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
        except Exception as e:
            _LOGGER.exception(e, "Failed to fetch /models/%s, checking legacy endpoint...")

        # LEGACY Endpoint
        # Example: [{"model_id": "hexgrad/Kokoro-82M", "voice_id": "af_sky"}]
        try:
            response = await self._client.get("/audio/speech/voices", params={"model_id": model_name})
            response.raise_for_status()
            result = response.json()
            return [voice["voice_id"] for voice in result]
        except Exception as e:
            _LOGGER.exception(e, "Failed to fetch /audio/speech/voices")
            raise

    # Unified API

    async def list_supported_voices(self, model_names: str | list[str], languages: list[str]) -> list[TtsVoiceModel]:
        """
        Fetches the available voices via unofficial specs.
        Note: this is not the list of CONFIGURED voices.
        """
        if isinstance(model_names, str):
            model_names = [model_names]

        tts_voice_models = []
        for model_name in model_names:
            if self.backend == OpenAIBackend.OPENAI:
                tts_voices = await self.list_openai_voices()
            elif self.backend == OpenAIBackend.SPEACHES:
                tts_voices = await self._list_speaches_voices(model_name)
            elif self.backend == OpenAIBackend.KOKORO_FASTAPI:
                tts_voices = await self._list_kokoro_fastapi_voices()
            else:
                _LOGGER.warning("Unknown backend: %s", self.backend)
                continue

            # Create TTS voices in Wyoming Protocol format
            tts_voice_models.extend(create_tts_voices(
                tts_models=[model_name],
                tts_voices=tts_voices,
                tts_url=str(self.base_url),
                languages=languages
            ))
        return tts_voice_models

    @classmethod
    def create_autodetected_factory(cls):
        """
        Create a factory that autodetects the backend type.
        This factory will initialize the client and set the backend based on the detected type.
        """
        async def factory(*args, **kwargs):
            client = cls(*args, **kwargs)
            if await client._is_speaches():
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
