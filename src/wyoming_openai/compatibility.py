import logging
from enum import Enum
from urllib.parse import urlparse

from openai import AsyncOpenAI
from wyoming.info import AsrModel, AsrProgram, Attribution, Info, TtsProgram, TtsVoice

from .const import ATTRIBUTION_NAME, ATTRIBUTION_URL, DEFAULT_OPENAI_BASE_URL, __version__

_LOGGER = logging.getLogger(__name__)


class OpenAIBackend(Enum):
    OPENAI = 0
    SPEACHES = 1
    KOKORO_FASTAPI = 2
    LOCALAI = 3


class TtsVoiceModel(TtsVoice):
    """TtsVoice with an associated model name."""

    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name


def _ordered_unique(primary: list[str], secondary: list[str]) -> list[str]:
    """Merge two lists preserving order, primary first, no duplicates."""
    seen: set[str] = set()
    result: list[str] = []
    for name in (*primary, *secondary):
        if name not in seen:
            result.append(name)
            seen.add(name)
    return result


def _make_attribution(url: str, streaming: bool = False) -> Attribution:
    name = f"{ATTRIBUTION_NAME} (Streaming)" if streaming else ATTRIBUTION_NAME
    return Attribution(name=name, url=url)


def create_asr_programs(
    stt_models: list[str], stt_streaming_models: list[str], stt_url: str, languages: list[str]
) -> list[AsrProgram]:
    ordered = _ordered_unique(stt_streaming_models, stt_models)
    streaming_set = set(stt_streaming_models)

    all_models = [
        AsrModel(
            name=m, description=m, attribution=_make_attribution(stt_url),
            installed=True, languages=languages, version=None,
        )
        for m in ordered
    ]

    streaming = [m for m in all_models if m.name in streaming_set]
    non_streaming = [m for m in all_models if m.name not in streaming_set]

    programs: list[AsrProgram] = []
    if streaming:
        programs.append(AsrProgram(
            name="openai-streaming", description="OpenAI (Streaming)",
            attribution=_make_attribution(stt_url, streaming=True),
            installed=True, version=__version__, models=streaming,
            supports_transcript_streaming=True,
        ))
    if non_streaming:
        programs.append(AsrProgram(
            name="openai", description="OpenAI (Non-Streaming)",
            attribution=_make_attribution(stt_url),
            installed=True, version=__version__, models=non_streaming,
            supports_transcript_streaming=False,
        ))
    return programs


def create_tts_voices(
    tts_models: list[str], tts_streaming_models: list[str], tts_voices: list[str], tts_url: str, languages: list[str]
) -> list[TtsVoiceModel]:
    ordered = _ordered_unique(tts_streaming_models, tts_models)
    return [
        TtsVoiceModel(
            name=v, description=f"{v} ({m})", model_name=m,
            attribution=_make_attribution(tts_url),
            installed=True, languages=languages, version=None,
        )
        for m in ordered for v in tts_voices
    ]


def create_tts_programs(
    tts_voices: list[TtsVoiceModel], tts_streaming_models: list[str] | None = None,
) -> list[TtsProgram]:
    if not tts_voices:
        return []
    streaming_set = set(tts_streaming_models or [])
    streaming = [v for v in tts_voices if v.model_name in streaming_set]
    non_streaming = [v for v in tts_voices if v.model_name not in streaming_set]

    programs: list[TtsProgram] = []
    if streaming:
        programs.append(TtsProgram(
            name="openai-streaming", description="OpenAI (Streaming)",
            attribution=_make_attribution(ATTRIBUTION_URL, streaming=True),
            installed=True, version=__version__, voices=streaming,
            supports_synthesize_streaming=True,
        ))
    if non_streaming:
        programs.append(TtsProgram(
            name="openai", description="OpenAI (Non-Streaming)",
            attribution=_make_attribution(ATTRIBUTION_URL),
            installed=True, version=__version__, voices=non_streaming,
            supports_synthesize_streaming=False,
        ))
    return programs


def create_info(asr_programs: list[AsrProgram], tts_programs: list[TtsProgram]) -> Info:
    return Info(asr=asr_programs, tts=tts_programs)


def asr_model_to_string(model: AsrModel, is_streaming: bool = False) -> str:
    return (
        f"ASR Model:\n"
        f"  Name: {model.name}\n"
        f"  Description: {model.description}\n"
        f"  Attribution: {model.attribution.name} - {model.attribution.url}\n"
        f"  Languages: {', '.join(model.languages)}\n"
        f"  Supports Streaming: {is_streaming}\n"
        f"  Installed: {'Yes' if model.installed else 'No'}\n"
        f"  Version: {model.version}"
    )


def tts_voice_to_string(voice: TtsVoiceModel) -> str:
    return (
        f"TTS Voice Model:\n"
        f"  Name: {voice.name}\n"
        f"  Description: {voice.description}\n"
        f"  Model Name: {voice.model_name}\n"
        f"  Attribution: {voice.attribution.name} - {voice.attribution.url}\n"
        f"  Installed: {'Yes' if voice.installed else 'No'}\n"
        f"  Languages: {', '.join(voice.languages)}\n"
        f"  Version: {voice.version}"
    )


# --- Client ---

_OPENAI_HOSTNAME = urlparse(DEFAULT_OPENAI_BASE_URL).hostname


class CustomAsyncOpenAI(AsyncOpenAI):
    """AsyncOpenAI wrapper with optional auth and backend detection."""

    def __init__(self, *args, **kwargs):
        if not kwargs.get("api_key"):
            kwargs["api_key"] = ""
        if not kwargs.get("base_url"):
            kwargs["base_url"] = DEFAULT_OPENAI_BASE_URL
        self.backend: OpenAIBackend = kwargs.pop("backend", OpenAIBackend.OPENAI)
        super().__init__(*args, **kwargs)

    # --- Backend detection ---

    async def _probe(self, path: str, check=None) -> bool:
        try:
            resp = await self._client.get(path)
            resp.raise_for_status()
            return check(resp) if check else True
        except Exception:
            return False

    async def autodetect_backend(self) -> None:
        """Detect backend type from health/probe endpoints."""
        url = str(getattr(self, "base_url", ""))
        try:
            if urlparse(url).hostname == _OPENAI_HOSTNAME:
                self.backend = OpenAIBackend.OPENAI
                return
        except Exception:
            pass

        if await self._is_localai():
            self.backend = OpenAIBackend.LOCALAI
        elif await self._is_speaches():
            self.backend = OpenAIBackend.SPEACHES
        elif await self._is_kokoro_fastapi():
            self.backend = OpenAIBackend.KOKORO_FASTAPI
        else:
            self.backend = OpenAIBackend.OPENAI

    # --- Voice listing ---

    async def list_voices_for_model(self, model_name: str) -> list[str]:
        """Fetch available voices for a model based on the detected backend."""
        if self.backend == OpenAIBackend.OPENAI:
            return ["alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"]
        if self.backend == OpenAIBackend.LOCALAI:
            return [model_name]
        if self.backend == OpenAIBackend.KOKORO_FASTAPI:
            return await self._fetch_kokoro_voices()
        if self.backend == OpenAIBackend.SPEACHES:
            return await self._fetch_speaches_voices(model_name)
        _LOGGER.warning("Unknown backend: %s", self.backend)
        return []

    async def _fetch_kokoro_voices(self) -> list[str]:
        resp = await self._client.get("/audio/voices")
        resp.raise_for_status()
        return resp.json().get("voices", [])

    async def _fetch_speaches_voices(self, model_name: str) -> list[str]:
        # Try new endpoint first
        try:
            resp = await self._client.get(f"/models/{model_name}")
            resp.raise_for_status()
            data = resp.json()
            if "voices" in data:
                return [v["name"] for v in data["voices"]]
        except Exception:
            _LOGGER.debug("Failed /models/%s, trying legacy endpoint", model_name)
        # Legacy fallback
        resp = await self._client.get("/audio/speech/voices", params={"model_id": model_name})
        resp.raise_for_status()
        return [v["voice_id"] for v in resp.json()]

    async def list_supported_voices(
        self, model_names: list[str], streaming_model_names: list[str], languages: list[str]
    ) -> list[TtsVoiceModel]:
        """Fetch voices for all models and return as TtsVoiceModel list."""
        ordered = _ordered_unique(streaming_model_names, model_names)
        result: list[TtsVoiceModel] = []
        for model in ordered:
            if self.backend == OpenAIBackend.OPENAI:
                voices = await self.list_openai_voices()
            elif self.backend == OpenAIBackend.SPEACHES:
                voices = await self._list_speaches_voices(model)
            elif self.backend == OpenAIBackend.KOKORO_FASTAPI:
                voices = await self._list_kokoro_fastapi_voices()
            elif self.backend == OpenAIBackend.LOCALAI:
                voices = await self._list_localai_voices(model)
            else:
                _LOGGER.warning("Unknown backend: %s", self.backend)
                continue
            result.extend(create_tts_voices(
                tts_models=[model], tts_streaming_models=streaming_model_names,
                tts_voices=voices, tts_url=str(self.base_url), languages=languages,
            ))
        return result

    # Keep old aliases used by tests
    async def list_openai_voices(self) -> list[str]:
        return ["alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"]

    async def _is_kokoro_fastapi(self) -> bool:
        return await self._probe("/test", lambda r: r.json().get("status") == "ok")

    async def _is_localai(self) -> bool:
        return await self._probe("/readyz")

    async def _is_speaches(self) -> bool:
        return await self._probe("../../health", lambda r: r.text == "OK")

    async def _list_kokoro_fastapi_voices(self) -> list[str]:
        if self.backend != OpenAIBackend.KOKORO_FASTAPI:
            return []
        return await self._fetch_kokoro_voices()

    async def _list_speaches_voices(self, model_name: str) -> list[str]:
        if self.backend != OpenAIBackend.SPEACHES:
            return []
        return await self._fetch_speaches_voices(model_name)

    async def _list_localai_voices(self, model_name: str) -> list[str]:
        return [model_name]

    @classmethod
    def _is_openai_domain(cls, base_url: str | None) -> bool:
        if not base_url:
            return False
        try:
            return urlparse(str(base_url)).hostname == _OPENAI_HOSTNAME
        except Exception:
            return False

    @classmethod
    def create_autodetected_factory(cls):
        async def factory(*args, **kwargs):
            client = cls(*args, **kwargs)
            await client.autodetect_backend()
            return client
        return factory

    @classmethod
    def create_backend_factory(cls, backend: OpenAIBackend):
        async def factory(*args, **kwargs):
            return cls(*args, **kwargs, backend=backend)
        return factory
