from io import BytesIO
from typing import List

from wyoming.info import AsrModel, TtsVoice, Attribution

# def is_matching_model(model_id: str, patterns: List[str]) -> bool:
#     """Check if model ID matches any of the given patterns"""
#     return any(re.match(pattern, model_id, re.IGNORECASE) for pattern in patterns)

# async def get_openai_models(
#     api_key: str, 
#     base_urls: Set[str]
# ):
#     logger = logging.getLogger(__name__)
#     logger.debug("Fetching OpenAI models...")

#     for base_url in base_urls:
#         async with AsyncOpenAI(api_key=api_key, base_url=base_url) as client:
#             try:
#                 models_response = await client.models.list()

#                 for model in models_response.data:
#                     logger.info("Found ASR model: %s", model.id)
#                 else:
#                     logger.debug("Skipping model: %s (no matching patterns)", model.id)

#             except Exception as e:
#                 logger.error("Failed to fetch OpenAI models: %s", e)

def create_asr_models(stt_models: List[str], stt_url: str):
    asr_models = []
    for model in stt_models:
        asr_models.append(AsrModel(
            name=model,
            description=model,
            attribution=Attribution(
                name="OpenAI Compatible",
                url=stt_url
            ),
            installed=True,
            languages=['en'],
            version=None
        ))
    return asr_models

def create_tts_voices(tts_models: List[str], tts_voices: List[str], tts_url: str):
    voices = []
    for model in tts_models:
        for voice in tts_voices:
            voices.append(TtsVoice(
                name=voice,
                description=f"{voice} ({model})",
                attribution=Attribution(
                    name="OpenAI Compatible",
                    url=tts_url
                ),
                installed=True,
                languages=['en'],
                version=None
            ))
    return voices

class NamedBytesIO(BytesIO):
    def __init__(self, *args, **kwargs):
        self._name = kwargs.pop('name', 'audio.wav')
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        return self._name