from typing import List
from wyoming.info import AsrModel, TtsVoice, Attribution

class TtsVoiceModel(TtsVoice):
    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name


def create_asr_models(stt_models: List[str], stt_url: str) -> List[AsrModel]:
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
            languages=['en'],
            version=None
        ))
    return asr_models

def create_tts_voices(tts_models: List[str], tts_voices: List[str], tts_url: str) -> List[TtsVoiceModel]:
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
                languages=['en'],
                version=None
            ))
    return voices

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

#             except Exception as e:
#                 logger.error("Failed to fetch OpenAI models: %s", e)