from typing import List, Set
from wyoming.info import AsrModel, TtsVoice, Attribution

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

def create_asr_models(stt_models: List[str], stt_url: str, languages: List[str]) -> List[AsrModel]:
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

def create_tts_voices(tts_models: List[str], tts_voices: List[str], tts_url: str, languages: List[str]) -> List[TtsVoiceModel]:
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
#                     logger.info("Found ASR model: %s", model.id)
#
#             except Exception as e:
#                 logger.error("Failed to fetch OpenAI models: %s", e)