from io import BytesIO
import logging
import re
from typing import List, Tuple
from openai import AsyncOpenAI
from wyoming.info import AsrModel, TtsVoice, Attribution

def is_matching_model(model_id: str, patterns: List[str]) -> bool:
    """Check if model ID matches any of the given patterns"""
    return any(re.match(pattern, model_id, re.IGNORECASE) for pattern in patterns)

async def get_openai_models(
    api_key: str, 
    base_url: str,
    stt_patterns: List[str],
    tts_patterns: List[str],
    tts_voices: List[str]
) -> Tuple[List[AsrModel], List[TtsVoice]]:
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    logger = logging.getLogger(__name__)
    logger.debug("Fetching OpenAI models...")

    try:
        models_response = await client.models.list()
        
        asr_models = []
        tts_model_voices = []

        for model in models_response.data:
            if is_matching_model(model.id, stt_patterns):
                asr_models.append(
                    AsrModel(
                        name=model.id,
                        description=model.id,
                        attribution=Attribution(
                            name="OpenAI Compatible",
                            url=base_url
                        ),
                        installed=True,
                        languages=['en'],
                        version=model.id
                    )
                )
                logger.debug("Found ASR model: %s", model.id)
                
            elif is_matching_model(model.id, tts_patterns):
                # For each TTS model, create voices for all available voice options
                for voice in tts_voices:
                    tts_model_voices.append(
                        TtsVoice(
                            name=voice,
                            description=f"{voice} ({model.id})",
                            attribution=Attribution(
                                name="OpenAI Compatible",
                                url=base_url
                            ),
                            installed=True,
                            languages=['en'],
                            version=model.id
                        )
                    )
                logger.debug("Found TTS model: %s with voices", model.id)

        return asr_models, tts_model_voices

    except Exception as e:
        logger.error("Failed to fetch OpenAI models: %s", e)
        return [], []

class NamedBytesIO(BytesIO):
    def __init__(self, *args, **kwargs):
        self._name = kwargs.pop('name', 'audio.wav')
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        return self._name