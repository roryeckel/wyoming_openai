import asyncio
import logging
import wave
from typing import List, Optional

from openai import AsyncOpenAI
from wyoming.info import Info, Describe, AsrProgram, Attribution, AsrModel, TtsProgram, TtsVoice
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.asr import Transcribe, Transcript
from wyoming.tts import Synthesize
from wyoming.server import AsyncEventHandler

from .utilities import NamedBytesIO
from . import __version__

_LOGGER = logging.getLogger(__name__)

AUDIO_WIDTH = 2  # 16-bit audio
AUDIO_CHANNELS = 1  # Mono audio
TTS_AUDIO_RATE = 24000
ASR_AUDIO_RATE = 16000
ASR_CHUNK_SIZE = 2048

class OpenAIEventHandler(AsyncEventHandler):
    def __init__(
        self,
        *args,
        stt_api_key: str,
        stt_base_url: str,
        tts_api_key: str,
        tts_base_url: str,
        client_lock: asyncio.Lock,
        asr_models: List[AsrModel],
        tts_voices: List[TtsVoice],
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self._stt_client = AsyncOpenAI(api_key=stt_api_key, base_url=stt_base_url)
        self._tts_client = AsyncOpenAI(api_key=tts_api_key, base_url=tts_base_url)

        self._client_lock = client_lock
        
        # Default to first available model/voice if available
        self._stt_model = asr_models[0].name if asr_models else "whisper-1"
        self._tts_model = "tts-1"  # Base TTS model
        
        self._wyoming_info = Info(
            asr=[
                AsrProgram(
                    name="openai",
                    description="OpenAI-Compatible Proxy",
                    attribution=Attribution(
                        name="Rory Eckel",
                        url="https://github.com/roryeckel/wyoming-openai/",
                    ),
                    installed=True,
                    version=__version__,
                    models=asr_models
                )
            ],
            tts=[
                TtsProgram(
                    name="openai",
                    description="OpenAI-Compatible Proxy",
                    attribution=Attribution(
                        name="Rory Eckel",
                        url="https://github.com/roryeckel/wyoming-openai/",
                    ),
                    installed=True,
                    version=__version__,
                    voices=tts_voices
                )
            ]
        )

        # State for current transcription
        self._wav_buffer: Optional[NamedBytesIO] = None
        self._wav_write_buffer: Optional[wave.Wave_write] = None
        self._is_recording = False

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            # Non-logging because spammy
            await self._handle_audio_chunk(AudioChunk.from_event(event))
            return True
        
        _LOGGER.debug("Incoming event type %s", event.type)

        if Transcribe.is_type(event.type):
            return await self._handle_transcribe(Transcribe.from_event(event))

        if AudioStart.is_type(event.type):
            await self._handle_audio_start()
            return True
            
        if AudioStop.is_type(event.type):
            await self._handle_audio_stop()
            return True
        
        if Synthesize.is_type(event.type):
            return await self._handle_synthesize(Synthesize.from_event(event))
        
        if Describe.is_type(event.type):
            await self.write_event(self._wyoming_info.event())
            return True

        _LOGGER.warning("Unhandled event type: %s", event.type)
        return False
    
    async def _handle_transcribe(self, transcribe: Transcribe) -> None:
        """Handle transcription request"""
        return True
        # if transcribe.language:
        #     if transcribe.language not in self._languages:
        #         _LOGGER.error(f"Language {transcribe.language} is not supported. The following languages are set in the configuration: {self._languages}")
        #         return False
        #     else:
        #         self._language = transcribe.language
        #         _LOGGER.debug("Language set to %s", transcribe.language)
        #         return True

    async def _handle_audio_start(self) -> None:
        """Handle start of audio stream"""
        self._is_recording = True
        self._wav_buffer = NamedBytesIO(name='recording.wav')
        self._wav_write_buffer = wave.open(self._wav_buffer, "wb")
        self._wav_write_buffer.setnchannels(AUDIO_CHANNELS)
        self._wav_write_buffer.setsampwidth(AUDIO_WIDTH)
        self._wav_write_buffer.setframerate(ASR_AUDIO_RATE)
        _LOGGER.info("Recording started")

    async def _handle_audio_chunk(self, chunk: AudioChunk) -> None:
        """Handle audio chunk"""
        if self._is_recording and chunk.audio and self._wav_write_buffer:
            self._wav_write_buffer.writeframes(chunk.audio)
        else:
            _LOGGER.warning(f"Problem handling audio chunk")

    async def _handle_audio_stop(self) -> None:
        """Handle end of audio stream and perform transcription"""
        if not self._is_recording or not self._wav_buffer:
            _LOGGER.warning("Received audio stop event without recording")
            return

        self._is_recording = False
        
        try:
            # Close the WAV file
            if self._wav_write_buffer:
                self._wav_write_buffer.close()
                self._wav_write_buffer = None

            # Reset buffer position to start
            self._wav_buffer.seek(0)

            # Send to OpenAI for transcription
            async with self._client_lock:
                result = await self._stt_client.audio.transcriptions.create(
                    file=self._wav_buffer,
                    model=self._stt_model
                )
                
            if result.text:
                _LOGGER.info(f"Successfully transcribed: {result.text}")

                # Send transcript event
                transcript = Transcript(
                    text=result.text
                )
                await self.write_event(transcript.event())
            else:
                _LOGGER.warning("Received empty transcription result")

        except Exception as e:
            _LOGGER.exception("Error during transcription: %s", e)
        finally:
            self._wav_buffer = None

    async def _handle_synthesize(self, synthesize: Synthesize) -> bool:
        """Handle text-to-speech synthesis request"""
        try:
            _LOGGER.debug("Handling synthesize request %s", synthesize)

            requested_voice = synthesize.voice.name

            # Validate voice against self._wyoming_info
            if requested_voice:
                tts_voices = {voice.name for program in self._wyoming_info.tts for voice in program.voices}
                if requested_voice not in tts_voices:
                    _LOGGER.error(f"Voice {requested_voice} is not supported. Available voices: {tts_voices}")
                    return False

            # Validate language against self._languages
            # requested_language = synthesize.voice.language
            # if requested_language and requested_language not in self._languages:
            #     _LOGGER.error(f"Language {requested_language} is not supported. The following languages are set in the configuration: {self._languages}")
            #     return False

            async with self._client_lock:
                async with self._tts_client.audio.speech.with_streaming_response.create(
                    model=self._tts_model,
                    voice=requested_voice,
                    input=synthesize.text) as response:
                
                    # Send audio start with required audio parameters
                    await self.write_event(
                        AudioStart(
                            rate=TTS_AUDIO_RATE,
                            width=AUDIO_WIDTH,
                            channels=AUDIO_CHANNELS
                        ).event()
                    )
                    
                    # Stream the audio in chunks
                    timestamp = 0
                    samples_per_chunk = ASR_CHUNK_SIZE // 2  # 2 bytes per sample
                    timestamp_increment = (samples_per_chunk / TTS_AUDIO_RATE) * 1000  # ms
                    
                    async for chunk in response.iter_bytes(chunk_size=ASR_CHUNK_SIZE):
                        await self.write_event(
                            AudioChunk(
                                audio=chunk,
                                rate=TTS_AUDIO_RATE,
                                width=AUDIO_WIDTH,
                                channels=AUDIO_CHANNELS,
                                timestamp=int(timestamp)
                            ).event()
                        )
                        timestamp += timestamp_increment
                    
                    # Send audio stop
                    await self.write_event(AudioStop().event())
                    
                    _LOGGER.debug("Successfully synthesized: %s", synthesize.text[:100])
                    return True

        except Exception as e:
            _LOGGER.exception("Error during synthesis: %s", e)
            return False
        
    async def stop(self) -> None:
        await super().stop()
        self._stt_client.close()
        self._tts_client.close()