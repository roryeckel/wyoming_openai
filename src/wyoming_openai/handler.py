import asyncio
import io
import logging
import wave

import pysbd
from openai import NOT_GIVEN, AsyncStream
from openai.resources.audio.transcriptions import TranscriptionCreateResponse
from wyoming.asr import (
    Transcribe,
    Transcript,
    TranscriptChunk,
    TranscriptStart,
    TranscriptStop,
)
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, Describe, Info, TtsVoice
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
    SynthesizeVoice,
)

from .compatibility import CustomAsyncOpenAI, OpenAIBackend, TtsVoiceModel
from .utilities import NamedBytesIO

_LOGGER = logging.getLogger(__name__)

DEFAULT_AUDIO_WIDTH = 2  # 16-bit audio
DEFAULT_AUDIO_CHANNELS = 1  # Mono audio
DEFAULT_ASR_AUDIO_RATE = 16000  # Hz (Wyoming default)
TTS_AUDIO_RATE = 24000  # Hz (OpenAI spec, fallback)
TTS_CHUNK_SIZE = 2048  # Magical guess - but must be larger than 44 bytes for a potential WAV header

class OpenAIEventHandler(AsyncEventHandler):
    def __init__(
        self,
        *args,
        info: Info,
        stt_client: CustomAsyncOpenAI,
        tts_client: CustomAsyncOpenAI,
        client_lock: asyncio.Lock,
        stt_temperature: float | None = None,
        stt_prompt: str | None = None,
        tts_speed: float | None = None,
        tts_instructions: str | None = None,
        tts_streaming_min_words: int | None = None,
        tts_streaming_max_chars: int | None = None,
        **kwargs,
    ) -> None:
        """
        Initializes the OpenAIEventHandler.

        Args:
            *args: Variable length argument list for the superclass.
            info (Info): The Wyoming info object.
            stt_client (CustomAsyncOpenAI): The client for speech-to-text.
            tts_client (CustomAsyncOpenAI): The client for text-to-speech.
            client_lock (asyncio.Lock): A lock to ensure thread-safe client access.
            stt_temperature (float | None): The temperature for STT, or None for default.
            stt_prompt (str | None): An optional prompt for STT.
            tts_speed (float | None): The speed for TTS, or None for default.
            tts_instructions (str | None): Optional instructions for TTS.
            tts_streaming_min_words (int | None): Minimum words per chunk for streaming TTS.
            tts_streaming_max_chars (int | None): Maximum characters per chunk for streaming TTS.
            **kwargs: Arbitrary keyword arguments for the superclass.
        """
        super().__init__(*args, **kwargs)
        self._wyoming_info = info

        self._client_lock = client_lock

        self._stt_client = stt_client
        self._stt_temperature = stt_temperature
        self._stt_prompt = stt_prompt

        self._tts_client = tts_client
        self._tts_speed = tts_speed
        self._tts_instructions = tts_instructions
        self._tts_streaming_min_words = tts_streaming_min_words
        self._tts_streaming_max_chars = tts_streaming_max_chars

        # State for current transcription
        self._wav_buffer: NamedBytesIO | None = None
        self._wav_write_buffer: wave.Wave_write | None = None
        self._is_recording: bool = False
        self._current_asr_model: AsrModel | None = None

        # State for event logging
        self._last_event_type: str | None = None
        self._event_counter: int = 0

        # State for streaming synthesis
        self._synthesis_buffer: list[str] = []
        self._synthesis_voice: SynthesizeVoice | None = None
        self._is_synthesizing: bool = False

    async def handle_event(self, event: Event) -> bool:
        """
        Handle incoming events
        https://github.com/OHF-Voice/wyoming?tab=readme-ov-file#event-types
        """
        if AudioChunk.is_type(event.type):
            # Non-logging because spammy
            await self._handle_audio_chunk(AudioChunk.from_event(event))
            return True

        _LOGGER.debug("Incoming event type %s", event.type)

        if Transcribe.is_type(event.type):
            return await self._handle_transcribe(Transcribe.from_event(event))

        if AudioStart.is_type(event.type):
            sample_rate = DEFAULT_ASR_AUDIO_RATE
            audio_width = DEFAULT_AUDIO_WIDTH
            audio_channels = DEFAULT_AUDIO_CHANNELS
            if event.data:
                if 'rate' in event.data:
                    sample_rate = event.data['rate']
                if 'width' in event.data:
                    audio_width = event.data['width']
                if 'channels' in event.data:
                    audio_channels = event.data['channels']
            await self._handle_audio_start(sample_rate, audio_width, audio_channels)
            return True

        if AudioStop.is_type(event.type):
            await self._handle_audio_stop()
            return True

        if Synthesize.is_type(event.type):
            return await self._handle_synthesize(Synthesize.from_event(event))

        if SynthesizeStart.is_type(event.type):
            return await self._handle_synthesize_start(SynthesizeStart.from_event(event))

        if SynthesizeChunk.is_type(event.type):
            return await self._handle_synthesize_chunk(SynthesizeChunk.from_event(event))

        if SynthesizeStop.is_type(event.type):
            return await self._handle_synthesize_stop()

        if Describe.is_type(event.type):
            await self.write_event(self._wyoming_info.event())
            return True

        _LOGGER.info("Ignoring unhandled event type: %s", event.type)
        return True

    async def _handle_transcribe(self, transcribe: Transcribe) -> bool:
        """Handle transcription request"""
        self._current_asr_model = self._get_asr_model(transcribe.name)
        if self._current_asr_model:
            if self._is_asr_language_supported(transcribe.language, self._current_asr_model):
                return True
            self._log_unsupported_asr_language(transcribe.name, transcribe.language)
        else:
            self._log_unsupported_asr_model(transcribe.name)
        return False

    async def _handle_audio_start(self, sample_rate: int, audio_width: int, audio_channels: int) -> None:
        """Handle start of audio stream"""
        self._is_recording = True
        self._wav_buffer = NamedBytesIO(name='recording.wav')
        self._wav_write_buffer = wave.open(self._wav_buffer, "wb")
        self._wav_write_buffer.setnchannels(audio_channels)
        self._wav_write_buffer.setsampwidth(audio_width)
        self._wav_write_buffer.setframerate(sample_rate)
        _LOGGER.info("Recording started at %d Hz, %d channels, %d bytes per sample",
                     sample_rate, audio_channels, audio_width)

    async def _handle_audio_chunk(self, chunk: AudioChunk) -> None:
        """Handle audio chunk"""
        if self._is_recording and chunk.audio and self._wav_write_buffer:
            self._wav_write_buffer.writeframes(chunk.audio)
        else:
            _LOGGER.warning("Problem handling audio chunk")

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
                use_streaming = self._is_asr_model_streaming(self._current_asr_model.name)

                # Prepare extra_body for SPEACHES backend
                extra_body = {}
                if hasattr(self._stt_client, 'backend') and self._stt_client.backend == OpenAIBackend.SPEACHES:
                    extra_body["vad_filter"] = False
                    _LOGGER.debug("Adding vad_filter=False for SPEACHES backend")

                transcription = await self._stt_client.audio.transcriptions.create(
                    file=self._wav_buffer,
                    model=self._current_asr_model.name,
                    temperature=self._stt_temperature or NOT_GIVEN,
                    prompt=self._stt_prompt or NOT_GIVEN,
                    response_format="json",
                    stream=use_streaming,
                    extra_body=extra_body if extra_body else None
                )

                await self.write_event(TranscriptStart().event())

                if isinstance(transcription, AsyncStream):
                    _LOGGER.debug("Handling streaming transcription response")
                    full_text = ""
                    async for chunk in transcription:
                        if chunk.type == "transcript.text.delta":
                            if chunk.delta:
                                full_text += chunk.delta
                                _LOGGER.debug("Transcribed chunk: %s", chunk.delta)
                                await self.write_event(
                                    TranscriptChunk(text=chunk.delta).event()
                                )
                    if full_text:
                        _LOGGER.info("Successfully transcribed stream: %s", full_text)
                    else:
                        _LOGGER.warning("Received empty transcription from stream. If this is unexpected, please check your STT_STREAMING_MODELS configuration.")
                    await self.write_event(Transcript(text=full_text).event())

                elif isinstance(transcription, TranscriptionCreateResponse):
                    # Handle non-streaming response
                    _LOGGER.debug("Handling non-streaming transcription response")
                    if transcription.text:
                        _LOGGER.info("Successfully transcribed: %s", transcription.text)
                    else:
                        _LOGGER.warning("Received empty transcription result")
                    await self.write_event(Transcript(text=transcription.text).event())

                else:
                    _LOGGER.error("Unexpected transcription response type: %s", type(transcription))

                await self.write_event(TranscriptStop().event())

        except Exception as e:
            _LOGGER.exception("Error during transcription: %s", e)
        finally:
            if self._wav_buffer:
                self._wav_buffer.close()
                self._wav_buffer = None

    def _get_asr_model(self, model_name: str | None = None) -> AsrModel | None:
        """Get an ASR model by name or None"""
        for program in self._wyoming_info.asr:
            for model in program.models:
                if model.name == model_name or not model_name:
                    return model
        return None

    def _is_asr_model_streaming(self, model_name: str) -> bool:
        """Check if an ASR model supports streaming"""
        for program in self._wyoming_info.asr:
            for model in program.models:
                if model.name == model_name:
                    return program.supports_transcript_streaming
        return False

    def _is_tts_voice_streaming(self, voice_name: str) -> bool:
        """Check if a TTS voice supports streaming synthesis"""
        for program in self._wyoming_info.tts:
            for voice in program.voices:
                if voice.name == voice_name:
                    return getattr(program, 'supports_synthesize_streaming', False)
        return False

    def _get_pysbd_language(self, language: str | None) -> str:
        """
        Get pysbd-compatible language code.

        Args:
            language (str | None): Language code (e.g., 'en', 'en-US', 'es', etc.)

        Returns:
            str: pysbd-compatible language code, defaults to 'en' if unsupported
        """
        if not language:
            return 'en'

        # Extract base language code from potential BCP-47 tags (e.g., 'en-US' -> 'en')
        base_lang = language[:2].lower() if len(language) >= 2 else 'en'

        # Test if the language is supported by trying to create a segmenter
        try:
            pysbd.Segmenter(language=base_lang)
            return base_lang
        except (ValueError, KeyError):
            _LOGGER.warning(f"Language '{base_lang}' not supported by pysbd, using English")
            return 'en'

    def _chunk_text_for_streaming(self, text: str, min_words: int | None = None, max_chars: int | None = None, language: str | None = None) -> list[str]:
        """
        Chunk text into meaningful segments using pySBD sentence segmentation.

        Args:
            text (str): The text to chunk.
            min_words (int | None): Minimum words per chunk. If None, no minimum enforced.
            max_chars (int | None): Maximum characters per chunk. If None, no maximum enforced.
            language (str | None): Language code for sentence segmentation. If None, defaults to 'en'.

        Returns:
            list[str]: List of text chunks ready for TTS streaming.
        """
        if not text.strip():
            return []

        # Get pysbd-compatible language code
        pysbd_language = self._get_pysbd_language(language)
        segmenter = pysbd.Segmenter(language=pysbd_language, clean=False)
        sentences = segmenter.segment(text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Check if adding this sentence would exceed max_chars
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if max_chars and len(potential_chunk) > max_chars and current_chunk:
                # Current chunk is ready, start new chunk with this sentence
                if not min_words or self._meets_min_criteria(current_chunk, min_words):
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            elif not max_chars and not min_words:
                # No limits set - each sentence becomes its own chunk for natural streaming
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = potential_chunk

        # Add remaining chunk if it meets criteria
        if current_chunk and (not min_words or self._meets_min_criteria(current_chunk, min_words)):
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]  # Fallback to original text if no valid chunks

    def _meets_min_criteria(self, text: str, min_words: int) -> bool:
        """Check if text chunk meets minimum word requirement."""
        word_count = len(text.split())
        return word_count >= min_words

    def _log_unsupported_asr_model(self, model_name: str | None = None):
        """Log an unsupported ASR model"""
        if model_name:
            _LOGGER.warning("Unsupported ASR model: %s", model_name)
        else:
            _LOGGER.warning("No ASR models specified")

    def _is_asr_language_supported(self, language: str, model: AsrModel) -> bool:
        """Check if a language is supported by an ASR model"""
        return not model.languages or language in model.languages

    def _log_unsupported_asr_language(self, model_name: str, language: str):
        """Log an unsupported ASR language"""
        _LOGGER.error("Unsupported ASR model %s for language %s", model_name, language)

    def _get_voice(self, name: str | None = None) -> TtsVoiceModel | None:
        """Get a TTS voice by name or None"""
        for program in self._wyoming_info.tts:
            for voice in program.voices:
                if not name or voice.name == name:
                    return voice
        return None

    def _is_tts_language_supported(self, language: str, voice: TtsVoice) -> bool:
        """Check if a language is supported by a TTS voice"""
        return not voice.languages or language in voice.languages

    def _validate_tts_voice_and_language(self, requested_voice: str | None, requested_language: str | None) -> TtsVoiceModel | None:
        """
        Validate and get a TTS voice by name and language.

        Args:
            requested_voice (str | None): The requested voice name.
            requested_language (str | None): The requested language.

        Returns:
            TtsVoiceModel | None: The validated voice, or None if validation failed.
        """
        # Get voice
        voice = self._get_voice(requested_voice)
        if not voice:
            self._log_unsupported_voice(requested_voice)
            return None

        # Validate language
        if not self._validate_tts_language(requested_language, voice):
            return None

        return voice

    def _validate_tts_language(self, language: str | None, voice: TtsVoice) -> bool:
        """Validate if a language is supported by a TTS voice. Returns True if supported. If no language is specified, also returns True."""
        if language and not self._is_tts_language_supported(language, voice):
            _LOGGER.error(f"Language {language} is not supported for voice {voice.name}. Available languages: {voice.languages}")
            return False
        return True

    def _log_unsupported_voice(self, requested_voice: str | None) -> None:
        """Log an error message if a voice is not supported"""
        if requested_voice:
            _LOGGER.error(f"Voice {requested_voice} is not supported. Available voices: {[voice.name for program in self._wyoming_info.tts for voice in program.voices]}")
        else:
            _LOGGER.error("No TTS voices specified")

    async def _handle_synthesize(self, synthesize: Synthesize) -> bool:
        """Handle text-to-speech synthesis request"""
        try:
            _LOGGER.debug("Handling synthesize request %s", synthesize)

            # IMPORTANT: Ignore standalone synthesize events when streaming synthesis is already active
            # This prevents duplicate audio synthesis when both streaming events (synthesize-start/chunk/stop)
            # and standalone synthesize events are used together
            if self._is_synthesizing:
                _LOGGER.debug("Ignoring standalone synthesize event - streaming synthesis is already active")
                return True

            if synthesize.voice:
                requested_voice = synthesize.voice.name
                requested_language = synthesize.voice.language
            else:
                requested_voice = None
                requested_language = None

            # Validate voice and language
            voice = self._validate_tts_voice_and_language(requested_voice, requested_language)
            if not voice:
                return False

            # Use shared streaming logic
            final_timestamp = await self._stream_tts_audio(voice, synthesize.text, send_audio_start=True)

            if final_timestamp is not None:
                # Send audio stop after streaming completes
                await self.write_event(AudioStop(timestamp=final_timestamp).event())
                _LOGGER.info("Successfully synthesized: %s", synthesize.text[:100])
                return True
            return False

        except Exception as e:
            _LOGGER.exception("Error during synthesis: %s", e)
            return False

    async def _handle_synthesize_start(self, synthesize_start: SynthesizeStart) -> bool:
        """Handle start of streaming synthesis"""
        _LOGGER.debug("Handling synthesize-start event: %s", synthesize_start)

        # Reset synthesis state
        self._synthesis_buffer = []
        self._is_synthesizing = True

        # Store voice information if provided
        if synthesize_start.voice:
            self._synthesis_voice = synthesize_start.voice
            requested_voice = synthesize_start.voice.name
            requested_language = synthesize_start.voice.language

            # Validate voice and language
            voice = self._validate_tts_voice_and_language(requested_voice, requested_language)
            if not voice:
                self._is_synthesizing = False
                return False
        else:
            self._synthesis_voice = None

        return True

    async def _handle_synthesize_chunk(self, synthesize_chunk: SynthesizeChunk) -> bool:
        """Handle text chunk during streaming synthesis"""
        if not self._is_synthesizing:
            _LOGGER.warning("Received synthesize-chunk without active synthesis")
            return False

        _LOGGER.debug("Received synthesis chunk: %s", synthesize_chunk.text[:50] if synthesize_chunk.text else "")
        self._synthesis_buffer.append(synthesize_chunk.text)
        return True

    async def _handle_synthesize_stop(self) -> bool:
        """Handle end of streaming synthesis"""
        if not self._is_synthesizing:
            _LOGGER.warning("Received synthesize-stop without active synthesis")
            return False

        self._is_synthesizing = False

        # Get accumulated text and voice
        full_text = "".join(self._synthesis_buffer)
        voice_info = self._synthesis_voice

        _LOGGER.debug("Streaming synthesis completed with text: %s", full_text[:100])

        # Clear synthesis state early
        self._synthesis_buffer = []
        self._synthesis_voice = None

        if not full_text.strip():
            _LOGGER.warning("No text to synthesize")
            await self.write_event(SynthesizeStopped().event())
            return True

        try:
            # Determine voice for synthesis
            if voice_info:
                requested_voice = voice_info.name
                requested_language = voice_info.language
            else:
                requested_voice = None
                requested_language = None

            # Validate voice and language
            voice = self._validate_tts_voice_and_language(requested_voice, requested_language)
            if not voice:
                await self.write_event(SynthesizeStopped().event())
                return False

            # Check if streaming is enabled for this voice
            use_streaming = self._is_tts_voice_streaming(voice.name)

            if use_streaming:
                # Chunk text for streaming synthesis
                chunks = self._chunk_text_for_streaming(
                    full_text,
                    self._tts_streaming_min_words,
                    self._tts_streaming_max_chars,
                    requested_language
                )
                _LOGGER.debug("Text chunked into %d parts for streaming synthesis", len(chunks))

                # Start all OpenAI TTS calls concurrently
                _LOGGER.debug("Starting parallel synthesis for %d chunks", len(chunks))
                synthesis_tasks = [
                    asyncio.create_task(self._get_tts_audio_stream(chunk, voice), name=f"chunk_{i}")
                    for i, chunk in enumerate(chunks)
                ]

                # Stream results sequentially as they complete, maintaining order
                total_timestamp = 0
                for i, task in enumerate(synthesis_tasks):
                    try:
                        _LOGGER.debug("Streaming chunk %d/%d to Wyoming", i + 1, len(chunks))

                        # Wait for this specific chunk to complete
                        audio_stream = await task
                        if audio_stream is None:
                            _LOGGER.error("Failed to synthesize chunk %d", i + 1)
                            await self.write_event(SynthesizeStopped().event())
                            return False

                        # Stream the audio to Wyoming with proper timestamp calculation
                        chunk_timestamp = await self._stream_audio_to_wyoming(
                            audio_stream,
                            is_first_chunk=(i == 0),
                            start_timestamp=total_timestamp
                        )

                        if chunk_timestamp is not None:
                            total_timestamp = chunk_timestamp
                        else:
                            _LOGGER.error("Failed to stream chunk %d to Wyoming", i + 1)
                            await self.write_event(SynthesizeStopped().event())
                            return False

                    except Exception as e:
                        _LOGGER.exception("Error processing chunk %d: %s", i + 1, e)
                        await self.write_event(SynthesizeStopped().event())
                        return False

                # Send final audio stop
                await self.write_event(AudioStop(timestamp=total_timestamp).event())
                _LOGGER.info("Successfully completed parallel streaming synthesis: %s", full_text[:100])
            else:
                # Use non-streaming synthesis for non-streaming voices
                _LOGGER.debug("Using non-streaming synthesis for voice: %s", voice.name)
                success = await self._synthesize_non_streaming(full_text, voice)
                if not success:
                    await self.write_event(SynthesizeStopped().event())
                    return False

            await self.write_event(SynthesizeStopped().event())
            return True

        except Exception as e:
            _LOGGER.exception("Error during streaming synthesis: %s", e)
            await self.write_event(SynthesizeStopped().event())
            return False

    async def _get_tts_audio_stream(self, text: str, voice: TtsVoiceModel) -> bytes | None:
        """
        Get TTS audio stream from OpenAI for a text chunk (parallel-safe).

        Args:
            text (str): Text chunk to synthesize.
            voice (TtsVoiceModel): Voice to use for synthesis.

        Returns:
            bytes | None: Complete audio data for the chunk, or None on error.
        """
        try:
            audio_data = b""
            async with self._client_lock, self._tts_client.audio.speech.with_streaming_response.create(
                model=voice.model_name,
                voice=voice.name,
                input=text,
                speed=self._tts_speed or NOT_GIVEN,
                instructions=self._tts_instructions or NOT_GIVEN
            ) as response:
                async for chunk in response.iter_bytes(chunk_size=TTS_CHUNK_SIZE):
                    audio_data += chunk

            _LOGGER.debug("Completed synthesis for chunk: %s", text[:50])
            return audio_data

        except Exception as e:
            _LOGGER.exception("Error getting TTS audio stream: %s", e)
            return None

    async def _stream_audio_to_wyoming(self, audio_data: bytes, is_first_chunk: bool, start_timestamp: float) -> float | None:
        """
        Stream audio data to Wyoming with proper timestamp calculation.

        Args:
            audio_data (bytes): Complete audio data to stream.
            is_first_chunk (bool): Whether this is the first chunk (sends AudioStart).
            start_timestamp (float): Starting timestamp for this chunk.

        Returns:
            float | None: Final timestamp after streaming, or None on error.
        """
        try:
            audio_rate = TTS_AUDIO_RATE
            audio_width = DEFAULT_AUDIO_WIDTH
            audio_channels = DEFAULT_AUDIO_CHANNELS
            timestamp = start_timestamp

            # Parse WAV header from first bytes to get audio parameters
            wav_params = self._parse_wav_header(audio_data[:1024])  # Check first 1KB
            data_offset = 0
            if wav_params:
                audio_rate, audio_channels, audio_width, data_offset = wav_params
                _LOGGER.debug("Detected audio format: %d Hz, %d channels, %d bytes/sample, header offset: %d",
                            audio_rate, audio_channels, audio_width, data_offset)
            else:
                _LOGGER.debug("Could not parse WAV header, using defaults: %d Hz", TTS_AUDIO_RATE)

            # Send audio start if this is the first chunk
            if is_first_chunk:
                await self.write_event(
                    AudioStart(
                        rate=audio_rate,
                        width=audio_width,
                        channels=audio_channels
                    ).event()
                )

            # Strip WAV header from audio data
            actual_audio_data = audio_data[data_offset:] if data_offset > 0 else audio_data

            # Stream audio data in chunks
            chunk_size = TTS_CHUNK_SIZE
            for i in range(0, len(actual_audio_data), chunk_size):
                chunk = actual_audio_data[i:i + chunk_size]
                if chunk:
                    await self.write_event(
                        AudioChunk(
                            audio=chunk,
                            rate=audio_rate,
                            width=audio_width,
                            channels=audio_channels,
                            timestamp=int(timestamp)
                        ).event()
                    )
                    # Calculate timestamp increment based on actual audio data length
                    actual_samples = len(chunk) // audio_width
                    timestamp += (actual_samples / audio_rate) * 1000

            return timestamp

        except Exception as e:
            _LOGGER.exception("Error streaming audio to Wyoming: %s", e)
            return None

    async def _synthesize_chunk(self, text: str, voice: TtsVoiceModel, is_first_chunk: bool, start_timestamp: float) -> float | None:
        """
        Synthesize a single text chunk and stream the audio (legacy method for compatibility).

        Args:
            text (str): Text chunk to synthesize.
            voice (TtsVoiceModel): Voice to use for synthesis.
            is_first_chunk (bool): Whether this is the first chunk (sends AudioStart).
            start_timestamp (float): Starting timestamp for this chunk.

        Returns:
            float | None: Final timestamp after this chunk, or None on error.
        """
        return await self._stream_tts_audio(voice, text, send_audio_start=is_first_chunk, start_timestamp=start_timestamp)

    async def _synthesize_non_streaming(self, text: str, voice: TtsVoiceModel) -> bool:
        """
        Synthesize text using the existing non-streaming approach.

        Args:
            text (str): Text to synthesize.
            voice (TtsVoiceModel): Voice to use for synthesis.

        Returns:
            bool: True on success, False on error.
        """
        final_timestamp = await self._stream_tts_audio(voice, text, send_audio_start=True)

        if final_timestamp is not None:
            # Send audio stop after streaming completes
            await self.write_event(AudioStop(timestamp=final_timestamp).event())
            _LOGGER.info("Successfully synthesized non-streaming: %s", text[:100])
            return True
        return False

    async def _stream_tts_audio(self, voice: TtsVoiceModel, text: str, send_audio_start: bool = True, start_timestamp: float = 0) -> float | None:
        """
        Stream TTS audio for the given text and voice.

        Args:
            voice (TtsVoiceModel): Voice to use for synthesis.
            text (str): Text to synthesize.
            send_audio_start (bool): Whether to send AudioStart event.
            start_timestamp (float): Starting timestamp for audio chunks.

        Returns:
            float | None: Final timestamp after streaming, or None on error.
        """
        try:
            first_chunk = None
            audio_rate = TTS_AUDIO_RATE
            audio_width = DEFAULT_AUDIO_WIDTH
            audio_channels = DEFAULT_AUDIO_CHANNELS
            timestamp = start_timestamp

            async with self._client_lock, self._tts_client.audio.speech.with_streaming_response.create(
                model=voice.model_name,
                voice=voice.name,
                input=text,
                speed=self._tts_speed or NOT_GIVEN,
                instructions=self._tts_instructions or NOT_GIVEN
            ) as response:

                async for chunk in response.iter_bytes(chunk_size=TTS_CHUNK_SIZE):
                    if first_chunk is None:
                        first_chunk = chunk
                        audio_data = chunk

                        # Try to parse WAV header from first chunk
                        wav_params = self._parse_wav_header(chunk)
                        if wav_params:
                            audio_rate, audio_channels, audio_width, data_offset = wav_params
                            audio_data = chunk[data_offset:]
                            _LOGGER.debug("Detected audio format: %d Hz, %d channels, %d bytes/sample, header offset: %d",
                                        audio_rate, audio_channels, audio_width, data_offset)
                        else:
                            _LOGGER.debug("Could not parse WAV header, using defaults: %d Hz", TTS_AUDIO_RATE)

                        # Send audio start if requested
                        if send_audio_start:
                            await self.write_event(
                                AudioStart(
                                    rate=audio_rate,
                                    width=audio_width,
                                    channels=audio_channels
                                ).event()
                            )
                    else:
                        audio_data = chunk

                    # Send audio chunk (header stripped for first chunk)
                    if audio_data:
                        await self.write_event(
                            AudioChunk(
                                audio=audio_data,
                                rate=audio_rate,
                                width=audio_width,
                                channels=audio_channels,
                                timestamp=int(timestamp)
                            ).event()
                        )
                        # Calculate timestamp increment based on actual audio data length
                        actual_samples = len(audio_data) // audio_width
                        timestamp += (actual_samples / audio_rate) * 1000

            return timestamp

        except Exception as e:
            _LOGGER.exception("Error streaming TTS audio: %s", e)
            return None

    def _parse_wav_header(self, wav_data: bytes) -> tuple[int, int, int, int] | None:
        """
        Parse WAV header to extract sample rate, channels, sample width, and data offset.
        Returns (sample_rate, channels, sample_width, data_offset) or None if parsing fails.
        """
        try:
            # Create a BytesIO object from the data
            wav_io = io.BytesIO(wav_data)

            # Open with wave module
            with wave.open(wav_io, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()

                # Get the current position which should be at the start of audio data
                data_offset = wav_io.tell()

                return sample_rate, channels, sample_width, data_offset
        except Exception as e:
            _LOGGER.debug("Failed to parse WAV header: %s", e)
            return None

    async def write_event(self, event: Event) -> None:
        """Override write_event to add debug logging with AudioChunk filtering"""
        # Check if this is a new event type
        if self._last_event_type != event.type:
            self._last_event_type = event.type
            self._event_counter = 1
        else:
            self._event_counter += 1

        # Handle AudioChunk logging specially
        if event.type == "audio-chunk":
            if self._event_counter == 1:
                _LOGGER.debug("Outgoing event type %s", event.type)
            elif self._event_counter == 2:
                _LOGGER.debug("Outgoing event type %s (subsequent audio chunks will not be logged)", event.type)
            # Subsequent AudioChunk events are silenced
        else:
            _LOGGER.debug("Outgoing event type %s", event.type)

        await super().write_event(event)

    async def stop(self) -> None:
        """Stop the handler and close the clients"""
        await super().stop()
        self._stt_client.close()
        self._tts_client.close()
