import asyncio
import base64
import contextlib
import io
import logging
import struct
import wave
from dataclasses import dataclass
from typing import Any, cast

import pysbd
from openai import AsyncStream, omit
from openai.types.audio.transcription_create_response import TranscriptionCreateResponse
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
from .utilities import NamedBytesIO, get_extra_body_boolean_field, validate_stt_extra_body, validate_tts_extra_body

_LOGGER = logging.getLogger(__name__)


def _truncate_for_log(text: str, max_length: int = 100) -> str:
    """Truncate text for logging, adding ellipsis only if truncated."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


DEFAULT_AUDIO_WIDTH = 2  # 16-bit audio
DEFAULT_AUDIO_CHANNELS = 1  # Mono audio
DEFAULT_ASR_AUDIO_RATE = 16000  # Hz (Wyoming default)
REALTIME_AUDIO_RATE = 24000  # Hz (OpenAI Realtime audio/pcm requirement)
REALTIME_AUDIO_WIDTH = 2  # 16-bit audio
REALTIME_AUDIO_CHANNELS = 1  # Mono audio
TTS_AUDIO_RATE = 24000  # Hz (OpenAI spec, fallback)
TTS_CHUNK_SIZE = 2048  # Magical guess - but must be larger than 44 bytes for a potential WAV header
TTS_CONCURRENT_REQUESTS = 3  # Number of concurrent OpenAI TTS requests when streaming sentences
TTS_WAV_HEADER_MAX_BYTES = 65536  # Bound header buffering if a backend never yields a complete WAV header


@dataclass(frozen=True)
class TtsStreamResult:
    """Container for TTS streaming outcomes."""

    streamed: bool
    audio: bytes | None = None


class TtsStreamError(Exception):
    """Raised when TTS streaming fails for a specific text chunk."""

    def __init__(self, message: str, chunk_preview: str, voice: str):
        super().__init__(message)
        self.chunk_preview = chunk_preview
        self.voice = voice


class RealtimeTranscriptionError(Exception):
    """Raised when OpenAI Realtime transcription fails."""


class OpenAIEventHandler(AsyncEventHandler):
    def __init__(
        self,
        *args,
        info: Info,
        stt_client: CustomAsyncOpenAI | None,
        tts_client: CustomAsyncOpenAI | None,
        stt_temperature: float | None = None,
        stt_prompt: str | None = None,
        stt_extra_body: dict[str, object] | None = None,
        stt_realtime_models: list[str] | set[str] | None = None,
        tts_speed: float | None = None,
        tts_instructions: str | None = None,
        tts_extra_body: dict[str, object] | None = None,
        tts_streaming_min_words: int | None = None,
        tts_streaming_max_chars: int | None = None,
        **kwargs,
    ) -> None:
        """
        Initializes the OpenAIEventHandler.

        Args:
            *args: Variable length argument list for the superclass.
            info (Info): The Wyoming info object.
            stt_client (CustomAsyncOpenAI | None): The client for speech-to-text.
            tts_client (CustomAsyncOpenAI | None): The client for text-to-speech.
            stt_temperature (float | None): The temperature for STT, or None for default.
            stt_prompt (str | None): An optional prompt for STT.
            stt_extra_body (dict[str, object] | None): Optional JSON body fields merged into STT requests.
            stt_realtime_models (list[str] | set[str] | None): STT models that use OpenAI Realtime transcription.
            tts_speed (float | None): The speed for TTS, or None for default.
            tts_instructions (str | None): Optional instructions for TTS.
            tts_extra_body (dict[str, object] | None): Optional JSON body fields merged into TTS requests.
            tts_streaming_min_words (int | None): Minimum words per chunk for streaming TTS.
            tts_streaming_max_chars (int | None): Maximum characters per chunk for streaming TTS.
            Note: The caller owns the STT/TTS clients and is responsible for closing them.
            **kwargs: Arbitrary keyword arguments for the superclass.
        """
        super().__init__(*args, **kwargs)
        self._wyoming_info = info

        self._stt_client = stt_client
        self._stt_temperature = stt_temperature
        self._stt_prompt = stt_prompt
        self._stt_extra_body = dict(stt_extra_body) if stt_extra_body else None
        self._stt_realtime_models = set(stt_realtime_models or self._get_asr_program_model_names("openai-realtime"))
        if self._has_asr_models():
            validate_stt_extra_body(self._stt_extra_body)

        self._tts_client = tts_client
        self._tts_speed = tts_speed
        self._tts_instructions = tts_instructions
        self._tts_extra_body = dict(tts_extra_body) if tts_extra_body else None
        if self._has_tts_voices():
            validate_tts_extra_body(self._tts_extra_body)
        self._tts_streaming_min_words = tts_streaming_min_words
        self._tts_streaming_max_chars = tts_streaming_max_chars

        # State for current transcription
        self._wav_buffer: NamedBytesIO | None = None
        self._wav_write_buffer: wave.Wave_write | None = None
        self._is_recording: bool = False
        self._current_asr_model: AsrModel | None = None
        self._current_language: str | None = None
        self._audio_sample_rate: int = DEFAULT_ASR_AUDIO_RATE
        self._audio_width: int = DEFAULT_AUDIO_WIDTH
        self._audio_channels: int = DEFAULT_AUDIO_CHANNELS

        # State for realtime transcription
        self._realtime_connection_manager: Any | None = None
        self._realtime_connection: Any | None = None
        self._realtime_receive_task: asyncio.Task[None] | None = None
        self._realtime_transcript_future: asyncio.Future[str] | None = None

        # State for event logging
        self._last_event_type: str | None = None
        self._event_counter: int = 0

        # State for streaming synthesis
        self._synthesis_buffer: list[str] = []
        self._synthesis_voice: SynthesizeVoice | None = None
        self._is_synthesizing: bool = False

        # State for incremental sentence detection
        self._text_accumulator: str = ""
        self._ready_chunks: list[str] = []
        self._pysbd_segmenters: dict[str, pysbd.Segmenter] = {}  # Cache segmenters per language
        self._audio_started: bool = False  # Track if AudioStart has been sent
        self._current_timestamp: float = 0  # Track timestamp continuity across chunks

        self._tts_semaphore = asyncio.Semaphore(TTS_CONCURRENT_REQUESTS)
        self._allow_streaming_task_id: str | None = None  # ID of task allowed to stream directly

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
                if "rate" in event.data:
                    sample_rate = event.data["rate"]
                if "width" in event.data:
                    audio_width = event.data["width"]
                if "channels" in event.data:
                    audio_channels = event.data["channels"]
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

    async def disconnect(self) -> None:
        """Clean up handler-owned realtime resources when the Wyoming client disconnects."""
        await self._cleanup_realtime_transcription()

    async def stop(self) -> None:
        """Stop the event handler without closing shared OpenAI clients."""
        await self._cleanup_realtime_transcription()
        await super().stop()

    async def _handle_transcribe(self, transcribe: Transcribe) -> bool:
        """Handle transcription request"""
        requested_model = self._get_asr_model(transcribe.name)
        requested_language = transcribe.language

        self._current_asr_model = None
        self._current_language = None

        if requested_model:
            if self._is_asr_language_supported(requested_language, requested_model):
                self._current_asr_model = requested_model
                self._current_language = requested_language
                return True
            self._log_unsupported_asr_language(transcribe.name, requested_language)
        else:
            self._log_unsupported_asr_model(transcribe.name)
        return False

    async def _handle_audio_start(self, sample_rate: int, audio_width: int, audio_channels: int) -> None:
        """Handle start of audio stream"""
        self._audio_sample_rate = sample_rate
        self._audio_width = audio_width
        self._audio_channels = audio_channels

        if self._current_asr_model and self._is_asr_model_realtime(self._current_asr_model.name):
            await self._handle_realtime_audio_start(sample_rate, audio_width, audio_channels)
            return

        self._is_recording = True
        self._wav_buffer = NamedBytesIO(name="recording.wav")
        self._wav_write_buffer = wave.open(self._wav_buffer, "wb")
        self._wav_write_buffer.setnchannels(audio_channels)
        self._wav_write_buffer.setsampwidth(audio_width)
        self._wav_write_buffer.setframerate(sample_rate)
        _LOGGER.info(
            "Recording started at %d Hz, %d channels, %d bytes per sample", sample_rate, audio_channels, audio_width
        )

    async def _handle_audio_chunk(self, chunk: AudioChunk) -> None:
        """Handle audio chunk"""
        if self._realtime_connection is not None:
            await self._handle_realtime_audio_chunk(chunk)
            return

        if self._is_recording and chunk.audio and self._wav_write_buffer:
            self._wav_write_buffer.writeframes(chunk.audio)
        else:
            _LOGGER.warning("Problem handling audio chunk")

    async def _handle_realtime_audio_start(self, sample_rate: int, audio_width: int, audio_channels: int) -> None:
        """Open an OpenAI Realtime transcription session."""
        if not self._current_asr_model:
            _LOGGER.warning("No ASR model set for realtime transcription")
            return

        if self._stt_client is None:
            _LOGGER.error("No STT client configured for realtime transcription")
            return

        await self._cleanup_realtime_transcription()

        try:
            self._realtime_connection_manager = self._connect_realtime_transcription()
            connection = await self._enter_realtime_connection(self._realtime_connection_manager)
            self._realtime_connection = connection
            self._realtime_transcript_future = asyncio.get_running_loop().create_future()
            self._realtime_receive_task = asyncio.create_task(
                self._receive_realtime_transcription_events(), name="openai_realtime_transcription"
            )

            await connection.session.update(session=self._get_realtime_transcription_session())
            self._is_recording = True
            await self.write_event(TranscriptStart().event())
            _LOGGER.info(
                "Realtime recording started at %d Hz, %d channels, %d bytes per sample",
                sample_rate,
                audio_channels,
                audio_width,
            )
        except Exception as err:
            _LOGGER.exception("Error starting realtime transcription: %s", err)
            self._is_recording = False
            await self._cleanup_realtime_transcription()

    async def _handle_realtime_audio_chunk(self, chunk: AudioChunk) -> None:
        """Append a Wyoming audio chunk to the OpenAI Realtime input buffer."""
        if not self._is_recording or self._realtime_connection is None:
            _LOGGER.warning("Received realtime audio chunk without an active recording")
            return

        if not chunk.audio:
            return

        try:
            audio = self._convert_audio_to_realtime_pcm(
                chunk.audio,
                sample_rate=self._audio_sample_rate,
                audio_width=self._audio_width,
                audio_channels=self._audio_channels,
            )
            if not audio:
                return

            encoded_audio = base64.b64encode(audio).decode("ascii")
            await self._realtime_connection.input_audio_buffer.append(audio=encoded_audio)
        except Exception as err:
            _LOGGER.exception("Error sending realtime audio chunk: %s", err)
            self._set_realtime_transcription_exception(err)

    async def _handle_realtime_audio_stop(self) -> None:
        """Commit realtime audio and emit the final transcription."""
        if not self._is_recording or self._realtime_connection is None:
            _LOGGER.warning("Received realtime audio stop event without recording")
            await self._cleanup_realtime_transcription()
            return

        self._is_recording = False

        transcript_sent = False
        try:
            if self._realtime_transcript_future is None:
                raise RealtimeTranscriptionError("Realtime transcription future was not initialized")

            await self._realtime_connection.input_audio_buffer.commit()
            transcript = await self._realtime_transcript_future
            if transcript:
                _LOGGER.info("Successfully transcribed realtime stream: %s", _truncate_for_log(transcript))
            else:
                _LOGGER.warning("Received empty realtime transcription result")
            await self.write_event(Transcript(text=transcript).event())
            transcript_sent = True
        except Exception as err:
            _LOGGER.exception("Error during realtime transcription: %s", err)
            if not transcript_sent:
                await self.write_event(Transcript(text="").event())
        finally:
            await self.write_event(TranscriptStop().event())
            await self._cleanup_realtime_transcription()

    async def _enter_realtime_connection(self, connection_manager: Any) -> Any:
        """Enter an SDK realtime connection manager."""
        enter = getattr(connection_manager, "enter", None)
        if enter is not None:
            return await enter()

        return await connection_manager.__aenter__()

    def _connect_realtime_transcription(self) -> Any:
        """Create a Realtime transcription-session websocket connection manager."""
        if self._stt_client is None:
            raise RealtimeTranscriptionError("No STT client configured for realtime transcription")
        return self._stt_client.realtime.connect(extra_query={"intent": "transcription"})

    def _get_realtime_transcription_session(self) -> dict[str, object]:
        """Build a Realtime transcription session update payload."""
        if not self._current_asr_model:
            raise RealtimeTranscriptionError("No ASR model set for realtime transcription")

        transcription: dict[str, object] = {"model": self._current_asr_model.name}
        if self._current_language is not None:
            transcription["language"] = self._current_language
        if self._stt_prompt is not None:
            transcription["prompt"] = self._stt_prompt

        return {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": REALTIME_AUDIO_RATE},
                    "transcription": transcription,
                    "turn_detection": None,
                }
            },
        }

    async def _receive_realtime_transcription_events(self) -> None:
        """Receive Realtime server events and forward transcript deltas to Wyoming."""
        connection = self._realtime_connection
        transcript_future = self._realtime_transcript_future
        if connection is None or transcript_future is None:
            return

        try:
            async for event in connection:
                event_type = getattr(event, "type", "")
                if event_type == "conversation.item.input_audio_transcription.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        _LOGGER.debug("Realtime transcription chunk: %s", delta)
                        await self.write_event(TranscriptChunk(text=delta).event())
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = getattr(event, "transcript", "")
                    if not transcript_future.done():
                        transcript_future.set_result(transcript)
                    return
                elif event_type == "conversation.item.input_audio_transcription.failed":
                    error = RealtimeTranscriptionError(self._get_realtime_event_error_message(event))
                    if not transcript_future.done():
                        transcript_future.set_exception(error)
                    return
                elif event_type == "error":
                    error = RealtimeTranscriptionError(self._get_realtime_event_error_message(event))
                    if not transcript_future.done():
                        transcript_future.set_exception(error)
                    return
        except asyncio.CancelledError:
            raise
        except Exception as err:
            _LOGGER.exception("Error receiving realtime transcription events: %s", err)
            if not transcript_future.done():
                transcript_future.set_exception(err)
        else:
            if not transcript_future.done():
                transcript_future.set_exception(
                    RealtimeTranscriptionError("Realtime connection closed before transcription completed")
                )

    def _get_realtime_event_error_message(self, event: Any) -> str:
        """Extract a readable error message from a Realtime server event."""
        error = getattr(event, "error", None)
        if isinstance(error, dict):
            return str(error.get("message") or error)

        message = getattr(error, "message", None)
        if message:
            return str(message)

        if error:
            return str(error)

        return f"Realtime transcription failed with event type {getattr(event, 'type', 'unknown')}"

    def _set_realtime_transcription_exception(self, err: Exception) -> None:
        """Fail the pending realtime transcription, if any."""
        if self._realtime_transcript_future and not self._realtime_transcript_future.done():
            self._realtime_transcript_future.set_exception(err)

    async def _cleanup_realtime_transcription(self) -> None:
        """Close the realtime connection and background receive task."""
        receive_task = self._realtime_receive_task
        self._realtime_receive_task = None
        if receive_task and not receive_task.done():
            receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receive_task

        transcript_future = self._realtime_transcript_future
        self._realtime_transcript_future = None
        if transcript_future is not None and not transcript_future.done():
            transcript_future.cancel()

        connection_manager = self._realtime_connection_manager
        connection = self._realtime_connection
        self._realtime_connection_manager = None
        self._realtime_connection = None

        if connection_manager is not None:
            await connection_manager.__aexit__(None, None, None)
        elif connection is not None:
            await connection.close()

    def _convert_audio_to_realtime_pcm(
        self, audio: bytes, *, sample_rate: int, audio_width: int, audio_channels: int
    ) -> bytes:
        """Convert incoming Wyoming PCM to 24 kHz mono PCM16 for OpenAI Realtime."""
        if (
            sample_rate == REALTIME_AUDIO_RATE
            and audio_width == REALTIME_AUDIO_WIDTH
            and audio_channels == REALTIME_AUDIO_CHANNELS
        ):
            return audio

        samples = self._decode_pcm_samples(audio, audio_width)
        if not samples:
            return b""

        samples = self._downmix_pcm_samples(samples, audio_channels)
        samples = self._resample_pcm_samples(samples, sample_rate, REALTIME_AUDIO_RATE)
        return self._encode_pcm16_samples(samples)

    def _decode_pcm_samples(self, audio: bytes, audio_width: int) -> list[int]:
        """Decode PCM bytes into signed 16-bit sample values."""
        sample_count = len(audio) // audio_width
        if sample_count == 0:
            return []

        audio = audio[: sample_count * audio_width]
        if audio_width == 1:
            return [(sample - 128) << 8 for sample in audio]

        if audio_width == 2:
            return list(struct.unpack(f"<{sample_count}h", audio))

        if audio_width == 4:
            return [int.from_bytes(audio[i : i + 4], "little", signed=True) >> 16 for i in range(0, len(audio), 4)]

        raise ValueError(f"Unsupported audio sample width for realtime transcription: {audio_width}")

    def _downmix_pcm_samples(self, samples: list[int], audio_channels: int) -> list[int]:
        """Downmix PCM samples to mono."""
        if audio_channels <= 0:
            raise ValueError(f"Unsupported audio channel count for realtime transcription: {audio_channels}")

        if audio_channels == REALTIME_AUDIO_CHANNELS:
            return samples

        frame_count = len(samples) // audio_channels
        mono_samples = []
        for frame_index in range(frame_count):
            start = frame_index * audio_channels
            mono_samples.append(round(sum(samples[start : start + audio_channels]) / audio_channels))
        return mono_samples

    def _resample_pcm_samples(self, samples: list[int], source_rate: int, target_rate: int) -> list[int]:
        """Linearly resample PCM samples to the target sample rate."""
        if source_rate <= 0:
            raise ValueError(f"Unsupported audio sample rate for realtime transcription: {source_rate}")

        if source_rate == target_rate or len(samples) < 2:
            return samples

        output_count = max(1, round(len(samples) * target_rate / source_rate))
        resampled = []
        for output_index in range(output_count):
            source_position = output_index * source_rate / target_rate
            left_index = int(source_position)
            if left_index >= len(samples) - 1:
                resampled.append(samples[-1])
                continue

            fraction = source_position - left_index
            sample = samples[left_index] + (samples[left_index + 1] - samples[left_index]) * fraction
            resampled.append(round(sample))

        return resampled

    def _encode_pcm16_samples(self, samples: list[int]) -> bytes:
        """Encode signed sample values as little-endian PCM16."""
        if not samples:
            return b""

        clamped_samples = (max(-32768, min(32767, int(sample))) for sample in samples)
        return struct.pack(f"<{len(samples)}h", *clamped_samples)

    async def _handle_audio_stop(self) -> None:
        """Handle end of audio stream and perform transcription"""
        if self._realtime_connection is not None or (
            self._current_asr_model and self._is_asr_model_realtime(self._current_asr_model.name)
        ):
            await self._handle_realtime_audio_stop()
            return

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

            if not self._current_asr_model:
                _LOGGER.warning("No ASR model set for transcription")
                return

            if self._stt_client is None:
                _LOGGER.error("No STT client configured for transcription")
                return

            # Send to OpenAI for transcription
            extra_body = self._get_stt_extra_body()
            use_streaming = get_extra_body_boolean_field(
                extra_body,
                field_name="stream",
                default=self._is_asr_model_streaming(self._current_asr_model.name),
                body_name="STT",
            )

            transcription_kwargs = {
                "file": self._wav_buffer,
                "model": self._current_asr_model.name,
                "language": self._current_language if self._current_language is not None else omit,
                "temperature": self._stt_temperature if self._stt_temperature is not None else omit,
                "prompt": self._stt_prompt if self._stt_prompt is not None else omit,
                "response_format": "json",
                "stream": use_streaming if use_streaming else omit,
            }
            if extra_body:
                transcription_kwargs["extra_body"] = extra_body

            transcription = await self._stt_client.audio.transcriptions.create(**transcription_kwargs)

            await self.write_event(TranscriptStart().event())

            if isinstance(transcription, AsyncStream):
                _LOGGER.debug("Handling streaming transcription response")
                full_text = ""
                async for chunk in transcription:
                    if chunk.type == "transcript.text.delta":
                        if chunk.delta:
                            full_text += chunk.delta
                            _LOGGER.debug("Transcribed chunk: %s", chunk.delta)
                            await self.write_event(TranscriptChunk(text=chunk.delta).event())
                if full_text:
                    _LOGGER.info("Successfully transcribed stream: %s", full_text)
                else:
                    _LOGGER.warning(
                        "Received empty transcription from stream."
                        " If this is unexpected, please check your"
                        " STT_STREAMING_MODELS configuration."
                    )
                await self.write_event(Transcript(text=full_text).event())

            elif isinstance(transcription, TranscriptionCreateResponse):
                # Handle non-streaming response
                _LOGGER.debug("Handling non-streaming transcription response")
                if transcription.text:
                    _LOGGER.info("Successfully transcribed: %s", _truncate_for_log(transcription.text))
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

    def _get_asr_program_model_names(self, program_name: str) -> set[str]:
        """Return ASR model names advertised by a specific program."""
        return {
            model.name
            for program in self._wyoming_info.asr
            if getattr(program, "name", None) == program_name
            for model in program.models
        }

    def _has_asr_models(self) -> bool:
        """Return True when STT is configured for this handler."""
        return any(getattr(program, "models", None) for program in self._wyoming_info.asr)

    def _has_tts_voices(self) -> bool:
        """Return True when TTS is configured for this handler."""
        return any(getattr(program, "voices", None) for program in self._wyoming_info.tts)

    def _get_stt_extra_body(self) -> dict[str, object] | None:
        """Get STT extra_body merged with backend-specific fields."""
        extra_body = dict(self._stt_extra_body or {})
        if self._stt_client is not None and self._stt_client.backend == OpenAIBackend.SPEACHES:
            if "vad_filter" not in extra_body:
                extra_body["vad_filter"] = False
                _LOGGER.debug("Adding default vad_filter=False for SPEACHES backend")
        return extra_body or None

    def _get_tts_extra_body(self) -> dict[str, object] | None:
        """Get TTS extra_body for request construction."""
        return dict(self._tts_extra_body) if self._tts_extra_body else None

    def _get_tts_response_format(self) -> str:
        """Get the effective TTS response format expected from the backend."""
        if not self._tts_extra_body:
            return "wav"

        response_format = self._tts_extra_body.get("response_format")
        if isinstance(response_format, str):
            return response_format

        return "wav"

    def _is_asr_model_streaming(self, model_name: str) -> bool:
        """Check if an ASR model supports streaming"""
        for program in self._wyoming_info.asr:
            for model in program.models:
                if model.name == model_name:
                    return program.supports_transcript_streaming
        return False

    def _is_asr_model_realtime(self, model_name: str) -> bool:
        """Check if an ASR model should use OpenAI Realtime transcription."""
        return model_name in self._stt_realtime_models

    def _is_tts_voice_streaming(self, voice_name: str) -> bool:
        """Check if a TTS voice supports streaming synthesis"""
        for program in self._wyoming_info.tts:
            for voice in program.voices:
                if voice.name == voice_name:
                    return getattr(program, "supports_synthesize_streaming", False)
        return False

    def _iter_tts_voices(self):
        """Iterate over configured TTS voices."""
        for program in self._wyoming_info.tts:
            for voice in program.voices:
                yield cast(TtsVoiceModel, voice)

    def _get_pysbd_language(self, language: str | None) -> str:
        """
        Get pysbd-compatible language code.

        Args:
            language (str | None): Language code (e.g., 'en', 'en-US', 'es', etc.)

        Returns:
            str: pysbd-compatible language code, defaults to 'en' if unsupported
        """
        if not language:
            return "en"

        # Extract base language code from potential BCP-47 tags (e.g., 'en-US' -> 'en')
        base_lang = language[:2].lower() if len(language) >= 2 else "en"

        # Test if the language is supported by trying to create a segmenter
        try:
            pysbd.Segmenter(language=base_lang)
            return base_lang
        except (ValueError, KeyError):
            _LOGGER.warning(f"Language '{base_lang}' not supported by pysbd, using English")
            return "en"

    def _chunk_text_for_streaming(
        self, text: str, min_words: int | None = None, max_chars: int | None = None, language: str | None = None
    ) -> list[str]:
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
        segmenter = pysbd.Segmenter(language=pysbd_language, clean=True)
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

    async def _process_ready_sentences(self, sentences: list[str], language: str | None = None) -> bool:
        """
        Process complete sentences for immediate TTS synthesis with concurrent requests.

        This method handles incremental synthesis of complete sentences detected during
        streaming text input. API requests start concurrently for all sentences, with
        sequential playback to maintain correct audio order.

        Concurrency Strategy:
        - Create tasks for ALL sentences immediately (API calls start concurrently)
        - Await tasks in order for sequential playback
        - Semaphore naturally limits concurrency to TTS_CONCURRENT_REQUESTS

        Args:
            sentences (list[str]): Complete sentences ready for synthesis.
            language (str | None): Language code for the sentences.

        Returns:
            bool: True if processing succeeded, False if synthesis was aborted.
        """
        if not sentences or not self._synthesis_voice:
            return True

        try:
            requested_voice = self._synthesis_voice.name
            requested_language = self._synthesis_voice.language
            voice = self._validate_tts_voice_and_language(requested_voice, requested_language)
            if not voice:
                _LOGGER.error("Failed to validate voice for incremental synthesis")
                return await self._abort_synthesis()

            use_streaming = self._is_tts_voice_streaming(voice.name)

            if use_streaming:
                valid_sentences = [s for s in sentences if s.strip()]
                if not valid_sentences:
                    _LOGGER.debug("No non-empty sentences available for incremental synthesis.")
                    return True

                _LOGGER.info("Starting concurrent synthesis for %d sentences", len(valid_sentences))

                # Create ALL tasks with IDs - API calls start concurrently
                # Semaphore limits actual concurrency to TTS_CONCURRENT_REQUESTS
                synthesis_tasks = [
                    (
                        f"sentence_{i}",
                        asyncio.create_task(
                            self._get_tts_audio_stream(sentence, voice, task_id=f"sentence_{i}"),
                            name=f"incremental_sentence_{i}",
                        ),
                    )
                    for i, sentence in enumerate(valid_sentences)
                ]

                # Await tasks IN ORDER for sequential playback
                # Enable streaming for whichever task we're currently awaiting
                for i, (task_id, task) in enumerate(synthesis_tasks):
                    sentence_preview = _truncate_for_log(valid_sentences[i], 50)
                    _LOGGER.debug("Processing sentence %d: %s", i + 1, sentence_preview)

                    self._allow_streaming_task_id = task_id
                    try:
                        result = await task
                    except TtsStreamError as err:
                        _LOGGER.error(
                            "Failed to synthesize sentence %d (%s) with voice %s: %s",
                            i + 1,
                            err.chunk_preview,
                            err.voice,
                            err,
                        )
                        return await self._abort_synthesis()
                    except Exception as err:
                        _LOGGER.exception(
                            "Unexpected error while synthesizing sentence %d (%s): %s",
                            i + 1,
                            sentence_preview,
                            err,
                        )
                        return await self._abort_synthesis()
                    finally:
                        self._allow_streaming_task_id = None

                    if result.streamed:
                        _LOGGER.debug("Sentence %d streamed directly with minimal latency", i + 1)
                        # Timestamp and audio_started already updated by _stream_tts_audio_incremental
                        continue

                    # Otherwise, task completed and buffered - stream the buffered data now
                    audio_data = result.audio
                    if not audio_data:
                        _LOGGER.error(
                            "Buffered synthesis returned no audio for sentence %d (%s)",
                            i + 1,
                            sentence_preview,
                        )
                        return await self._abort_synthesis()

                    chunk_timestamp = await self._stream_audio_to_wyoming(
                        audio_data,
                        is_first_chunk=(not self._audio_started),
                        start_timestamp=self._current_timestamp,
                    )

                    if chunk_timestamp is None:
                        _LOGGER.error("Failed to stream sentence %d to Wyoming", i + 1)
                        return await self._abort_synthesis()

                    self._current_timestamp = chunk_timestamp
                    self._audio_started = True
                    _LOGGER.debug(
                        "Successfully streamed buffered sentence %d, timestamp: %.2f",
                        i + 1,
                        chunk_timestamp,
                    )

            return True
        except Exception as e:
            _LOGGER.exception("Error processing ready sentences: %s", e)
            return await self._abort_synthesis()

    async def _stream_tts_audio_incremental(self, text: str, voice: TtsVoiceModel) -> float | None:
        """
        Stream TTS audio directly to Wyoming for incremental synthesis.

        This method is used when a sentence synthesis task is still running when we await it.
        It streams audio chunks as they arrive from the OpenAI API, minimizing latency.

        Args:
            text (str): Text to synthesize.
            voice (TtsVoiceModel): Voice to use for synthesis.

        Returns:
            float | None: Final timestamp after streaming, or None on error.
        """
        timestamp = await self._stream_tts_audio(
            voice=voice, text=text, send_audio_start=(not self._audio_started), start_timestamp=self._current_timestamp
        )

        if timestamp is not None:
            self._current_timestamp = timestamp
            self._audio_started = True

        return timestamp

    async def _abort_synthesis(self) -> bool:
        """Abort the current synthesis session, emitting stop events and resetting state."""
        if self._audio_started:
            await self.write_event(AudioStop(timestamp=int(self._current_timestamp)).event())

        await self.write_event(SynthesizeStopped().event())

        self._audio_started = False
        self._current_timestamp = 0
        self._allow_streaming_task_id = None
        self._is_synthesizing = False
        self._synthesis_buffer = []
        self._text_accumulator = ""
        self._ready_chunks = []
        self._pysbd_segmenters.clear()
        self._synthesis_voice = None

        return False

    def _log_unsupported_asr_model(self, model_name: str | None = None):
        """Log an unsupported ASR model"""
        if model_name:
            _LOGGER.warning("Unsupported ASR model: %s", model_name)
        else:
            _LOGGER.warning("No ASR models specified")

    def _is_asr_language_supported(self, language: str | None, model: AsrModel) -> bool:
        """Check if a language is supported by an ASR model"""
        return not language or not model.languages or language in model.languages

    def _log_unsupported_asr_language(self, model_name: str | None, language: str | None):
        """Log an unsupported ASR language"""
        _LOGGER.error("Unsupported ASR model %s for language %s", model_name, language)

    def _get_voice(self, name: str | None = None) -> TtsVoiceModel | None:
        """Get a TTS voice by name or None"""
        for voice in self._iter_tts_voices():
            if not name or voice.name == name:
                return voice
        return None

    def _get_voices_by_backend_name(self, backend_voice_name: str) -> list[TtsVoiceModel]:
        """Get TTS voices by the raw backend voice identifier."""
        return [
            voice
            for voice in self._iter_tts_voices()
            if getattr(voice, "backend_voice_name", voice.name) == backend_voice_name
        ]

    def _get_backend_voice_name(self, voice: TtsVoice) -> str:
        """Get the raw backend voice identifier for synthesis requests."""
        return getattr(voice, "backend_voice_name", voice.name)

    def _is_tts_language_supported(self, language: str, voice: TtsVoice) -> bool:
        """Check if a language is supported by a TTS voice"""
        return not voice.languages or language in voice.languages

    def _validate_tts_voice_and_language(
        self, requested_voice: str | None, requested_language: str | None
    ) -> TtsVoiceModel | None:
        """
        Validate and get a TTS voice by name and language.

        Args:
            requested_voice (str | None): The requested voice name.
            requested_language (str | None): The requested language.

        Returns:
            TtsVoiceModel | None: The validated voice, or None if validation failed.
        """
        voice = self._get_voice(requested_voice)
        if voice:
            if not self._validate_tts_language(requested_language, voice):
                return None
            return voice

        if not requested_voice:
            self._log_unsupported_voice(requested_voice)
            return None

        backend_matches = self._get_voices_by_backend_name(requested_voice)
        if not backend_matches:
            self._log_unsupported_voice(requested_voice)
            return None

        if len(backend_matches) == 1:
            voice = backend_matches[0]
            if not self._validate_tts_language(requested_language, voice):
                return None
            return voice

        compatible_matches = backend_matches
        if requested_language:
            language_matches = [
                voice for voice in backend_matches if self._is_tts_language_supported(requested_language, voice)
            ]
            if language_matches:
                compatible_matches = language_matches

        voice = compatible_matches[0]
        self._log_legacy_voice_fallback(requested_voice, voice, backend_matches)
        if not self._validate_tts_language(requested_language, voice):
            return None
        return voice

    def _validate_tts_language(self, language: str | None, voice: TtsVoice) -> bool:
        """Validate if a language is supported by a TTS voice.

        Returns True if supported. If no language is specified, also returns True.
        """
        if language and not self._is_tts_language_supported(language, voice):
            _LOGGER.error(
                f"Language {language} is not supported for voice {voice.name}. Available languages: {voice.languages}"
            )
            return False
        return True

    def _log_unsupported_voice(self, requested_voice: str | None) -> None:
        """Log an error message if a voice is not supported"""
        if requested_voice:
            available = [voice.name for program in self._wyoming_info.tts for voice in program.voices]
            _LOGGER.error(f"Voice {requested_voice} is not supported. Available voices: {available}")
        else:
            _LOGGER.error("No TTS voices specified")

    def _log_legacy_voice_fallback(
        self, requested_voice: str, selected_voice: TtsVoiceModel, matches: list[TtsVoiceModel]
    ) -> None:
        """Log when a legacy raw voice name falls back to a configured voice."""
        available = [voice.name for voice in matches]
        _LOGGER.warning(
            "Voice %s matched multiple configured voices. Falling back to %s for backward compatibility. "
            "Update the client to use one of: %s",
            requested_voice,
            selected_voice.name,
            available,
        )

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
                await self.write_event(AudioStop(timestamp=int(final_timestamp)).event())
                _LOGGER.info("Successfully synthesized: %s", _truncate_for_log(synthesize.text))
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

        # Reset incremental detection state
        self._text_accumulator = ""
        self._ready_chunks = []
        self._pysbd_segmenters.clear()  # Clear segmenter cache for new session
        self._audio_started = False  # Reset audio started flag
        self._current_timestamp = 0  # Reset timestamp for new synthesis session

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
        """Handle text chunk during streaming synthesis with incremental sentence detection"""
        if not self._is_synthesizing:
            _LOGGER.warning("Received synthesize-chunk without active synthesis")
            return False

        chunk_text = synthesize_chunk.text if synthesize_chunk.text else ""
        _LOGGER.debug("Received synthesis chunk: '%s' (length: %d)", _truncate_for_log(chunk_text, 50), len(chunk_text))

        # Store in buffer for fallback compatibility
        self._synthesis_buffer.append(synthesize_chunk.text)

        # Add to accumulator for sentence detection across chunks
        self._text_accumulator += chunk_text

        # Get or create segmenter for the current language
        requested_language = self._synthesis_voice.language if self._synthesis_voice else None
        pysbd_language = self._get_pysbd_language(requested_language)

        # Use cached segmenter or create a new one
        if pysbd_language not in self._pysbd_segmenters:
            _LOGGER.debug("Creating new pysbd segmenter for language: %s", pysbd_language)
            self._pysbd_segmenters[pysbd_language] = pysbd.Segmenter(language=pysbd_language, clean=True)

        segmenter = self._pysbd_segmenters[pysbd_language]

        # Segment the entire accumulated text
        sentences: list[str] = list(segmenter.segment(self._text_accumulator))

        # Process complete sentences (all but the last one)
        if len(sentences) > 1:
            ready_sentences = sentences[:-1]

            # Keep only the last sentence in the accumulator
            self._text_accumulator = sentences[-1]

            _LOGGER.info(
                "Detected %d ready sentences for immediate synthesis: %s",
                len(ready_sentences),
                [_truncate_for_log(s, 30) for s in ready_sentences],
            )
            if not await self._process_ready_sentences(ready_sentences, requested_language):
                return False
        else:
            _LOGGER.debug(
                "No complete sentences ready yet, accumulator has: '%s'", _truncate_for_log(self._text_accumulator)
            )

        return True

    async def _handle_synthesize_stop(self) -> bool:
        """Handle end of streaming synthesis"""
        if not self._is_synthesizing:
            _LOGGER.warning("Received synthesize-stop without active synthesis")
            return False

        self._is_synthesizing = False

        # Process any remaining text in the accumulator (even if it's incomplete)
        # This is the final text, so we process it regardless of sentence completion
        if self._text_accumulator.strip():
            _LOGGER.info("Processing final remaining text: '%s'", _truncate_for_log(self._text_accumulator))
            requested_language = self._synthesis_voice.language if self._synthesis_voice else None
            if not await self._process_ready_sentences([self._text_accumulator], requested_language):
                return False

        # Get accumulated text and voice for fallback
        full_text = "".join(self._synthesis_buffer)
        voice_info = self._synthesis_voice

        _LOGGER.debug("Streaming synthesis completed with text: %s", _truncate_for_log(full_text))

        # Clear synthesis state early
        self._synthesis_buffer = []
        self._synthesis_voice = None
        self._text_accumulator = ""
        self._ready_chunks = []
        self._pysbd_segmenters.clear()  # Clear segmenter cache

        # Send audio stop if we processed any audio incrementally
        if self._audio_started:
            await self.write_event(AudioStop(timestamp=int(self._current_timestamp)).event())
            await self.write_event(SynthesizeStopped().event())
            _LOGGER.info(
                "Successfully completed incremental streaming synthesis, final timestamp: %.2f", self._current_timestamp
            )
            self._audio_started = False  # Reset for next session
            self._current_timestamp = 0  # Reset for next session
            self._pysbd_segmenters.clear()  # Clear segmenter cache
            return True  # Exit early to prevent duplicate events

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
                    full_text, self._tts_streaming_min_words, self._tts_streaming_max_chars, requested_language
                )
                _LOGGER.debug("Text chunked into %d parts for streaming synthesis", len(chunks))

                # Create ALL tasks with IDs - API calls start concurrently
                # Semaphore limits actual concurrency to TTS_CONCURRENT_REQUESTS
                _LOGGER.info("Starting concurrent synthesis for %d chunks", len(chunks))
                synthesis_tasks = [
                    (
                        f"fallback_chunk_{i}",
                        asyncio.create_task(
                            self._get_tts_audio_stream(chunk, voice, task_id=f"fallback_chunk_{i}"), name=f"chunk_{i}"
                        ),
                    )
                    for i, chunk in enumerate(chunks)
                ]

                # Stream results sequentially to preserve playback order
                # Enable streaming for whichever task we're currently awaiting
                total_timestamp = 0
                for i, (task_id, task) in enumerate(synthesis_tasks):
                    chunk_preview = _truncate_for_log(chunks[i], 50)
                    _LOGGER.debug("Streaming chunk %d/%d to Wyoming", i + 1, len(chunks))

                    self._allow_streaming_task_id = task_id
                    try:
                        result = await task
                    except TtsStreamError as err:
                        _LOGGER.error(
                            "Failed to synthesize chunk %d (%s) with voice %s: %s",
                            i + 1,
                            err.chunk_preview,
                            err.voice,
                            err,
                        )
                        return await self._abort_synthesis()
                    except Exception as err:
                        _LOGGER.exception(
                            "Unexpected error while synthesizing chunk %d (%s): %s",
                            i + 1,
                            chunk_preview,
                            err,
                        )
                        return await self._abort_synthesis()
                    finally:
                        self._allow_streaming_task_id = None

                    if result.streamed:
                        _LOGGER.debug("Chunk %d streamed directly", i + 1)
                        # Update timestamp from streamed audio
                        total_timestamp = self._current_timestamp
                        continue

                    # Otherwise, stream the buffered data
                    audio_data = result.audio
                    if not audio_data:
                        _LOGGER.error(
                            "Buffered synthesis returned no audio for chunk %d (%s)",
                            i + 1,
                            chunk_preview,
                        )
                        return await self._abort_synthesis()

                    chunk_timestamp = await self._stream_audio_to_wyoming(
                        audio_data,
                        is_first_chunk=(i == 0),
                        start_timestamp=total_timestamp,
                    )

                    if chunk_timestamp is None:
                        _LOGGER.error("Failed to stream chunk %d to Wyoming", i + 1)
                        return await self._abort_synthesis()

                    total_timestamp = chunk_timestamp

                # Send final audio stop
                await self.write_event(AudioStop(timestamp=int(total_timestamp)).event())
                _LOGGER.info("Successfully completed concurrent streaming synthesis: %s", _truncate_for_log(full_text))
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

    async def _get_tts_audio_stream(
        self, text: str, voice: TtsVoiceModel, task_id: str | None = None
    ) -> TtsStreamResult:
        """
        Get TTS audio stream from OpenAI for a text chunk (parallel-safe).

        If task_id matches _allow_streaming_task_id, streams audio directly to Wyoming
        as chunks arrive (minimal latency). Otherwise, buffers complete audio before returning.

        Args:
            text (str): Text chunk to synthesize.
            voice (TtsVoiceModel): Voice to use for synthesis.
            task_id (str | None): Optional task identifier for streaming coordination.

        Returns:
            TtsStreamResult: Container with streaming status and optional buffered audio.
        """
        chunk_preview = _truncate_for_log(text, 50)

        try:
            # Check if this task is allowed to stream directly
            should_stream = task_id is not None and task_id == self._allow_streaming_task_id

            if self._tts_client is None:
                raise TtsStreamError("TTS client is not configured", chunk_preview, voice.name)

            if should_stream:
                # Stream directly to Wyoming (no buffering) - minimal latency
                _LOGGER.debug("Streaming chunk directly (task %s): %s", task_id, chunk_preview)
                timestamp = await self._stream_tts_audio_incremental(text, voice)
                if timestamp is None:
                    raise TtsStreamError("OpenAI returned no audio while streaming chunk", chunk_preview, voice.name)
                _LOGGER.debug("Completed direct streaming for chunk: %s", chunk_preview)
                return TtsStreamResult(streamed=True)

            # Buffer audio (default behavior for parallel tasks)
            chunks: list[bytes] = []
            async with self._tts_semaphore:
                request_kwargs = {
                    "model": voice.model_name,
                    "voice": self._get_backend_voice_name(voice),
                    "input": text,
                    "response_format": "wav",
                    "speed": self._tts_speed if self._tts_speed is not None else omit,
                    "instructions": self._tts_instructions if self._tts_instructions is not None else omit,
                }
                if extra_body := self._get_tts_extra_body():
                    request_kwargs["extra_body"] = extra_body

                async with self._tts_client.audio.speech.with_streaming_response.create(**request_kwargs) as response:
                    async for chunk in response.iter_bytes(chunk_size=TTS_CHUNK_SIZE):
                        chunks.append(chunk)

            audio_data = b"".join(chunks)
            if not audio_data:
                raise TtsStreamError("OpenAI returned empty audio response", chunk_preview, voice.name)

            _LOGGER.debug("Completed buffered synthesis for chunk: %s", chunk_preview)
            return TtsStreamResult(streamed=False, audio=audio_data)

        except TtsStreamError:
            raise
        except Exception as exc:
            _LOGGER.exception("Error getting TTS audio stream for %s: %s", chunk_preview, exc)
            raise TtsStreamError("Unexpected error while retrieving TTS audio", chunk_preview, voice.name) from exc

    async def _stream_audio_to_wyoming(
        self, audio_data: bytes, is_first_chunk: bool, start_timestamp: float
    ) -> float | None:
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

            # Try to parse WAV header
            wav_params = self._parse_wav_header(audio_data)
            if wav_params:
                audio_rate, audio_channels, audio_width, data_offset, data_size = wav_params
                available_audio = audio_data[data_offset:]
                if len(available_audio) < data_size:
                    _LOGGER.warning(
                        "TTS WAV response ended after %d of %d declared PCM bytes",
                        len(available_audio),
                        data_size,
                    )
                audio_data = available_audio[:data_size]
                _LOGGER.debug(
                    "Detected audio format: %d Hz, %d channels, %d bytes/sample, "
                    "header offset: %d, PCM size: %d bytes",
                    audio_rate,
                    audio_channels,
                    audio_width,
                    data_offset,
                    data_size,
                )
            else:
                _LOGGER.debug("Could not parse WAV header, using defaults: %d Hz", TTS_AUDIO_RATE)

            # Send audio start if requested
            if is_first_chunk:
                await self.write_event(AudioStart(rate=audio_rate, width=audio_width, channels=audio_channels).event())

            # Send audio chunk (header stripped if present)
            if audio_data:
                await self.write_event(
                    AudioChunk(
                        audio=audio_data,
                        rate=audio_rate,
                        width=audio_width,
                        channels=audio_channels,
                        timestamp=int(timestamp),
                    ).event()
                )
                timestamp = self._advance_audio_timestamp(
                    timestamp,
                    audio_data=audio_data,
                    audio_rate=audio_rate,
                    audio_width=audio_width,
                    audio_channels=audio_channels,
                )

            return timestamp

        except Exception as e:
            _LOGGER.exception("Error streaming audio to Wyoming: %s", e)
            return None

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
            await self.write_event(AudioStop(timestamp=int(final_timestamp)).event())
            _LOGGER.info("Successfully synthesized non-streaming: %s", _truncate_for_log(text))
            return True
        return False

    async def _stream_tts_audio(
        self, voice: TtsVoiceModel, text: str, send_audio_start: bool = True, start_timestamp: float = 0
    ) -> float | None:
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
            audio_rate = TTS_AUDIO_RATE
            audio_width = DEFAULT_AUDIO_WIDTH
            audio_channels = DEFAULT_AUDIO_CHANNELS
            timestamp = start_timestamp
            awaiting_wav_header = self._get_tts_response_format() == "wav"
            pending_header = b""
            remaining_wav_data: int | None = None

            if self._tts_client is None:
                _LOGGER.error("No TTS client configured for synthesis")
                return None

            async with self._tts_semaphore:
                request_kwargs = {
                    "model": voice.model_name,
                    "voice": self._get_backend_voice_name(voice),
                    "input": text,
                    "response_format": "wav",
                    "speed": self._tts_speed if self._tts_speed is not None else omit,
                    "instructions": self._tts_instructions if self._tts_instructions is not None else omit,
                }
                if extra_body := self._get_tts_extra_body():
                    request_kwargs["extra_body"] = extra_body

                async with self._tts_client.audio.speech.with_streaming_response.create(**request_kwargs) as response:
                    async for chunk in response.iter_bytes(chunk_size=TTS_CHUNK_SIZE):
                        if awaiting_wav_header:
                            pending_header += chunk
                            wav_params = self._parse_wav_header(pending_header)
                            if wav_params:
                                audio_rate, audio_channels, audio_width, data_offset, data_size = wav_params
                                available_audio = pending_header[data_offset:]
                                audio_data = available_audio[:data_size]
                                remaining_wav_data = data_size - len(audio_data)
                                pending_header = b""
                                awaiting_wav_header = False
                                _LOGGER.debug(
                                    "Detected audio format: %d Hz, %d channels, %d bytes/sample, "
                                    "header offset: %d, PCM size: %d bytes",
                                    audio_rate,
                                    audio_channels,
                                    audio_width,
                                    data_offset,
                                    data_size,
                                )
                            elif len(pending_header) <= TTS_WAV_HEADER_MAX_BYTES:
                                continue
                            else:
                                _LOGGER.warning(
                                    "Could not parse WAV header after buffering %d bytes, falling back to raw PCM",
                                    len(pending_header),
                                )
                                audio_data = pending_header
                                pending_header = b""
                                awaiting_wav_header = False
                        else:
                            audio_data = chunk
                            if remaining_wav_data is not None:
                                audio_data = audio_data[:remaining_wav_data]
                                remaining_wav_data -= len(audio_data)

                        if send_audio_start:
                            await self.write_event(
                                AudioStart(rate=audio_rate, width=audio_width, channels=audio_channels).event()
                            )
                            send_audio_start = False

                        # Send audio chunk after any buffered WAV header bytes have been removed.
                        if audio_data:
                            await self.write_event(
                                AudioChunk(
                                    audio=audio_data,
                                    rate=audio_rate,
                                    width=audio_width,
                                    channels=audio_channels,
                                    timestamp=int(timestamp),
                                ).event()
                            )
                            timestamp = self._advance_audio_timestamp(
                                timestamp,
                                audio_data=audio_data,
                                audio_rate=audio_rate,
                                audio_width=audio_width,
                                audio_channels=audio_channels,
                            )

            if remaining_wav_data:
                _LOGGER.warning(
                    "TTS WAV response ended with %d declared PCM bytes missing",
                    remaining_wav_data,
                )

            if awaiting_wav_header and pending_header:
                _LOGGER.warning(
                    "TTS response ended before a complete WAV header was available, falling back to raw PCM"
                )
                if send_audio_start:
                    await self.write_event(
                        AudioStart(rate=audio_rate, width=audio_width, channels=audio_channels).event()
                    )

                await self.write_event(
                    AudioChunk(
                        audio=pending_header,
                        rate=audio_rate,
                        width=audio_width,
                        channels=audio_channels,
                        timestamp=int(timestamp),
                    ).event()
                )
                timestamp = self._advance_audio_timestamp(
                    timestamp,
                    audio_data=pending_header,
                    audio_rate=audio_rate,
                    audio_width=audio_width,
                    audio_channels=audio_channels,
                )

            return timestamp

        except Exception as e:
            _LOGGER.exception("Error streaming TTS audio: %s", e)
            return None

    def _advance_audio_timestamp(
        self,
        timestamp: float,
        *,
        audio_data: bytes,
        audio_rate: int,
        audio_width: int,
        audio_channels: int,
    ) -> float:
        """Advance a Wyoming audio timestamp using PCM frame count."""
        frame_size = audio_width * audio_channels
        if frame_size <= 0:
            return timestamp

        actual_frames = len(audio_data) // frame_size
        return timestamp + (actual_frames / audio_rate) * 1000

    def _parse_wav_header(self, wav_data: bytes) -> tuple[int, int, int, int, int] | None:
        """
        Parse WAV header to extract the PCM format, data offset, and data size.
        Returns (sample_rate, channels, sample_width, data_offset, data_size) or None if parsing fails.
        """
        try:
            # Create a BytesIO object from the data
            wav_io = io.BytesIO(wav_data)

            # Open with wave module
            with wave.open(wav_io, "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                data_size = wav_file.getnframes() * channels * sample_width

                # Get the current position which should be at the start of audio data
                data_offset = wav_io.tell()

                return sample_rate, channels, sample_width, data_offset, data_size
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
