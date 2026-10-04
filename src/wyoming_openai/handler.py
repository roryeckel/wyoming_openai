import asyncio
import base64
import contextlib
import io
import logging
import struct
import unicodedata
import wave
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Any, cast

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
from wyoming.info import AsrModel, AsrProgram, Describe, Info, SelectProgram, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
    SynthesizeTextFormat,
    SynthesizeVoice,
)
from yasbd import BoundaryDetector, get_supported_langs

from .compatibility import CustomAsyncOpenAI, OpenAIBackend, TtsVoiceModel, parse_tts_voice_name
from .const import (
    OPENAI_PLURAL_LANGUAGE_STT_MODEL_PREFIXES,
    REALTIME_AUDIO_RATE,
    REALTIME_TTS_AUDIO_FORMAT,
    REALTIME_TTS_EVENT_TIMEOUT,
    REALTIME_TTS_INSTRUCTIONS,
)
from .utilities import (
    NamedBytesIO,
    SsmlTextTransformer,
    clamp_realtime_tts_speed,
    get_extra_body_boolean_field,
    get_realtime_tts_audio_output,
    get_realtime_tts_speed,
    strip_ssml,
    validate_realtime_stt_extra_body,
    validate_realtime_tts_extra_body,
    validate_stt_extra_body,
    validate_tts_extra_body,
)

_LOGGER = logging.getLogger(__name__)


def _truncate_for_log(text: str, max_length: int = 100) -> str:
    """Truncate text for logging, adding ellipsis only if truncated."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def _normalize_for_comparison(text: str) -> str:
    """
    Reduce text to case-folded alphanumerics for comparing requested and spoken text.
    Numbers and symbols a transcript spells out ("50%" as "fifty percent") still compare as different.
    """
    # NFKC so composed and decomposed forms of the same letter compare equal
    return "".join(char for char in unicodedata.normalize("NFKC", text).casefold() if char.isalnum())


# Symbols that are read aloud on their own ("plus", "and", "percent"), unlike punctuation or emoji
_SPEAKABLE_SYMBOLS = frozenset("+=±×÷&@%#")


def _has_speakable_content(text: str) -> bool:
    """Check if text has anything to read aloud: a letter, a digit, or a symbol with a spoken name."""
    return any(
        char.isalnum() or char in _SPEAKABLE_SYMBOLS or unicodedata.category(char) == "Sc"  # Sc: currency
        for char in text
    )


DEFAULT_AUDIO_WIDTH = 2  # 16-bit audio
DEFAULT_AUDIO_CHANNELS = 1  # Mono audio
DEFAULT_ASR_AUDIO_RATE = 16000  # Hz (Wyoming default)
REALTIME_AUDIO_WIDTH = 2  # 16-bit audio
REALTIME_AUDIO_CHANNELS = 1  # Mono audio
TTS_AUDIO_RATE = 24000  # Hz (OpenAI spec, fallback)
TTS_CHUNK_SIZE = 2048  # Magical guess - but must be larger than 44 bytes for a potential WAV header
TTS_CONCURRENT_REQUESTS = 3  # Default number of concurrent OpenAI TTS requests per connection when streaming sentences
TTS_WAV_HEADER_MAX_BYTES = 65536  # Bound header buffering if a backend never yields a complete WAV header
WAV_UNBOUNDED_SIZE = 0xFFFFFFFF  # Streaming WAV data chunk size sentinel

@dataclass(frozen=True)
class TtsStreamResult:
    """Container for TTS streaming outcomes."""

    streamed: bool
    audio: bytes | None = None
    skipped: bool = False  # The backend had nothing to say for text with nothing to read aloud


@dataclass(frozen=True)
class TtsAudioFormat:
    """How the audio bytes a voice's transport yields are framed."""

    headerless: bool  # Raw PCM16 mono rather than a WAV file
    rate: int  # Hz; for WAV only a fallback until the header is parsed


def _scan_riff_chunks(data: bytes) -> bool | None:
    """
    Check if bytes are nothing but whole RIFF metadata chunks.
    Returns True when they are, False when they cannot be, and None when more bytes could still complete one.
    """
    offset = 0
    while offset < len(data):
        chunk_id = data[offset : offset + 4]
        # Any chunk id is four printable ASCII characters ("LIST", "JUNK", "cue "), which PCM rarely starts with
        if not all(0x20 <= byte <= 0x7E for byte in chunk_id):
            return False
        if len(data) - offset < 8:
            return None
        chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
        if chunk_size > TTS_WAV_HEADER_MAX_BYTES:
            # PCM that happens to start with a chunk id; real metadata is small
            return False
        end = offset + 8 + chunk_size
        if end > len(data):
            return None
        # Odd-sized chunks are followed by a pad byte
        offset = end + (chunk_size & 1 if end < len(data) else 0)
    return True


class _WavFramer:
    """Turns the bytes of a TTS response into PCM, for the streamed and the buffered path alike."""

    def __init__(
        self,
        audio_format: TtsAudioFormat,
        parse_header: Callable[[bytes], tuple[int, int, int, int, int | None] | None],
    ) -> None:
        self.rate = audio_format.rate
        self.width = DEFAULT_AUDIO_WIDTH
        self.channels = DEFAULT_AUDIO_CHANNELS
        self.awaiting_header = not audio_format.headerless
        self.declared_size: int | None = None  # PCM bytes the header declares; None when unbounded or unparsed
        self.declared_empty = False  # The header declares zero samples or an unknown length
        self._parse_header = parse_header
        self._pending = b""  # Bytes held until the header, or what follows an empty data chunk, is understood
        self._remaining: int | None = None  # Declared PCM bytes not received yet; None when unbounded
        self._check_trailer = False  # Still deciding if the bytes after a zero-size header are metadata

    @property
    def missing_bytes(self) -> int:
        """Declared PCM bytes that have not arrived."""
        return self._remaining or 0

    def feed(self, chunk: bytes) -> bytes:
        """Take the next response bytes and return the PCM to play, which is empty while the header is incomplete."""
        if not self.awaiting_header:
            return self._bound(chunk)

        self._pending += chunk
        wav_params = self._parse_header(self._pending)
        if wav_params:
            self.rate, self.channels, self.width, data_offset, data_size = wav_params
            available_audio = self._pending[data_offset:]
            self._pending = b""
            self.awaiting_header = False
            self.declared_empty = not data_size
            # A zero size is also what servers write when streaming audio of unknown length
            self.declared_size = data_size or None
            self._remaining = self.declared_size
            self._check_trailer = data_size == 0
            _LOGGER.debug(
                "Detected audio format: %d Hz, %d channels, %d bytes/sample, header offset: %d, PCM size: %s",
                self.rate,
                self.channels,
                self.width,
                data_offset,
                f"{data_size} bytes" if data_size else "unknown (unbounded or zero size)",
            )
            return self._bound(available_audio)

        if len(self._pending) <= TTS_WAV_HEADER_MAX_BYTES:
            return b""

        _LOGGER.warning(
            "Could not parse WAV header after buffering %d bytes, falling back to raw PCM", len(self._pending)
        )
        return self.finish()

    def finish(self) -> bytes:
        """
        Return the bytes still held when the response ends, to be played as raw PCM: a response without a
        WAV header, or bytes after an empty data chunk that did not turn out to be whole metadata chunks.
        """
        audio, self._pending = self._pending, b""
        is_metadata = self._check_trailer and _scan_riff_chunks(audio) is True
        self.awaiting_header = False
        self._check_trailer = False
        return b"" if is_metadata else audio

    def _bound(self, audio: bytes) -> bytes:
        """Keep only the PCM the header covers."""
        if self._check_trailer:
            # A zero size means either an empty file followed by metadata chunks or audio of unknown length.
            # Bytes are only held while they are consistent with whole metadata chunks, so PCM that merely
            # starts with a chunk id is released as soon as it stops looking like one
            self._pending += audio
            if _scan_riff_chunks(self._pending) is not False and len(self._pending) <= TTS_WAV_HEADER_MAX_BYTES:
                return b""
            self._check_trailer = False
            audio, self._pending = self._pending, b""
        if self._remaining is None:
            return audio
        audio = audio[: self._remaining]
        self._remaining -= len(audio)
        return audio


class TtsStreamError(Exception):
    """Raised when TTS streaming fails for a specific text chunk."""

    def __init__(self, message: str, chunk_preview: str, voice: str):
        super().__init__(message)
        self.chunk_preview = chunk_preview
        self.voice = voice


class RealtimeTranscriptionError(Exception):
    """Raised when OpenAI Realtime transcription fails."""


class RealtimeSynthesisError(Exception):
    """Raised when OpenAI Realtime speech synthesis fails."""


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
        stt_realtime_extra_body: dict[str, object] | None = None,
        tts_speed: float | None = None,
        tts_instructions: str | None = None,
        tts_extra_body: dict[str, object] | None = None,
        tts_realtime_models: list[str] | set[str] | None = None,
        tts_realtime_extra_body: dict[str, object] | None = None,
        tts_streaming_min_words: int | None = None,
        tts_streaming_max_chars: int | None = None,
        tts_concurrent_requests: int = TTS_CONCURRENT_REQUESTS,
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
            stt_realtime_extra_body (dict[str, object] | None): Optional fields merged into Realtime
                transcription settings.
            tts_speed (float | None): The speed for TTS, or None for default.
            tts_instructions (str | None): Optional instructions for TTS.
            tts_extra_body (dict[str, object] | None): Optional JSON body fields merged into TTS requests.
            tts_realtime_models (list[str] | set[str] | None): TTS models that synthesize over OpenAI Realtime.
            tts_realtime_extra_body (dict[str, object] | None): Optional fields merged into Realtime TTS sessions.
            tts_streaming_min_words (int | None): Minimum words per chunk for streaming TTS.
            tts_streaming_max_chars (int | None): Maximum characters per chunk for streaming TTS.
            tts_concurrent_requests (int): Maximum simultaneous TTS requests per connection.
            Note: The caller owns the STT/TTS clients and is responsible for closing them.
            **kwargs: Arbitrary keyword arguments for the superclass.
        """
        super().__init__(*args, **kwargs)
        self._wyoming_info = info

        # Connection-lifetime program selection (select-program event). The
        # server constructs a fresh handler per connection, so no reset needed.
        self._selected_asr_program: AsrProgram | None = None
        self._selected_tts_program: TtsProgram | None = None

        self._stt_client = stt_client
        self._stt_temperature = stt_temperature
        self._stt_prompt = stt_prompt
        self._stt_extra_body = dict(stt_extra_body) if stt_extra_body else None
        self._stt_realtime_extra_body = dict(stt_realtime_extra_body) if stt_realtime_extra_body else None
        self._stt_realtime_models = set(stt_realtime_models or self._get_asr_program_model_names("openai-realtime"))
        if self._has_asr_models():
            validate_stt_extra_body(self._stt_extra_body)
            validate_realtime_stt_extra_body(self._stt_realtime_extra_body)

        self._tts_client = tts_client
        self._tts_speed = tts_speed
        self._tts_instructions = tts_instructions
        self._tts_extra_body = dict(tts_extra_body) if tts_extra_body else None
        self._tts_realtime_models = set(tts_realtime_models or [])
        self._tts_realtime_extra_body = dict(tts_realtime_extra_body) if tts_realtime_extra_body else None
        if self._has_tts_voices():
            validate_tts_extra_body(self._tts_extra_body)
            if self._tts_realtime_models:
                validate_realtime_tts_extra_body(self._tts_realtime_extra_body)
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
        self._resolved_synthesis_voice: TtsVoiceModel | None = None  # Resolved once per streaming synthesis
        self._synthesis_text_format: str | SynthesizeTextFormat | None = None
        self._ssml_transformer: SsmlTextTransformer | None = None
        self._is_synthesizing: bool = False

        # State for incremental sentence detection
        self._text_accumulator: str = ""
        self._ready_chunks: list[str] = []
        self._segmenters: dict[str, BoundaryDetector] = {}  # Cache sentence segmenters per language
        self._audio_started: bool = False  # Track if AudioStart has been sent
        self._current_timestamp: float = 0  # Track timestamp continuity across chunks

        if tts_concurrent_requests < 1:
            raise ValueError(f"tts_concurrent_requests must be at least 1, got {tts_concurrent_requests}")
        self._tts_semaphore = asyncio.Semaphore(tts_concurrent_requests)
        self._allow_streaming_task_id: str | None = None  # ID of task allowed to stream directly
        self._synthesis_tasks: set[asyncio.Task[Any]] = set()  # Sentence tasks still running
        self._background_tasks: set[asyncio.Task[Any]] = set()  # Cleanup that must not delay audio

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

        if SelectProgram.is_type(event.type):
            return await self._handle_select_program(SelectProgram.from_event(event))

        if Describe.is_type(event.type):
            await self.write_event(self._wyoming_info.event())
            return True

        _LOGGER.info("Ignoring unhandled event type: %s", event.type)
        return True

    async def disconnect(self) -> None:
        """Clean up handler-owned realtime resources when the Wyoming client disconnects."""
        try:
            await self._cancel_synthesis_tasks()
            await self._cleanup_realtime_transcription()
        finally:
            # Closed before the drain so the client is not kept waiting on websocket closing handshakes
            self.writer.close()
            await self._drain_background_tasks()

    async def stop(self) -> None:
        """Stop the event handler without closing shared OpenAI clients."""
        try:
            await self._cancel_synthesis_tasks()
            await self._cleanup_realtime_transcription()
        finally:
            await super().stop()
            await self._drain_background_tasks()

    async def _handle_select_program(self, select_program: SelectProgram) -> bool:
        """Handle select-program request pinning a program for this connection.

        Program names are only unique within a domain, so the name is resolved
        against ASR and TTS programs independently; one event may select both.
        A selection applies only to the domains whose programs match: prior
        selections in other domains remain active for the lifetime of the
        connection (per the Wyoming protocol), so clients can pick a distinct
        program per domain before sending request events. Unrecognized names
        are dropped per the Wyoming protocol.
        """
        name = select_program.name
        matched = False

        for program in self._wyoming_info.asr:
            if program.name == name:
                self._selected_asr_program = program
                matched = True
                _LOGGER.debug("Selected ASR program: %s", name)
                break

        for program in self._wyoming_info.tts:
            if program.name == name:
                self._selected_tts_program = program
                matched = True
                _LOGGER.debug("Selected TTS program: %s", name)
                break

        if not matched:
            _LOGGER.info("Ignoring select-program for unknown program: %s", name)

        return True

    async def _handle_transcribe(self, transcribe: Transcribe) -> bool:
        """Handle transcription request"""
        # No OpenAI-compatible endpoint accepts VAD sensitivity or hotword biasing
        if transcribe.vad_sensitivity is not None:
            _LOGGER.debug("Ignoring unsupported Transcribe field vad_sensitivity: %s", transcribe.vad_sensitivity)
        if transcribe.transcript_names:
            _LOGGER.debug("Ignoring unsupported Transcribe field transcript_names: %s", transcribe.transcript_names)
        if transcribe.transcript_terms:
            _LOGGER.debug("Ignoring unsupported Transcribe field transcript_terms: %s", transcribe.transcript_terms)

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

        # STT_EXTRA_BODY targets /v1/audio/transcriptions, so sessions only take the Realtime extra body
        language, languages, extra_body = self._apply_plural_stt_languages(
            self._current_asr_model.name, self._current_language, self._stt_realtime_extra_body
        )

        transcription: dict[str, object] = {"model": self._current_asr_model.name}
        if language is not None:
            transcription["language"] = language
        if languages is not None:
            transcription["languages"] = languages
        if self._stt_prompt is not None:
            transcription["prompt"] = self._stt_prompt
        transcription.update(extra_body)

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

        return f"Realtime request failed with event type {getattr(event, 'type', 'unknown')}"

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

        # As with Realtime TTS, the closing handshake must not hold up the Wyoming client
        if connection_manager is not None:
            self._run_in_background(connection_manager.__aexit__(None, None, None), name="openai_realtime_stt_close")
        elif connection is not None:
            self._run_in_background(connection.close(), name="openai_realtime_stt_close")

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

            language, languages, extra_body = self._apply_plural_stt_languages(
                self._current_asr_model.name, self._current_language, extra_body
            )

            transcription_kwargs = {
                "file": self._wav_buffer,
                "model": self._current_asr_model.name,
                "language": language if language is not None else omit,
                "languages": languages if languages is not None else omit,
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
        """Get an ASR model by name or None.

        Without a select-program event, nameless requests resolve to the first
        program (per wyoming 1.10 defaults) but explicit names deliberately
        resolve across all programs: every advertised model stays addressable
        by clients that cannot send select-program.
        """
        programs = [self._selected_asr_program] if self._selected_asr_program else self._wyoming_info.asr
        for program in programs:
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
        return any(program.models for program in self._wyoming_info.asr)

    def _has_tts_voices(self) -> bool:
        """Return True when TTS is configured for this handler."""
        return any(program.voices for program in self._wyoming_info.tts)

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
        """Check if an ASR model supports streaming.

        Restricted to the selected program when set, so the streaming decision
        matches the program the model was resolved from.
        """
        programs = [self._selected_asr_program] if self._selected_asr_program else self._wyoming_info.asr
        for program in programs:
            for model in program.models:
                if model.name == model_name:
                    return program.supports_transcript_streaming
        return False

    def _is_asr_model_realtime(self, model_name: str) -> bool:
        """Check if an ASR model should use OpenAI Realtime transcription."""
        return model_name in self._stt_realtime_models

    def _uses_plural_stt_languages(self, model_name: str) -> bool:
        """Check if an ASR model takes a `languages` list instead of the singular `language` field."""
        # Not limited to the official domain: proxies in front of OpenAI need `languages` for these models too
        if self._stt_client is None or self._stt_client.backend != OpenAIBackend.OPENAI:
            return False
        return model_name.startswith(OPENAI_PLURAL_LANGUAGE_STT_MODEL_PREFIXES)

    def _apply_plural_stt_languages(
        self, model_name: str, language: str | None, extra_body: dict[str, object] | None
    ) -> tuple[str | None, list[str] | None, dict[str, object]]:
        """Return the `language` and `languages` fields to send, and a copy of extra_body that does not conflict."""
        extra_body = dict(extra_body or {})
        if not self._uses_plural_stt_languages(model_name):
            # Only the official API is known to reject the field; other servers get what was configured
            is_official = self._stt_client is not None and self._stt_client.is_official_openai
            if is_official and extra_body.pop("languages", None) is not None:
                # A body shared with the models above would otherwise send both fields, which OpenAI rejects
                _LOGGER.debug("Ignoring extra_body `languages` for %s, which takes `language`", model_name)
            return language, None, extra_body

        # These models reject `language` alongside `languages`, so a singular extra_body override is translated
        override = extra_body.pop("language", language)
        if not isinstance(override, str) or not override or "languages" in extra_body:
            # An extra_body `languages` list replaces the request language when the body is merged,
            # and without a usable language the model detects it
            return None, None, extra_body
        return None, [override], extra_body

    def _is_tts_voice_realtime(self, voice: TtsVoiceModel) -> bool:
        """Check if a TTS voice should be synthesized over OpenAI Realtime."""
        return voice.model_name in self._tts_realtime_models

    def _is_tts_text_speakable(self, text: str, voice: TtsVoiceModel) -> bool:
        """Check if a voice's model can be given this text; Realtime models need something to read aloud."""
        # A conversational model handed only punctuation or emoji may improvise a reply instead of staying silent
        return not self._is_tts_voice_realtime(voice) or _has_speakable_content(text)

    def _get_tts_audio_format(self, voice: TtsVoiceModel) -> TtsAudioFormat:
        """Describe the audio bytes `_iter_tts_audio` yields for a voice."""
        if self._is_tts_voice_realtime(voice):
            return TtsAudioFormat(headerless=True, rate=REALTIME_AUDIO_RATE)
        return TtsAudioFormat(headerless=self._get_tts_response_format() != "wav", rate=TTS_AUDIO_RATE)

    def _is_tts_voice_streaming(self, voice_name: str) -> bool:
        """Check if a TTS voice supports streaming synthesis.

        Restricted to the selected program when set, so the streaming decision
        matches the program the voice was resolved from.
        """
        programs = [self._selected_tts_program] if self._selected_tts_program else self._wyoming_info.tts
        for program in programs:
            for voice in program.voices:
                if voice.name == voice_name:
                    return getattr(program, "supports_synthesize_streaming", False)
        return False

    def _iter_tts_voices(self):
        """Iterate over configured TTS voices, limited to the selected program when set.

        Without a selection all programs' voices stay addressable by name; the
        collision-aware public voice names (e.g. "alloy (tts-1)") exist so
        clients without select-program support can reach any advertised voice.
        """
        programs = [self._selected_tts_program] if self._selected_tts_program else self._wyoming_info.tts
        for program in programs:
            for voice in program.voices:
                yield cast(TtsVoiceModel, voice)

    def _get_segmenter_language(self, language: str | None) -> str:
        """
        Get a yasbd-compatible language code.

        Args:
            language (str | None): Language code (e.g., 'en', 'en-US', 'es', etc.)

        Returns:
            str: yasbd-compatible language code, defaults to 'en' if unsupported
        """
        if not language:
            return "en"

        # Extract base language code from potential BCP-47 tags (e.g., 'en-US' -> 'en')
        base_lang = language[:2].lower() if len(language) >= 2 else "en"

        # Test if the language is supported by querying the known language set
        if base_lang not in get_supported_langs():
            _LOGGER.warning(f"Language '{base_lang}' not supported by yasbd, using English")
            return "en"
        return base_lang

    def _chunk_text_for_streaming(
        self, text: str, min_words: int | None = None, max_chars: int | None = None, language: str | None = None
    ) -> list[str]:
        """
        Chunk text into meaningful segments using yasbd sentence segmentation.

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

        # Get yasbd-compatible language code
        sd_language = self._get_segmenter_language(language)
        segmenter = BoundaryDetector(lang=sd_language)
        # preserve_whitespace=False strips leading/trailing whitespace from each sentence

        chunks = []
        current_chunk = ""

        for sentence in segmenter.segment(text):
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
        - Semaphore naturally limits concurrency to the configured number of concurrent TTS requests

        Args:
            sentences (list[str]): Complete sentences ready for synthesis.
            language (str | None): Language code for the sentences.

        Returns:
            bool: True if processing succeeded, False if synthesis was aborted.
        """
        if not sentences or not self._synthesis_voice:
            return True

        try:
            voice = self._resolved_synthesis_voice or self._validate_tts_voice_and_language(
                self._synthesis_voice.name, self._synthesis_voice.language
            )
            if not voice:
                _LOGGER.error("Failed to validate voice for incremental synthesis")
                return await self._abort_synthesis()
            self._resolved_synthesis_voice = voice

            use_streaming = self._is_tts_voice_streaming(voice.name)

            if use_streaming:
                valid_sentences = [s for s in sentences if s.strip() and self._is_tts_text_speakable(s, voice)]
                if not valid_sentences:
                    _LOGGER.debug("No non-empty sentences available for incremental synthesis.")
                    return True

                _LOGGER.info("Starting concurrent synthesis for %d sentences", len(valid_sentences))

                # Create ALL tasks with IDs - API calls start concurrently
                # Semaphore limits actual concurrency to the configured number of concurrent TTS requests
                synthesis_tasks = [
                    (
                        f"sentence_{i}",
                        self._create_synthesis_task(
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

                    if result.skipped:
                        continue

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
                        audio_format=self._get_tts_audio_format(voice),
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

    def _track_task(
        self, coro: Any, *, name: str, tasks: set[asyncio.Task[Any]], log_failure: bool
    ) -> asyncio.Task[Any]:
        """Start a task that stays in `tasks` until it finishes; its exception is retrieved either way."""

        def on_done(task: asyncio.Task[Any]) -> None:
            tasks.discard(task)
            if not task.cancelled() and (err := task.exception()) is not None and log_failure:
                _LOGGER.debug("Background task %s failed: %s", task.get_name(), err)

        task = asyncio.create_task(coro, name=name)
        tasks.add(task)
        task.add_done_callback(on_done)
        return task

    def _create_synthesis_task(self, coro: Any, *, name: str) -> asyncio.Task[TtsStreamResult]:
        """Start a sentence synthesis task that is cancelled if the session aborts or the client disconnects."""
        # Its failure is reported by whoever awaits it
        return self._track_task(coro, name=name, tasks=self._synthesis_tasks, log_failure=False)

    async def _cancel_synthesis_tasks(self) -> None:
        """Cancel sentence tasks that are still running so they stop holding backend connections."""
        tasks = [task for task in self._synthesis_tasks if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _run_in_background(self, coro: Any, *, name: str) -> None:
        """Run cleanup that should not delay the audio stream; awaited when the client disconnects."""
        self._track_task(coro, name=name, tasks=self._background_tasks, log_failure=True)

    async def _drain_background_tasks(self) -> None:
        """Wait for pending background cleanup."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

    async def _abort_synthesis(self) -> bool:
        """Abort the current synthesis session, emitting stop events and resetting state."""
        await self._cancel_synthesis_tasks()

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
        self._segmenters.clear()
        self._synthesis_voice = None
        self._resolved_synthesis_voice = None
        self._synthesis_text_format = None
        self._ssml_transformer = None

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

    def _get_default_voice(self) -> TtsVoiceModel | None:
        """Get the voice for a request that names none: the first one, preferring the speech API over Realtime."""
        voices = list(self._iter_tts_voices())
        return self._prefer_speech_api_voices(voices)[0] if voices else None

    def _get_voices_by_backend_name(self, backend_voice_name: str) -> list[TtsVoiceModel]:
        """Get TTS voices by the raw backend voice identifier."""
        return [
            voice
            for voice in self._iter_tts_voices()
            if getattr(voice, "backend_voice_name", voice.name) == backend_voice_name
        ]

    def _get_voice_by_suffixed_name(self, requested_voice: str) -> TtsVoiceModel | None:
        """Resolve a "voice (model)" name that stopped being advertised when the set of TTS models changed."""
        parsed = parse_tts_voice_name(requested_voice)
        if not parsed:
            return None

        voice_name, model_name = parsed
        matches = self._get_voices_by_backend_name(voice_name)
        if not matches:
            return None

        # A model that is gone falls back to the speech API where possible rather than onto a Realtime model
        model_matches = [voice for voice in matches if voice.model_name == model_name]
        voice = (model_matches or self._prefer_speech_api_voices(matches))[0]
        _LOGGER.warning(
            "Voice %s is no longer advertised under that name. Falling back to %s for backward compatibility. "
            "Update the client to use one of: %s",
            requested_voice,
            voice.name,
            [voice.name for voice in matches],
        )
        return voice

    def _prefer_speech_api_voices(self, voices: list[TtsVoiceModel]) -> list[TtsVoiceModel]:
        """Narrow voices a request did not pin down to the speech API ones, so it never moves onto Realtime unasked."""
        return [voice for voice in voices if not self._is_tts_voice_realtime(voice)] or voices

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
        voice = self._get_voice(requested_voice) if requested_voice else self._get_default_voice()
        if voice:
            if not self._validate_tts_language(requested_language, voice):
                return None
            return voice

        if not requested_voice:
            self._log_unsupported_voice(requested_voice)
            return None

        backend_matches = self._get_voices_by_backend_name(requested_voice)
        if not backend_matches:
            voice = self._get_voice_by_suffixed_name(requested_voice)
            if not voice:
                self._log_unsupported_voice(requested_voice)
                return None
            if not self._validate_tts_language(requested_language, voice):
                return None
            return voice

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

        voice = self._prefer_speech_api_voices(compatible_matches)[0]
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

    @staticmethod
    def _is_ssml_format(text_format: str | SynthesizeTextFormat | None) -> bool:
        """Check for SSML text format; str-enum equality also matches a raw "ssml" string."""
        return text_format == SynthesizeTextFormat.SSML

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

            text = synthesize.text
            if self._is_ssml_format(synthesize.text_format):
                text = strip_ssml(text)
                _LOGGER.debug("Stripped SSML markup from synthesize text (text_format=ssml)")

            # Use shared streaming logic
            final_timestamp = await self._stream_tts_audio(voice, text, send_audio_start=True)

            if final_timestamp is not None:
                # Send audio stop after streaming completes
                await self.write_event(AudioStop(timestamp=int(final_timestamp)).event())
                _LOGGER.info("Successfully synthesized: %s", _truncate_for_log(text))
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
        self._synthesis_text_format = synthesize_start.text_format
        self._ssml_transformer = SsmlTextTransformer() if self._is_ssml_format(synthesize_start.text_format) else None

        # Reset incremental detection state
        self._text_accumulator = ""
        self._ready_chunks = []
        self._segmenters.clear()  # Clear segmenter cache for new session
        self._audio_started = False  # Reset audio started flag
        self._current_timestamp = 0  # Reset timestamp for new synthesis session
        self._resolved_synthesis_voice = None

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
            # Sentences reuse this instead of resolving, and warning about, the name again
            self._resolved_synthesis_voice = voice
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

        if self._ssml_transformer is not None:
            chunk_text = self._ssml_transformer.feed(chunk_text)

        # Keep the fallback buffer and sentence accumulator as one identical
        # projected stream. The transformer owns all SSML whitespace repair.
        self._synthesis_buffer.append(chunk_text)

        # Add to accumulator for sentence detection across chunks
        self._text_accumulator += chunk_text

        # Get or create segmenter for the current language
        requested_language = self._synthesis_voice.language if self._synthesis_voice else None
        sd_language = self._get_segmenter_language(requested_language)

        # Use cached segmenter or create a new one
        if sd_language not in self._segmenters:
            _LOGGER.debug("Creating new yasbd segmenter for language: %s", sd_language)
            self._segmenters[sd_language] = BoundaryDetector(lang=sd_language)

        segmenter = self._segmenters[sd_language]

        # Segment the entire accumulated text. preserve_whitespace=True keeps every
        # character (the retained last segment is appended to by later events), so no
        # text is lost at sentence boundaries even when yasbd redistributes the space.
        sentences: list[str] = list(segmenter.segment(self._text_accumulator, preserve_whitespace=True))

        # Process complete sentences (all but the last one)
        if len(sentences) > 1:
            # Keep only the last sentence in the accumulator
            self._text_accumulator = sentences.pop()
            ready_sentences = sentences

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

        if self._ssml_transformer is not None:
            flushed = self._ssml_transformer.finish()
            self._text_accumulator += flushed
            self._synthesis_buffer.append(flushed)

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
        resolved_voice = self._resolved_synthesis_voice

        _LOGGER.debug("Streaming synthesis completed with text: %s", _truncate_for_log(full_text))

        # Clear synthesis state early
        self._synthesis_buffer = []
        self._synthesis_voice = None
        self._resolved_synthesis_voice = None
        self._synthesis_text_format = None
        self._ssml_transformer = None
        self._text_accumulator = ""
        self._ready_chunks = []
        self._segmenters.clear()  # Clear segmenter cache

        # Send audio stop if we processed any audio incrementally
        if self._audio_started:
            await self.write_event(AudioStop(timestamp=int(self._current_timestamp)).event())
            await self.write_event(SynthesizeStopped().event())
            _LOGGER.info(
                "Successfully completed incremental streaming synthesis, final timestamp: %.2f", self._current_timestamp
            )
            self._audio_started = False  # Reset for next session
            self._current_timestamp = 0  # Reset for next session
            self._segmenters.clear()  # Clear segmenter cache
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
            voice = resolved_voice or self._validate_tts_voice_and_language(requested_voice, requested_language)
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
                chunks = [chunk for chunk in chunks if self._is_tts_text_speakable(chunk, voice)]
                if not chunks:
                    # As with empty text above, a streaming synthesis ends on synthesize-stopped without audio events
                    _LOGGER.warning("No speakable text to synthesize")
                    await self.write_event(SynthesizeStopped().event())
                    return True
                _LOGGER.debug("Text chunked into %d parts for streaming synthesis", len(chunks))

                # Create ALL tasks with IDs - API calls start concurrently
                # Semaphore limits actual concurrency to the configured number of concurrent TTS requests
                _LOGGER.info("Starting concurrent synthesis for %d chunks", len(chunks))
                synthesis_tasks = [
                    (
                        f"fallback_chunk_{i}",
                        self._create_synthesis_task(
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

                    if result.skipped:
                        continue

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
                        audio_format=self._get_tts_audio_format(voice),
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
            async with self._tts_semaphore, contextlib.aclosing(self._iter_tts_audio(text, voice)) as audio_stream:
                async for chunk in audio_stream:
                    chunks.append(chunk)

            audio_data = b"".join(chunks)
            if not audio_data and not _has_speakable_content(text):
                # As on the direct path, nothing to say is not a failure; there is nothing left to play
                _LOGGER.debug("TTS backend returned no audio for unspeakable text: %s", chunk_preview)
                return TtsStreamResult(streamed=False, skipped=True)
            if not audio_data:
                raise TtsStreamError("OpenAI returned empty audio response", chunk_preview, voice.name)

            _LOGGER.debug("Completed buffered synthesis for chunk: %s", chunk_preview)
            return TtsStreamResult(streamed=False, audio=audio_data)

        except TtsStreamError:
            raise
        except Exception as exc:
            _LOGGER.exception("Error getting TTS audio stream for %s: %s", chunk_preview, exc)
            raise TtsStreamError("Unexpected error while retrieving TTS audio", chunk_preview, voice.name) from exc

    async def _iter_tts_audio(self, text: str, voice: TtsVoiceModel) -> AsyncGenerator[bytes, None]:
        """Yield synthesized audio bytes for a text chunk from the transport the voice's model uses."""
        assert self._tts_client is not None

        if self._is_tts_voice_realtime(voice):
            # Closed explicitly so the websocket does not outlive a consumer that stops early
            async with contextlib.aclosing(self._iter_realtime_tts_audio(text, voice)) as realtime_stream:
                async for chunk in realtime_stream:
                    yield chunk
            return

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
                yield chunk

    def _get_realtime_tts_session(self, voice: TtsVoiceModel) -> dict[str, object]:
        """Build a Realtime session update payload that speaks text as PCM audio."""
        extra_body = dict(self._tts_realtime_extra_body or {})
        output_override = get_realtime_tts_audio_output(extra_body)
        output_override.pop("speed", None)
        audio_override = extra_body.pop("audio", None)
        audio_override = dict(audio_override) if isinstance(audio_override, dict) else {}
        audio_override.pop("output", None)

        instructions = REALTIME_TTS_INSTRUCTIONS
        if self._tts_instructions:
            instructions = f"{instructions}\n\nDelivery style: {self._tts_instructions}"

        audio_output: dict[str, object] = {
            **output_override,
            # The Wyoming client picks the voice, and is told 24 kHz PCM16, so neither is overridable
            "voice": self._get_backend_voice_name(voice),
            "format": dict(REALTIME_TTS_AUDIO_FORMAT),
        }
        speed = get_realtime_tts_speed(self._tts_speed, self._tts_realtime_extra_body)
        if speed is not None:
            # An out-of-range speed is reported once at startup
            audio_output["speed"] = clamp_realtime_tts_speed(speed)

        return {
            **extra_body,
            "type": "realtime",
            "output_modalities": ["audio"],
            "tool_choice": "none",
            "instructions": instructions,
            "audio": {**audio_override, "output": audio_output},
        }

    async def _iter_realtime_tts_audio(self, text: str, voice: TtsVoiceModel) -> AsyncGenerator[bytes, None]:
        """Yield PCM audio for a text chunk from a single out-of-band Realtime response."""
        assert self._tts_client is not None
        if not _has_speakable_content(text):
            # Callers skip such text; a conversational model must never be handed it
            raise RealtimeSynthesisError("Text has nothing to read aloud")
        connection_manager = self._tts_client.realtime.connect(model=voice.model_name)
        connection = await self._enter_realtime_connection(connection_manager)
        try:
            await connection.session.update(session=self._get_realtime_tts_session(voice))
            await connection.response.create(
                response={
                    "conversation": "none",
                    "output_modalities": ["audio"],
                    "input": [
                        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}
                    ],
                }
            )

            received_audio = False
            pending_audio = b""  # Trailing part of a sample that a delta split
            spoken_parts: list[str] = []  # Transcripts of finished content parts
            spoken_pending = ""  # Transcript deltas of the part still being spoken
            # Closed explicitly so the SDK's event iterator is not left to garbage collection
            async with contextlib.aclosing(aiter(connection)) as events:
                while True:
                    try:
                        event = await asyncio.wait_for(anext(events), REALTIME_TTS_EVENT_TIMEOUT)
                    except StopAsyncIteration:
                        break
                    except TimeoutError as err:
                        raise RealtimeSynthesisError(
                            f"No Realtime event received for {REALTIME_TTS_EVENT_TIMEOUT:g} seconds during synthesis"
                        ) from err

                    event_type = getattr(event, "type", "")
                    if event_type == "response.output_audio.delta":
                        if delta := getattr(event, "delta", ""):
                            received_audio = True
                            # Wyoming chunks must hold whole samples, and deltas are not guaranteed to
                            audio = pending_audio + base64.b64decode(delta)
                            aligned_size = len(audio) - len(audio) % REALTIME_AUDIO_WIDTH
                            pending_audio = audio[aligned_size:]
                            if aligned_size:
                                yield audio[:aligned_size]
                    elif event_type == "response.output_audio_transcript.delta":
                        spoken_pending += getattr(event, "delta", "") or ""
                    elif event_type == "response.output_audio_transcript.done":
                        spoken_parts.append(getattr(event, "transcript", "") or spoken_pending)
                        spoken_pending = ""
                    elif event_type == "response.done":
                        response = getattr(event, "response", None)
                        status = getattr(response, "status", None)
                        if status not in (None, "completed"):
                            raise RealtimeSynthesisError(
                                f"Realtime response ended with status {status}"
                                f"{self._describe_realtime_status_details(response)}"
                            )
                        if not received_audio:
                            raise RealtimeSynthesisError("Realtime response completed without any audio")
                        self._check_realtime_tts_fidelity(text, " ".join([*spoken_parts, spoken_pending]))
                        return
                    elif event_type == "error":
                        raise RealtimeSynthesisError(self._get_realtime_event_error_message(event))

            raise RealtimeSynthesisError("Realtime connection closed before synthesis completed")
        finally:
            # The closing handshake must not hold up the end of the audio stream or an abort
            self._run_in_background(connection_manager.__aexit__(None, None, None), name="openai_realtime_tts_close")

    def _describe_realtime_status_details(self, response: Any) -> str:
        """Format why a Realtime response did not complete, for error messages."""
        details = getattr(response, "status_details", None)
        error = getattr(details, "error", None)
        candidates = (getattr(details, "reason", None), getattr(error, "code", None), getattr(error, "type", None))
        parts = [value for value in candidates if isinstance(value, str) and value]
        return f" ({', '.join(parts)})" if parts else ""

    def _check_realtime_tts_fidelity(self, text: str, spoken_text: str) -> None:
        """Warn when a Realtime model spoke something other than the requested text."""
        requested = _normalize_for_comparison(text)
        # Symbol-only text ("+") has nothing to compare against its spoken name
        if requested and spoken_text.strip() and _normalize_for_comparison(spoken_text) != requested:
            _LOGGER.warning(
                "Realtime TTS spoke text that differs from the request. Requested: %s | Spoken: %s",
                _truncate_for_log(text),
                _truncate_for_log(spoken_text),
            )

    async def _stream_audio_to_wyoming(
        self,
        audio_data: bytes,
        is_first_chunk: bool,
        start_timestamp: float,
        audio_format: TtsAudioFormat,
    ) -> float | None:
        """
        Stream audio data to Wyoming with proper timestamp calculation.

        Args:
            audio_data (bytes): Complete audio data to stream.
            is_first_chunk (bool): Whether this is the first chunk (sends AudioStart).
            start_timestamp (float): Starting timestamp for this chunk.
            audio_format (TtsAudioFormat): How the audio is framed, from `_get_tts_audio_format`.

        Returns:
            float | None: Final timestamp after streaming, or None on error.
        """
        try:
            timestamp = start_timestamp

            # The same framing as the direct path, fed the whole response at once
            framer = _WavFramer(audio_format, self._parse_wav_header)
            audio_data = framer.feed(audio_data)
            header_missing = framer.awaiting_header
            audio_data += framer.finish()
            if header_missing:
                _LOGGER.debug("Could not parse WAV header, using defaults: %d Hz", TTS_AUDIO_RATE)
            elif framer.missing_bytes and not audio_data:
                # As on the direct path, a WAV that declares samples and delivers none is truncated
                _LOGGER.error("TTS WAV response declared %d PCM bytes and delivered none", framer.declared_size)
                return None
            elif framer.missing_bytes:
                _LOGGER.warning(
                    "TTS WAV response ended after %d of %d declared PCM bytes", len(audio_data), framer.declared_size
                )
            audio_rate, audio_width, audio_channels = framer.rate, framer.width, framer.channels

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
        audio_start_requested = send_audio_start
        timestamp = start_timestamp
        try:
            framer = _WavFramer(self._get_tts_audio_format(voice), self._parse_wav_header)
            wrote_audio = False

            async def write_audio(audio_data: bytes) -> None:
                """Send the audio start once the format is known, then the PCM."""
                nonlocal send_audio_start, timestamp, wrote_audio
                if send_audio_start:
                    await self.write_event(
                        AudioStart(rate=framer.rate, width=framer.width, channels=framer.channels).event()
                    )
                    send_audio_start = False
                if not audio_data:
                    return
                await self.write_event(
                    AudioChunk(
                        audio=audio_data,
                        rate=framer.rate,
                        width=framer.width,
                        channels=framer.channels,
                        timestamp=int(timestamp),
                    ).event()
                )
                timestamp = self._advance_audio_timestamp(
                    timestamp,
                    audio_data=audio_data,
                    audio_rate=framer.rate,
                    audio_width=framer.width,
                    audio_channels=framer.channels,
                )
                wrote_audio = True

            if self._tts_client is None:
                _LOGGER.error("No TTS client configured for synthesis")
                return None

            if not self._is_tts_text_speakable(text, voice):
                _LOGGER.debug("Skipping text with nothing to speak: %s", _truncate_for_log(text, 50))
                await write_audio(b"")
                return timestamp

            async with self._tts_semaphore:
                async with contextlib.aclosing(self._iter_tts_audio(text, voice)) as audio_stream:
                    async for chunk in audio_stream:
                        audio_data = framer.feed(chunk)
                        if not framer.awaiting_header:
                            await write_audio(audio_data)

            if framer.missing_bytes:
                _LOGGER.warning("TTS WAV response ended with %d declared PCM bytes missing", framer.missing_bytes)

            header_missing = framer.awaiting_header
            if held_audio := framer.finish():
                if header_missing:
                    _LOGGER.warning(
                        "TTS response ended before a complete WAV header was available, falling back to raw PCM"
                    )
                await write_audio(held_audio)

            if not wrote_audio and not framer.declared_empty and not _has_speakable_content(text):
                # Headerless audio cannot declare itself empty, but punctuation or emoji leave nothing to say
                _LOGGER.debug("TTS backend returned no audio for unspeakable text: %s", _truncate_for_log(text, 50))
                await write_audio(b"")
                return timestamp

            if not wrote_audio and not framer.declared_empty:
                # Reporting success here would let callers close or continue an audio stream that never started.
                # A WAV declaring zero samples, or an unknown length, is not a failure: the backend had nothing
                # to say. One that declares samples and delivers none is truncated
                _LOGGER.error("TTS backend returned no audio for: %s", _truncate_for_log(text, 50))
                await self._stop_failed_tts_stream(audio_start_requested and not send_audio_start, timestamp)
                return None

            return timestamp

        except Exception as e:
            _LOGGER.exception("Error streaming TTS audio: %s", e)
            await self._stop_failed_tts_stream(audio_start_requested and not send_audio_start, timestamp)
            return None

    async def _stop_failed_tts_stream(self, opened_stream: bool, timestamp: float) -> None:
        """Close an audio stream a failed synthesis call opened; callers only close streams that completed."""
        if opened_stream:
            with contextlib.suppress(Exception):
                await self.write_event(AudioStop(timestamp=int(timestamp)).event())

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

    def _parse_wav_header(self, wav_data: bytes) -> tuple[int, int, int, int, int | None] | None:
        """
        Parse WAV header to extract the PCM format, data offset, and data size.
        Returns (sample_rate, channels, sample_width, data_offset, data_size) or None if parsing fails.
        The data size is None when the WAV data chunk uses the unbounded-size sentinel.
        """
        try:
            # Create a BytesIO object from the data
            wav_io = io.BytesIO(wav_data)

            # Open with wave module
            with wave.open(wav_io, "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()

                # Get the current position which should be at the start of audio data
                data_offset = wav_io.tell()
                declared_data_size = struct.unpack("<I", wav_data[data_offset - 4 : data_offset])[0]
                data_size = (
                    None
                    if declared_data_size == WAV_UNBOUNDED_SIZE
                    else wav_file.getnframes() * channels * sample_width
                )

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
