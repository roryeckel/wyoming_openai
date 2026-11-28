import asyncio
import base64
import contextlib
import functools
import logging
import unicodedata
import wave
import weakref
from collections.abc import AsyncGenerator, Coroutine
from dataclasses import dataclass, field
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
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Describe, Info, SelectProgram, TtsProgram, TtsVoice
from wyoming.ping import Ping, Pong
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
    SynthesizeTextFormat,
)
from yasbd import BoundaryDetector, get_supported_langs

from .compatibility import CustomAsyncOpenAI, OpenAIBackend, TtsVoiceModel, parse_tts_voice_name
from .concurrency import ConcurrencyLimit, ConcurrencySlot
from .const import (
    DEFAULT_AUDIO_CHANNELS,
    DEFAULT_AUDIO_WIDTH,
    DEFAULT_STT_PROMPT_MAX_CHARS,
    OPENAI_PLURAL_LANGUAGE_STT_MODEL_PREFIXES,
    OPENAI_PROMPTLESS_REALTIME_STT_MODEL_PREFIXES,
    REALTIME_AUDIO_FORMAT,
    REALTIME_AUDIO_RATE,
    REALTIME_AUDIO_WIDTH,
    REALTIME_STT_TRANSCRIPT_TIMEOUT,
    REALTIME_TTS_EVENT_TIMEOUT,
    REALTIME_TTS_INSTRUCTIONS,
)
from .home_assistant import HomeAssistantNames
from .utilities import (
    NamedBytesIO,
    NameGroup,
    RealtimeAudioConverter,
    SsmlTextTransformer,
    advertised_language,
    build_prompt,
    clamp_realtime_tts_speed,
    get_extra_body_boolean_field,
    get_realtime_tts_audio_output,
    get_realtime_tts_speed,
    primary_language_subtag,
    strip_ssml,
    supports_language,
    truncate_for_log,
    validate_audio_format,
    validate_realtime_stt_extra_body,
    validate_realtime_tts_extra_body,
    validate_stt_extra_body,
    validate_tts_extra_body,
)
from .wav import TtsAudioFormat, WavFramer

_LOGGER = logging.getLogger(__name__)


def _normalize_for_comparison(text: str) -> str:
    """
    Reduce text to case-folded alphanumerics for comparing requested and spoken text.
    Numbers and symbols a transcript spells out ("50%" as "fifty percent") still compare as different.
    """
    # NFKC so composed and decomposed forms of the same letter compare equal
    return "".join(char for char in unicodedata.normalize("NFKC", text).casefold() if char.isalnum())


# Symbols that are read aloud on their own ("plus", "and", "percent"), unlike punctuation or emoji
_SPEAKABLE_SYMBOLS = frozenset("+=±×÷&@%#")


def _is_spoken_symbol(char: str) -> bool:
    """Check for a symbol that is read aloud by name."""
    return char in _SPEAKABLE_SYMBOLS or unicodedata.category(char) == "Sc"  # Sc: currency


def _has_speakable_content(text: str) -> bool:
    """Check if text has anything to read aloud: a letter, a digit, or a symbol with a spoken name."""
    return any(char.isalnum() or _is_spoken_symbol(char) for char in text)


@functools.lru_cache(maxsize=64)
def _get_segmenter_language(base_language: str) -> str:
    """Get the yasbd language for a primary language subtag, English if unsupported; cached to report that once."""
    if base_language in get_supported_langs():
        return base_language
    _LOGGER.warning("Language '%s' not supported by yasbd, using English", base_language)
    return "en"


@functools.cache
def _build_boundary_detector(segmenter_language: str) -> BoundaryDetector:
    """Build the sentence segmenter of a yasbd language, once; it is shared by every synthesis."""
    _LOGGER.debug("Creating new yasbd segmenter for language: %s", segmenter_language)
    return BoundaryDetector(lang=segmenter_language)


def _get_boundary_detector(language: str | None) -> BoundaryDetector:
    """Get the sentence segmenter for a language tag (e.g. 'en', 'en-US', 'es'), English without one."""
    return _build_boundary_detector(_get_segmenter_language(primary_language_subtag(language or "en")))


def _retrieve_result(task: asyncio.Future[Any]) -> None:
    """Read a finished task's exception, so one nobody awaits any more is not logged as never retrieved."""
    if not task.cancelled():
        task.exception()


def _final_transcript(final_text: str | None, snapshot: str) -> str:
    """
    The final transcript of a stream whose deltas built `snapshot`. Any non-empty final text replaces them, as the
    completed ASR may revise what it streamed (OpenAI's `transcript.text.done` carries the complete transcription,
    and wyoming 1.11 chunks are revisable); a backend sending partial final texts is outside the supported contract.
    An empty one is not a revision: it cannot be told apart from a backend that only streams, as Speaches 0.9 answers
    a streamed transcription with text="" and the deltas alone, so the snapshots already sent stand.
    """
    return final_text or snapshot


DEFAULT_ASR_AUDIO_RATE = 16000  # Hz (Wyoming default)
TTS_AUDIO_RATE = 24000  # Hz (OpenAI spec, fallback)
TTS_CHUNK_SIZE = 2048  # Bytes read from the TTS response at a time; WavFramer buffers a header across reads
TTS_CONCURRENT_REQUESTS = 3  # Default number of TTS requests in flight across all connections; 0 means no limit
STT_CONCURRENT_REQUESTS = 3  # Default number of STT requests in flight across all connections; 0 means no limit
# Seconds a client may take to accept one event written while its request holds a shared slot. The limit is per event,
# so a client that reads audio at playback speed is not cut off, but one that stops reading fails the request rather
# than keep the slot from other connections
CLIENT_WRITE_TIMEOUT = 10.0
# Seconds a Realtime STT session may go without audio before it fails and gives its slot back: a client streams audio
# continuously, so a gap this long means it is stuck or gone
REALTIME_STT_IDLE_TIMEOUT = 10.0
TTS_STREAM_STOP_TIMEOUT = 1.0  # Seconds to wait for a client to take the events of a failed or cancelled synthesis
SHUTDOWN_DRAIN_TIMEOUT = 2.0  # Seconds stop() waits for Realtime websockets to finish closing before giving up on them
# Seconds a closed client connection may take to send what is left in it before it is aborted, at disconnect and
# at stop()
SHUTDOWN_CLOSE_TIMEOUT = 2.0
# Sent as a Wyoming error event; the cause is in the log
TTS_FAILED_ERROR = "Text-to-speech synthesis failed"
STT_FAILED_ERROR = "Speech-to-text transcription failed"
SHUTDOWN_ERROR = "Server is shutting down"

@dataclass(frozen=True)
class TtsStreamResult:
    """Container for TTS streaming outcomes."""

    streamed: bool
    audio: bytes | None = None


@dataclass
class _SynthesisState:
    """What one synthesis keeps between events. Replaced as a whole, so nothing carries over to the next one."""

    active: bool = False  # Between synthesize-start and synthesize-stop
    plain: bool = False  # A synthesize request on its own, which only an audio stop ends successfully
    buffer: list[str] = field(default_factory=list)  # All text of the session, for a voice that is not streamed
    text_accumulator: str = ""  # Text not handed to the backend yet
    voice: TtsVoiceModel | None = None  # Resolved once per streaming synthesis
    streamed: bool = False  # Its sentences are spoken as they arrive; decided once, with the voice
    language: str | None = None  # Requested along with that voice
    ssml_transformer: SsmlTextTransformer | None = None
    audio_started: bool = False  # AudioStart was sent and AudioStop was not
    timestamp: float = 0  # End of the audio sent so far, in milliseconds
    skipped_speakable_sentence: bool = False  # A sentence with something to say came back without audio
    failed: bool = False  # Its failure was reported to the client, or the client is gone; not reported twice


@dataclass
class _AudioStreamState:
    """
    What one audio stream is transcribed with: chosen at audio-start, dropped at audio-stop.

    The transport stays fixed. A transcribe during a /v1/audio/transcriptions recording may still change its model,
    language and streaming, since the audio is only sent at audio-stop.
    """

    model: AsrModel | None  # None when no model can be used, and then nothing is sent to the backend
    language: str | None
    realtime: bool  # Transcribed over a Realtime session rather than /v1/audio/transcriptions
    streaming: bool  # /v1/audio/transcriptions responds with a stream of transcript deltas; follows the model
    transcript_started: bool = False  # TranscriptStart was sent while the audio was still arriving
    failed: bool = False  # Reported with an error event; its audio is dropped and its audio-stop closes the connection
    converter: RealtimeAudioConverter | None = None  # Set for a Realtime stream once its format is validated
    # The transcript_names, then transcript_terms, of the transcribe it was selected by, for the STT prompt
    bias_names: tuple[NameGroup, ...] = ()

    @property
    def realtime_converter(self) -> RealtimeAudioConverter:
        """The converter of a Realtime stream, or an error if its session was set up incorrectly."""
        if self.converter is None:
            raise RealtimeTranscriptionError("Realtime audio converter was not initialized")
        return self.converter


@dataclass
class _DetachedRealtimeSession:
    """What a Realtime STT session owned once it is detached from the handler, for closing it."""

    receive_task: asyncio.Task[None] | None
    transcript_future: asyncio.Future[str] | None
    connection_manager: Any | None
    connection: Any | None


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


class SlowClientError(Exception):
    """
    Raised when the Wyoming client does not take an event of a request that may hold a shared slot within
    CLIENT_WRITE_TIMEOUT. Not a TimeoutError, so a backend time limit around the write, such as the wait for a
    Realtime transcript, does not report it as its own.
    """


class OpenAIEventHandler(AsyncEventHandler):
    def __init__(
        self,
        *args,
        info: Info,
        stt_client: CustomAsyncOpenAI | None,
        tts_client: CustomAsyncOpenAI | None,
        stt_temperature: float | None = None,
        stt_prompt: str | None = None,
        stt_hass_names: HomeAssistantNames | None = None,
        stt_prompt_max_chars: int = DEFAULT_STT_PROMPT_MAX_CHARS,
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
        stt_limit: ConcurrencyLimit | None = None,
        tts_limit: ConcurrencyLimit | None = None,
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
            stt_hass_names (HomeAssistantNames | None): Names read from Home Assistant, added to the STT prompt
                after stt_prompt and the names sent with transcribe. They are shared by every connection and owned by
                the caller.
            stt_prompt_max_chars (int): The length the names may bring the STT prompt to; stt_prompt is never
                shortened.
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
            stt_limit (ConcurrencyLimit | None): Shared by every connection of the server to limit the STT requests
                in flight across all of them, or None for no limit. A /v1/audio/transcriptions request holds a slot
                from audio-stop until its response is read; a Realtime session from audio-start until it ends.
            tts_limit (ConcurrencyLimit | None): Shared by every connection of the server to limit the TTS requests
                in flight across all of them, or None for no limit. Each sentence request of a synthesis holds a
                slot while its response is read, as does an unsplit request.
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
        self._hass_names = stt_hass_names
        self._stt_prompt_max_chars = stt_prompt_max_chars
        self._stt_extra_body = dict(stt_extra_body) if stt_extra_body else None
        self._stt_realtime_extra_body = dict(stt_realtime_extra_body) if stt_realtime_extra_body else None
        self._stt_realtime_models = set(stt_realtime_models or self._get_asr_program_model_names("openai-realtime"))
        self._stt_limit = stt_limit if stt_limit is not None else ConcurrencyLimit(0)
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
        # Selected by the last transcribe, for the audio streams that start after it
        self._current_asr_model: AsrModel | None = None
        self._current_language: str | None = None
        self._current_bias_names: tuple[NameGroup, ...] = ()
        # The last transcribe was refused. A refusal closes the connection, but a caller that keeps sending
        # events must not have the default model transcribe what the client asked for and was refused
        self._transcribe_rejected = False
        self._audio_stream: _AudioStreamState | None = None  # Between audio-start and audio-stop
        self._shutdown_reported = False  # stop() told the client its transcription was cut short
        self._stopping = False  # stop() was called; STT events are ignored from then on
        self._stopped = asyncio.Event()  # The first stop() has finished, which a repeated one waits for
        self._stt_request: asyncio.Task[Any] | None = None  # The backend part of an STT request, which stop() cancels
        self._stt_answer_owed = False  # An audio-stop has not queued its transcript or error yet

        # State for realtime transcription
        self._realtime_stt_slot: ConcurrencySlot | None = None  # Held from audio-start until the session ends
        self._realtime_idle_timer: asyncio.TimerHandle | None = None  # Fails a session left without audio
        self._realtime_connection_manager: Any | None = None
        self._realtime_connection: Any | None = None
        self._realtime_receive_task: asyncio.Task[None] | None = None
        self._realtime_transcript_future: asyncio.Future[str] | None = None

        # State for event logging
        self._last_event_type: str | None = None
        self._event_counter: int = 0

        # State of the current synthesis
        self._synthesis = _SynthesisState()

        self._tts_limit = tts_limit if tts_limit is not None else ConcurrencyLimit(0)
        self._allow_streaming_task_id: str | None = None  # ID of task allowed to stream directly
        self._synthesis_tasks: set[asyncio.Task[Any]] = set()  # Sentence and unsplit synthesis tasks still running
        self._background_tasks: set[asyncio.Task[Any]] = set()  # Cleanup that must not delay audio
        # Cleanup a drain has cancelled, kept here because a task can clear its own cancellation count
        self._cancelled_cleanup: weakref.WeakSet[asyncio.Task[Any]] = weakref.WeakSet()

    async def handle_event(self, event: Event) -> bool:
        """
        Handle incoming events
        https://github.com/OHF-Voice/wyoming?tab=readme-ov-file#event-types
        """
        if self._stopping and any(
            event_type.is_type(event.type) for event_type in (AudioChunk, AudioStart, AudioStop, Transcribe)
        ):
            # stop() answers the audio stream in progress, and no STT request starts after it
            return True

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
            return await self._handle_audio_stop()

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

        if Ping.is_type(event.type):
            # A liveness check, answered with its text; it does not touch a transcription or synthesis in progress
            await self.write_event(Pong(text=Ping.from_event(event).text).event())
            return True

        _LOGGER.info("Ignoring unhandled event type: %s", event.type)
        return True

    async def disconnect(self) -> None:
        """Drop the audio stream in progress and other handler-owned resources when the Wyoming client disconnects."""
        if self._stopping:
            # stop() answers the client and closes the connection. The run loop can end first, as at once for a client
            # that has stopped sending, so the stream stop() reports on is not dropped before it has
            try:
                await self._stopped.wait()
            except asyncio.CancelledError:
                # Cancelled meanwhile, as at Ctrl+C: the connection is still stop()'s to answer and close, so the
                # writer stays open for it, but neither the recording nor a Realtime session is left open, as they
                # would be if stop() was cancelled too
                self._discard_recording()
                try:
                    await self._cleanup_realtime_transcription()
                finally:
                    # The websocket close that schedules is waited for here, as stop() may have drained its own
                    # already, also when a second cancellation cuts the wait for the receiver short
                    await self._drain_background_tasks(SHUTDOWN_DRAIN_TIMEOUT)
                raise
        try:
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                # Cancelled rather than disconnected, as when Ctrl+C stops a server that has no SIGTERM handler to call
                # stop(): the client may still be reading, and takes a streaming synthesis without an error as complete
                await self._stop_sentence_playback(SHUTDOWN_ERROR if self._is_synthesis_in_flight() else None)
            else:
                # Nobody is left to report a failure to
                self._synthesis.failed = True
                await self._stop_sentence_playback()
            await self._drop_audio_stream()
        finally:
            # Closed before the drain so the client is not kept waiting on websocket closing handshakes
            self.writer.close()
            try:
                # A client that stopped reading, as one whose request failed for not taking its events, would keep the
                # closed connection open on data it never takes, and with it the server's shutdown, which waits for
                # every connection: as at stop(), it is aborted
                await self._close_writer()
            finally:
                await self._drain_background_tasks()

    async def stop(self) -> None:
        """Stop the event handler without closing shared OpenAI clients."""
        if self._stopping:
            # A repeated stop(), as for a second SIGTERM, waits for the first instead of cancelling its work again
            await self._stopped.wait()
            return
        # Set before the first await, so no STT request starts from here on
        self._stopping = True
        try:
            await self._shut_down()
        finally:
            self._stopped.set()

    async def _shut_down(self) -> None:
        """The work of stop(), done once."""
        try:
            # The STT request in flight is cancelled and the Realtime receiver stopped first, so no request is
            # admitted while the synthesis stops, and nothing of a request is written after the shutdown error. A
            # request that queued its answer before stands
            await self._cancel_stt_request()
            await self._cleanup_realtime_transcription()
            # The client is still connected, so it is told about a synthesis or a transcription cut short before the
            # writer is closed: a stream being recorded, or an audio-stop that has not queued its transcript or error.
            # Once it has, the client has its response, and a shutdown error would be a second one
            await self._stop_sentence_playback(SHUTDOWN_ERROR if self._is_synthesis_in_flight() else None)
            # Decided after the last await, so a stream that reported its own failure meanwhile gets no second error
            recording = self._audio_stream is not None and not self._audio_stream.failed
            if recording or self._stt_answer_owed:
                self._shutdown_reported = True
                await self._write_event_bounded(Error(text=SHUTDOWN_ERROR).event())
        finally:
            await super().stop()
            # A websocket whose closing handshake stalls must not hold up the shutdown of the server
            await self._drain_background_tasks(SHUTDOWN_DRAIN_TIMEOUT)
            # A client that stopped reading leaves the transport open on unsent audio, which Python 3.12 and
            # later wait for at server shutdown
            await self._close_writer()

    def _is_synthesis_in_flight(self) -> bool:
        """Whether a synthesis has started and not ended, so a client cut off from it is owed an error."""
        return (
            self._synthesis.active
            or self._synthesis.audio_started
            or any(not task.done() for task in self._synthesis_tasks)
        )

    async def _handle_select_program(self, select_program: SelectProgram) -> bool:
        """Handle select-program request pinning a program for this connection.

        Program names are only unique within a domain, so the name is resolved
        against ASR and TTS programs independently; one event may select both.
        A selection applies only to the domains whose programs match: prior
        selections in other domains remain active for the lifetime of the
        connection (per the Wyoming protocol), so clients can pick a distinct
        program per domain before sending request events. Unrecognized names
        are dropped per the Wyoming protocol. Selecting an ASR program that does
        not offer the model chosen by an earlier transcribe drops that model, so
        audio streams without a new transcribe use the program's default.
        """
        name = select_program.name
        matched = False

        for program in self._wyoming_info.asr:
            if program.name == name:
                self._selected_asr_program = program
                matched = True
                _LOGGER.debug("Selected ASR program: %s", name)
                if self._current_asr_model is not None and self._current_asr_model not in program.models:
                    # A model selected by an earlier transcribe would outlive the program, which no longer offers
                    # it; streams without a new transcribe use the program's default instead. A language the client
                    # asked for goes with it, which is worth seeing in the log
                    _LOGGER.log(
                        logging.DEBUG if self._current_language is None else logging.INFO,
                        "Dropping model %s and language %s, since program %s does not offer that model",
                        self._current_asr_model.name,
                        self._current_language,
                        name,
                    )
                    self._current_asr_model = None
                    self._current_language = None
                    self._current_bias_names = ()
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
        # No OpenAI-compatible endpoint accepts VAD sensitivity
        if transcribe.vad_sensitivity is not None:
            _LOGGER.debug("Ignoring unsupported Transcribe field vad_sensitivity: %s", transcribe.vad_sensitivity)

        requested_model = self._get_asr_model(transcribe.name)
        requested_language = transcribe.language

        self._current_asr_model = None
        self._current_language = None
        # Names bias the transcription through the STT prompt, ahead of the terms and the Home Assistant names
        self._current_bias_names = tuple(
            (name,)
            for field in (transcribe.transcript_names, transcribe.transcript_terms)
            if isinstance(field, list)
            for name in field
            if isinstance(name, str)
        )
        self._transcribe_rejected = True

        if requested_model:
            if self._is_asr_language_supported(requested_language, requested_model):
                if requested_language:
                    # Sent to the backend as WYOMING_LANGUAGES spells it, since a regional tag only matched it here
                    requested_language = advertised_language(requested_language, requested_model.languages)
                self._current_asr_model = requested_model
                self._current_language = requested_language
                self._transcribe_rejected = False
                if self._audio_stream is not None:
                    self._apply_transcribe_to_audio_stream(self._audio_stream, requested_model, requested_language)
                return True
            self._log_unsupported_asr_language(transcribe.name, requested_language)
        else:
            self._log_unsupported_asr_model(transcribe.name)
        # A stream in progress is left with nothing to transcribe, as one that starts after a refusal is
        abandoned_stream = await self._abandon_audio_stream("a transcribe during it was refused")
        if abandoned_stream is None or not abandoned_stream.failed:
            await self._report_stt_failure()
        return False

    def _apply_transcribe_to_audio_stream(
        self, audio_stream: _AudioStreamState, model: AsrModel, language: str | None
    ) -> None:
        """
        Apply a transcribe that arrived during an audio stream to that stream, as far as its transport allows.

        A /v1/audio/transcriptions recording is only sent at audio-stop, so it takes another model of that
        transport and its language. A Realtime session already has its model and language, and audio that was
        recorded for one transport cannot move to the other. What cannot be applied is reported; the selection
        still applies to the audio streams that start after this one.
        """
        if not audio_stream.realtime and not self._is_asr_model_realtime(model.name):
            audio_stream.model = model
            audio_stream.language = language
            audio_stream.bias_names = self._current_bias_names
            audio_stream.streaming = self._is_asr_model_streaming(model.name)
            _LOGGER.debug("Transcribe during the recording selected model %s, language %s", model.name, language)
            return

        current_model = audio_stream.model.name if audio_stream.model else None
        if (model.name, language) == (current_model, audio_stream.language):
            return
        _LOGGER.warning(
            "Transcribe for model %s, language %s arrived during a %s recording, which keeps model %s, language %s. "
            "Send transcribe before audio-start; the new selection applies from the next audio stream",
            model.name,
            language,
            "Realtime" if audio_stream.realtime else "/v1/audio/transcriptions",
            current_model,
            audio_stream.language,
        )

    async def _handle_audio_start(self, sample_rate: object, audio_width: object, audio_channels: object) -> None:
        """Handle start of audio stream"""
        # A stream the client never stopped must not leave its recording or Realtime session under this one
        await self._abandon_audio_stream("audio-start arrived before its audio-stop")

        # Without a selection the default model is used. It is resolved for every stream rather than kept, so a
        # program selected since then is honoured
        if self._transcribe_rejected:
            model = None
        else:
            model = self._current_asr_model or self._get_asr_model()
        audio_stream = _AudioStreamState(
            model=model,
            language=self._current_language,
            realtime=model is not None and self._is_asr_model_realtime(model.name),
            streaming=model is not None and self._is_asr_model_streaming(model.name),
            bias_names=self._current_bias_names,
        )
        self._audio_stream = audio_stream

        try:
            rate, width, channels = validate_audio_format(sample_rate, audio_width, audio_channels)
            if audio_stream.realtime:
                # The converter owns the frame tail, format conversion, and the 24 kHz mono/16-bit fast path
                audio_stream.converter = RealtimeAudioConverter(rate, width, channels)
        except ValueError as err:
            # Both transports fail the same way, before wave.open() or a Realtime session can raise for the format
            _LOGGER.warning("Rejecting audio-start: %s", err)
            # A failed stream drops its audio and closes the connection on its audio-stop, like other STT failures
            audio_stream.failed = True
            await self._report_stt_failure()
            return

        if audio_stream.realtime:
            await self._run_stt_request(self._handle_realtime_audio_start(audio_stream, rate, width, channels))
            return

        self._is_recording = True
        self._wav_buffer = NamedBytesIO(name="recording.wav")
        self._wav_write_buffer = wave.open(self._wav_buffer, "wb")
        self._wav_write_buffer.setnchannels(channels)
        self._wav_write_buffer.setsampwidth(width)
        self._wav_write_buffer.setframerate(rate)
        _LOGGER.info("Recording started at %d Hz, %d channels, %d bytes per sample", rate, channels, width)

    def _discard_recording(self) -> None:
        """Drop the /v1/audio/transcriptions recording, once sent or when its stream was never stopped."""
        if self._wav_write_buffer is not None:
            with contextlib.suppress(Exception):
                self._wav_write_buffer.close()
            self._wav_write_buffer = None
        if self._wav_buffer is not None:
            self._wav_buffer.close()
            self._wav_buffer = None
        self._is_recording = False

    async def _drop_audio_stream(self) -> _AudioStreamState | None:
        """Drop the audio stream in progress, with its recording or Realtime session, without transcribing it."""
        audio_stream, self._audio_stream = self._audio_stream, None
        self._discard_recording()
        await self._cleanup_realtime_transcription()
        return audio_stream

    async def _abandon_audio_stream(self, reason: str) -> _AudioStreamState | None:
        """Drop an audio stream the client did not stop, and close the transcript sequence it opened."""
        audio_stream = await self._drop_audio_stream()
        if audio_stream is None:
            return None
        _LOGGER.warning("Dropping an audio stream without transcribing it: %s", reason)
        if audio_stream.transcript_started:
            # Closed without a Transcript: a client reads the first one it gets as the result of the request it is
            # waiting for, which is no longer this stream
            await self.write_event(TranscriptStop().event())
        return audio_stream

    async def _handle_audio_chunk(self, chunk: AudioChunk) -> None:
        """Handle audio chunk"""
        if self._audio_stream is not None and self._audio_stream.failed:
            # Its failure was reported once; the audio has nowhere to go
            return
        if self._audio_stream is not None and self._audio_stream.realtime:
            await self._handle_realtime_audio_chunk(self._audio_stream, chunk)
            return

        if self._is_recording and chunk.audio and self._wav_write_buffer:
            self._wav_write_buffer.writeframes(chunk.audio)
        else:
            _LOGGER.warning("Problem handling audio chunk")

    async def _handle_realtime_audio_start(
        self, audio_stream: _AudioStreamState, sample_rate: int, audio_width: int, audio_channels: int
    ) -> None:
        """Open an OpenAI Realtime transcription session."""
        if self._stt_client is None:
            _LOGGER.error("No STT client configured for realtime transcription")
            await self._fail_realtime_audio_stream(audio_stream)
            return

        try:
            # Held while the session is open, as the backend serves it all that time, and released by
            # _cleanup_realtime_transcription. Audio that arrives meanwhile waits in the connection
            self._realtime_stt_slot = await self._stt_limit.acquire()
            self._check_stt_not_stopping()
            self._realtime_connection_manager = self._connect_realtime_transcription()
            connection = await self._enter_realtime_connection(self._realtime_connection_manager)
            self._realtime_connection = connection
            self._realtime_transcript_future = asyncio.get_running_loop().create_future()
            self._realtime_receive_task = asyncio.create_task(
                self._receive_realtime_transcription_events(), name="openai_realtime_transcription"
            )

            self._check_stt_not_stopping()
            prompt = self._get_stt_prompt(audio_stream)
            await connection.session.update(session=self._get_realtime_transcription_session(audio_stream, prompt))
            self._check_stt_not_stopping()
            self._is_recording = True
            await self._write_event_or_fail(TranscriptStart().event())
            audio_stream.transcript_started = True
            self._arm_realtime_idle_timer(audio_stream)
            _LOGGER.info(
                "Realtime recording started at %d Hz, %d channels, %d bytes per sample",
                sample_rate,
                audio_channels,
                audio_width,
            )
        except asyncio.CancelledError:
            # Cancelled while it opened, by stop() or with the handler: what it opened is closed here, so neither
            # the session nor its slot outlives it, even if stop() is itself cancelled before it cleans up
            await self._cleanup_realtime_transcription()
            raise
        except Exception as err:
            # A server error may arrive before session.update fails on the closed socket. Preserve that cause
            # before cleanup discards the future; exception() still includes the startup exception's traceback.
            failure = self._get_realtime_transcription_failure()
            _LOGGER.exception("Error starting realtime transcription: %s", failure if failure is not None else err)
            await self._cleanup_realtime_transcription()
            await self._fail_realtime_audio_stream(audio_stream)

    async def _fail_realtime_audio_stream(self, audio_stream: _AudioStreamState) -> None:
        """
        Report a Realtime session that could not be opened, or that failed while the audio was still arriving, as
        soon as that is known: a client that reads while it streams audio, such as Home Assistant 2026.10 and later,
        ends the request instead of recording on. The rest of the stream's audio is dropped, and its audio-stop
        closes the connection.
        """
        if audio_stream.failed:
            # Reported already, as when the idle limit and a failing chunk meet
            return
        self._is_recording = False
        audio_stream.failed = True
        await self._report_stt_failure()

    def _arm_realtime_idle_timer(self, audio_stream: _AudioStreamState) -> None:
        """(Re)start the time a Realtime session may go without audio before it fails."""
        self._cancel_realtime_idle_timer()
        self._realtime_idle_timer = asyncio.get_running_loop().call_later(
            REALTIME_STT_IDLE_TIMEOUT, self._on_realtime_idle, audio_stream
        )

    def _on_realtime_idle(self, audio_stream: _AudioStreamState) -> None:
        """The idle limit of a Realtime session expired: it fails, unless audio arrives before that runs."""
        self._cancel_realtime_idle_timer()
        self._run_in_background(self._fail_idle_realtime_stream(audio_stream), name="realtime_stt_idle")

    def _cancel_realtime_idle_timer(self) -> None:
        if self._realtime_idle_timer is not None:
            self._realtime_idle_timer.cancel()
            self._realtime_idle_timer = None

    async def _fail_idle_realtime_stream(self, audio_stream: _AudioStreamState) -> None:
        """
        Fail a Realtime stream that got no audio for REALTIME_STT_IDLE_TIMEOUT, closing its session so its slot is
        free for other connections. A stream that has ended, by its audio-stop included, failed, or got audio since the
        limit expired, is left alone; during a shutdown the client still gets one answer, as _report_stt_failure writes
        nothing after stop()'s. Everything up to the queued error is done with no await, so no event, stop() or
        receiver write comes between: the session is detached, the stream failed and its error queued. Its receiver is
        waited for and its socket closed after that, when nothing of the handler is left for it to change.
        """
        if self._audio_stream is not audio_stream or audio_stream.failed or self._realtime_idle_timer is not None:
            return
        _LOGGER.warning(
            "No audio for %g seconds; ending the Realtime transcription session", REALTIME_STT_IDLE_TIMEOUT
        )
        session = self._detach_realtime_session()
        try:
            await self._fail_realtime_audio_stream(audio_stream)
        finally:
            # Also when cancelled during the error write: nothing else owns the detached session. Closed in this
            # task, which drains wait for and cancel, rather than in one started while they may already be ending
            await self._close_realtime_session(session, in_background=False)

    async def _report_stt_failure(self) -> None:
        """
        Tell the client a transcription failed with an error event, after which the connection is closed. Home
        Assistant 2026.8.2 and later report the error and every version reports the closed connection as a failed
        transcription, while an empty transcript would count as nothing said. The write is bounded, since a client
        that has stopped reading must not hold up the close.
        """
        # The answer of the request, as soon as the error is queued
        self._stt_answer_owed = False
        if self._shutdown_reported:
            # stop() has told the client already and closed the writer
            return
        await self._write_event_bounded(Error(text=STT_FAILED_ERROR).event())

    async def _handle_realtime_audio_chunk(self, audio_stream: _AudioStreamState, chunk: AudioChunk) -> None:
        """Append a Wyoming audio chunk to the OpenAI Realtime input buffer."""
        connection = self._realtime_connection
        if not self._is_recording or connection is None:
            _LOGGER.warning("Received realtime audio chunk without an active recording")
            return

        # session.update only sends the settings, so a server that rejects them, or a socket that closes, is only
        # seen once the receiver has read it; the client is told then rather than after the rest of the audio
        failure = self._get_realtime_transcription_failure()
        if failure is not None:
            _LOGGER.error("Realtime transcription failed while the audio was arriving: %s", failure)
            await self._cleanup_realtime_transcription()
            await self._fail_realtime_audio_stream(audio_stream)
            return

        if not chunk.audio:
            return

        try:
            audio = audio_stream.realtime_converter.convert_chunk(chunk.audio)
            if audio:
                # Only audio restarts the idle limit, so a client sending empty chunks does not keep the session, or
                # its slot
                self._arm_realtime_idle_timer(audio_stream)
            await self._append_realtime_audio(audio, connection)
        except Exception as err:
            # The session cannot take the rest of the audio either, so it fails now, with one log entry
            _LOGGER.exception("Error sending realtime audio chunk: %s", err)
            await self._cleanup_realtime_transcription()
            await self._fail_realtime_audio_stream(audio_stream)

    async def _append_realtime_audio(self, audio: bytes, connection: Any) -> None:
        """Append PCM16 audio to the Realtime input buffer of the given connection."""
        if not audio:
            return
        await connection.input_audio_buffer.append(audio=base64.b64encode(audio).decode("ascii"))

    async def _handle_realtime_audio_stop(self, audio_stream: _AudioStreamState) -> bool:
        """Commit realtime audio and emit the final transcription. Returns False to close a connection it failed on."""
        connection = self._realtime_connection
        if not self._is_recording or connection is None:
            _LOGGER.warning("Received realtime audio stop event without recording")
            # Nothing is sent, so nothing is owed
            self._stt_answer_owed = False
            await self._cleanup_realtime_transcription()
            return True

        self._is_recording = False
        # Captured before the flush append awaits, since stop() can clear the handler's attributes meanwhile
        transcript_future = self._realtime_transcript_future

        transcript_sent = False
        try:
            if transcript_future is None:
                raise RealtimeTranscriptionError("Realtime transcription session was not initialized")

            # The commit is a websocket send that can stall too, so it shares the time limit
            raised_inside: TimeoutError | None = None
            try:
                async with asyncio.timeout(REALTIME_STT_TRANSCRIPT_TIMEOUT):
                    try:
                        await self._append_realtime_audio(audio_stream.realtime_converter.flush(), connection)
                        if transcript_future.done():
                            # stop() closed the session during the flush: await its failure instead of committing
                            transcript = await transcript_future
                        else:
                            self._check_stt_not_stopping()
                            await connection.input_audio_buffer.commit()
                            transcript = await transcript_future
                    except TimeoutError as err:
                        # Another time limit's, such as a transport timeout the receiver failed with, even one raised
                        # while the deadline cancels the wait: reported as it is
                        raised_inside = err
                        raise
            except TimeoutError as err:
                if err is raised_inside:
                    raise
                # The deadline's own: a server that stalls with the socket open would otherwise hold the Wyoming
                # client forever
                raise RealtimeTranscriptionError(
                    f"No Realtime transcript received within {REALTIME_STT_TRANSCRIPT_TIMEOUT:g} seconds "
                    "of the end of the audio"
                ) from err
            if transcript:
                _LOGGER.info("Successfully transcribed realtime stream: %s", truncate_for_log(transcript))
            else:
                _LOGGER.warning("Received empty realtime transcription result")
            # The session has delivered all it will, so it closes before the final events are written: a client
            # slow to take them does not keep it, and the STT slot it holds, from the next request
            await self._cleanup_realtime_transcription()
            await self._write_transcript(transcript)
            transcript_sent = True
            await self.write_event(TranscriptStop().event())
        except Exception as err:
            _LOGGER.exception("Error during realtime transcription: %s", err)
            if not transcript_sent:
                # The receiver is stopped first, so no late delta follows the error
                await self._cleanup_realtime_transcription()
                await self._report_stt_failure()
                return False
        finally:
            await self._cleanup_realtime_transcription()
        return True

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

    def _get_stt_prompt(self, audio_stream: _AudioStreamState) -> str | None:
        """
        The STT prompt: STT_PROMPT, followed by the names sent with transcribe and the names read from Home
        Assistant. A prompt in the extra body of the stream's transport replaces it, so it is STT_PROMPT alone then, as
        without names; OpenAI's Realtime sessions for some models take none, so it is None for those.
        """
        model = audio_stream.model
        extra_body = self._stt_realtime_extra_body if audio_stream.realtime else self._stt_extra_body
        if model is None or (extra_body and "prompt" in extra_body):
            return self._stt_prompt
        hass_names = self._hass_names.names if self._hass_names is not None else ()
        promptless = self._is_openai_stt_model(model.name, OPENAI_PROMPTLESS_REALTIME_STT_MODEL_PREFIXES)
        if audio_stream.realtime and promptless:
            # Called once per stream; only a prompt that would otherwise have been sent is worth reporting
            if self._stt_prompt is not None or audio_stream.bias_names or hass_names:
                _LOGGER.debug("Leaving out the STT prompt: %s does not support one", model.name)
            return None
        groups = audio_stream.bias_names + hass_names
        prompt = build_prompt(self._stt_prompt, groups, self._stt_prompt_max_chars)
        if groups:
            _LOGGER.debug("STT prompt: %s", prompt)
        return prompt

    def _get_realtime_transcription_session(
        self, audio_stream: _AudioStreamState, prompt: str | None
    ) -> dict[str, object]:
        """Build a Realtime transcription session update payload."""
        if not audio_stream.model:
            raise RealtimeTranscriptionError("No ASR model set for realtime transcription")
        model_name = audio_stream.model.name

        # STT_EXTRA_BODY targets /v1/audio/transcriptions, so sessions only take the Realtime extra body
        language, languages, extra_body = self._apply_plural_stt_languages(
            model_name, audio_stream.language, self._stt_realtime_extra_body
        )

        transcription: dict[str, object] = {"model": model_name}
        if language is not None:
            transcription["language"] = language
        if languages is not None:
            transcription["languages"] = languages
        if prompt is not None:
            transcription["prompt"] = prompt
        transcription.update(extra_body)
        # A null in the extra body drops the field, e.g. {"prompt": null} for a model that takes no prompt
        transcription = {key: value for key, value in transcription.items() if value is not None}

        return {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": dict(REALTIME_AUDIO_FORMAT),
                    "transcription": transcription,
                    "turn_detection": None,
                }
            },
        }

    async def _receive_realtime_transcription_events(self) -> None:
        """Accumulate Realtime deltas into Wyoming transcript snapshots for this session."""
        connection = self._realtime_connection
        transcript_future = self._realtime_transcript_future
        if connection is None or transcript_future is None:
            return

        full_text = ""
        try:
            async for event in connection:
                if self._realtime_connection is not connection:
                    # A cleanup has detached this session, but the connection delivered one more event: nothing more
                    # of it is written, as a shutdown error may already have ended it
                    return
                event_type = getattr(event, "type", "")
                if event_type == "conversation.item.input_audio_transcription.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        full_text += delta
                        _LOGGER.debug("Realtime transcription chunk: %s", delta)
                        # Always the transcript so far: wyoming 1.11 requires whole transcripts from a streaming ASR
                        # service, so there is deliberately no mode that sends the pre-1.11 fragments
                        await self._write_event_or_fail(TranscriptChunk(text=full_text).event())
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = _final_transcript(getattr(event, "transcript", None), full_text)
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
            # Settled before the receiver ends, however the cancellation came: from a cleanup, which fails it too, or
            # from the connection itself, which leaves nobody else to tell an audio-stop or the next chunk
            if not transcript_future.done():
                transcript_future.set_exception(RealtimeTranscriptionError("Realtime connection was cancelled"))
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

    def _get_realtime_transcription_failure(self) -> BaseException | None:
        """The error the Realtime session has already failed with, if any, before its transcript is awaited."""
        transcript_future = self._realtime_transcript_future
        if transcript_future is None or not transcript_future.done() or transcript_future.cancelled():
            return None
        return transcript_future.exception()

    async def _run_stt_request[T](self, request: Coroutine[Any, Any, T]) -> T | None:
        """
        Run the backend part of an STT request, from waiting for a slot to writing its answer, as a task stop() can
        cancel. Returns its result, or None when stop() cancelled it or had been called before it started; stop()
        answers the client then, and disconnect() waits for it to have done so.
        """
        if self._stopping:
            request.close()
            return None
        handler_task = asyncio.current_task()
        cancels_before = handler_task.cancelling() if handler_task is not None else 0

        def handler_cancelled() -> bool:
            return handler_task is not None and handler_task.cancelling() > cancels_before

        task = asyncio.create_task(request, name="stt_request")
        self._stt_request = task
        try:
            result = await task
        except asyncio.CancelledError:
            if handler_cancelled():
                # The handler itself is cancelled, not only its request
                raise
            if not self._stopping:
                # Neither stop() nor the handler cancelled it, as with a backend client that raises CancelledError or
                # cancels its own task: the request failed
                _LOGGER.error("STT request was cancelled outside a shutdown; reporting it as failed")
                await self._fail_stt_request()
            return None
        finally:
            if self._stt_request is task:
                self._stt_request = None
        if handler_cancelled():
            # The handler was cancelled while the request ran, and the request ended without that cancellation, as
            # with a backend client that catches it: the handler still ends
            raise asyncio.CancelledError
        return result

    def _check_stt_not_stopping(self) -> None:
        """
        Called before each backend operation of an STT request after the first: none starts once stop() has begun,
        even when the backend client caught the shutdown's cancellation and returned. The request then ends as
        cancelled, and stop() answers it.
        """
        if self._stopping:
            raise asyncio.CancelledError

    async def _fail_stt_request(self) -> None:
        """
        Report an STT request that ended cancelled although nobody cancelled it, as other failed requests are, unless
        it had already answered. The session is closed first, so no late delta follows the error; the failure is then
        marked and its error queued with no await between, so stop() either sees it or answers in its place.
        """
        await self._cleanup_realtime_transcription()
        audio_stream = self._audio_stream
        if audio_stream is not None and not audio_stream.failed:
            # A Realtime session that did not open: its audio is dropped, and its audio-stop closes the connection
            await self._fail_realtime_audio_stream(audio_stream)
        elif self._stt_answer_owed:
            await self._report_stt_failure()

    async def _cancel_stt_request(self) -> None:
        """
        Cancel the STT request in flight, if any, and wait for it to end, as stop() does before it answers. As for
        the synthesis requests stop() cancels, the wait has no time limit: a request that ended can write nothing
        after the shutdown error, and a cancelled one ends once its cleanup has run.
        """
        task = self._stt_request
        if task is None or task.done():
            return
        _LOGGER.info("Cancelling the STT request in flight: the server is shutting down")
        task.cancel()
        await asyncio.wait([task])

    async def _write_transcript(self, text: str) -> None:
        """Write the transcript of an audio-stop, which is its answer as soon as it is queued."""
        self._stt_answer_owed = False
        await self.write_event(Transcript(text=text).event())

    def _detach_realtime_session(self) -> _DetachedRealtimeSession:
        """
        Detach the Realtime session from the handler, with no await: its idle limit is stopped, its slot released, and
        its receiver, which writes nothing once its session is detached, told to stop. A cleanup that overlaps this
        one finds nothing left to close or release.
        """
        self._cancel_realtime_idle_timer()
        slot, self._realtime_stt_slot = self._realtime_stt_slot, None
        if slot is not None:
            # The session has ended. As for Realtime TTS, the limit counts requests rather than sockets, so the next
            # session may open while this one's closing handshake is still running
            slot.release()
        session = _DetachedRealtimeSession(
            receive_task=self._realtime_receive_task,
            transcript_future=self._realtime_transcript_future,
            connection_manager=self._realtime_connection_manager,
            connection=self._realtime_connection,
        )
        self._realtime_receive_task = None
        self._realtime_transcript_future = None
        self._realtime_connection_manager = None
        self._realtime_connection = None
        if session.receive_task is not None and not session.receive_task.done():
            session.receive_task.cancel()
        return session

    async def _close_realtime_session(self, session: _DetachedRealtimeSession, *, in_background: bool = True) -> None:
        """
        Wait for a detached session's receiver to stop, then close its connection: in the background, or, for a
        caller that is background cleanup itself, in its own task, so whoever waits for or cancels that task has the
        close too.
        """
        cleaner = asyncio.current_task()
        cancels_before = cleaner.cancelling() if cleaner is not None else 0
        try:
            if session.receive_task is not None and not session.receive_task.done():
                # Stopped before the session closes, and before this returns, so no late delta follows what the
                # caller writes next
                with contextlib.suppress(asyncio.CancelledError):
                    await session.receive_task
                if cleaner is not None and cleaner.cancelling() > cancels_before:
                    # The task cleaning up was cancelled while it waited, as at Ctrl+C, however the receiver ended:
                    # that cancellation goes on once the session is closed below
                    raise asyncio.CancelledError
        finally:
            transcript_future = session.transcript_future
            if transcript_future is not None:
                if not transcript_future.done():
                    # Failed rather than cancelled, so an audio-stop waiting on it ends through the failure path and is
                    # logged, instead of ending as a cancelled task
                    transcript_future.set_exception(
                        RealtimeTranscriptionError("Realtime transcription session was closed")
                    )
                if not transcript_future.cancelled():
                    # Read, so the failure of a stream that is dropped is not logged later as never retrieved
                    transcript_future.exception()

            # As with Realtime TTS, the closing handshake must not hold up the Wyoming client
            close = None
            if session.connection_manager is not None:
                close = session.connection_manager.__aexit__(None, None, None)
            elif session.connection is not None:
                close = session.connection.close()
            if close is not None and in_background:
                self._run_in_background(close, name="openai_realtime_stt_close")
            elif close is not None:
                try:
                    await close
                except Exception as err:
                    _LOGGER.debug("Closing the Realtime transcription websocket failed: %s", err)

    async def _cleanup_realtime_transcription(self) -> None:
        """Close the realtime connection and background receive task, and release the STT slot the session held."""
        await self._close_realtime_session(self._detach_realtime_session())

    async def _handle_audio_stop(self) -> bool:
        """Handle end of audio stream and perform transcription. Returns False to close a connection it failed on."""
        audio_stream, self._audio_stream = self._audio_stream, None
        if audio_stream is None:
            _LOGGER.warning("Received audio stop event without recording")
            return True

        if audio_stream.failed:
            # Its error was sent when it failed; closing the connection ends the request for clients that ignore it
            return False

        # Owed a transcript or an error until one is queued. Marked before the request is scheduled, so stop() also
        # answers one it cancels before it has started
        self._stt_answer_owed = True
        if audio_stream.realtime:
            request = self._handle_realtime_audio_stop(audio_stream)
        else:
            request = self._handle_http_audio_stop(audio_stream)
        result = await self._run_stt_request(request)
        # None when stop() cancelled the request: it answers the client, and the connection closes
        return False if result is None else result

    async def _handle_http_audio_stop(self, audio_stream: _AudioStreamState) -> bool:
        """Send a /v1/audio/transcriptions recording and emit its transcription. Returns False to close on failure."""
        if not self._is_recording or not self._wav_buffer:
            _LOGGER.warning("Received audio stop event without recording")
            # Nothing is sent, so nothing is owed
            self._stt_answer_owed = False
            return True

        self._is_recording = False

        transcript_sent = False
        try:
            # Close the WAV file
            if self._wav_write_buffer:
                self._wav_write_buffer.close()
                self._wav_write_buffer = None

            # Reset buffer position to start
            self._wav_buffer.seek(0)

            model = audio_stream.model
            if not model:
                raise RuntimeError("No ASR model set for transcription")

            if self._stt_client is None:
                raise RuntimeError("No STT client configured for transcription")

            # Send to OpenAI for transcription
            extra_body = self._get_stt_extra_body()
            use_streaming = get_extra_body_boolean_field(
                extra_body,
                field_name="stream",
                default=audio_stream.streaming,
                body_name="STT",
            )

            language, languages, extra_body = self._apply_plural_stt_languages(
                model.name, audio_stream.language, extra_body
            )

            prompt = self._get_stt_prompt(audio_stream)
            transcription_kwargs = {
                "file": self._wav_buffer,
                "model": model.name,
                "language": language if language is not None else omit,
                "languages": languages if languages is not None else omit,
                "temperature": self._stt_temperature if self._stt_temperature is not None else omit,
                "prompt": prompt if prompt is not None else omit,
                "response_format": "json",
                "stream": use_streaming if use_streaming else omit,
            }
            if extra_body:
                transcription_kwargs["extra_body"] = extra_body

            # Held until the response, streamed or not, has been read, and not while the final events are written to
            # a client that may be slow to take them
            async with self._stt_limit.slot():
                self._check_stt_not_stopping()
                transcription = await self._stt_client.audio.transcriptions.create(**transcription_kwargs)
                streamed = isinstance(transcription, AsyncStream)

                if isinstance(transcription, AsyncStream):
                    _LOGGER.debug("Handling streaming transcription response")
                    full_text = ""
                    completed = False
                    # Closed however it ends, as when stop() cancels the request, so the response does not stay open
                    # after its slot is released
                    async with transcription:
                        # Reading the response is backend work too, so no read starts once stop() has begun: each
                        # is preceded by a check, from the first one after transcript-start on
                        self._check_stt_not_stopping()
                        await self._write_event_or_fail(TranscriptStart().event())
                        self._check_stt_not_stopping()
                        async for chunk in transcription:
                            if chunk.type == "transcript.text.done":
                                full_text = _final_transcript(chunk.text, full_text)
                                # The complete transcription, so nothing more is read
                                completed = True
                                break
                            if chunk.type == "transcript.text.delta" and chunk.delta:
                                full_text += chunk.delta
                                _LOGGER.debug("Transcribed chunk: %s", chunk.delta)
                                # A snapshot, as on the Realtime path, with no pre-1.11 fragment mode
                                await self._write_event_or_fail(TranscriptChunk(text=full_text).event())
                            self._check_stt_not_stopping()
                    if not completed:
                        # Ended without the completed transcription, as a backend cut off by the shutdown may: outside
                        # one, the deltas so far are the transcript, but during one stop() answers
                        self._check_stt_not_stopping()
                    text = full_text
                    if text:
                        _LOGGER.info("Successfully transcribed stream: %s", truncate_for_log(text))
                    else:
                        _LOGGER.warning(
                            "Received empty transcription from stream."
                            " If this is unexpected, please check your"
                            " STT_STREAMING_MODELS configuration."
                        )

                elif isinstance(transcription, TranscriptionCreateResponse):
                    # Handle non-streaming response, which is read whole by create()
                    _LOGGER.debug("Handling non-streaming transcription response")
                    text = transcription.text
                    if text:
                        _LOGGER.info("Successfully transcribed: %s", truncate_for_log(text))
                    else:
                        _LOGGER.warning("Received empty transcription result")

                else:
                    raise TypeError(f"Unexpected transcription response type: {type(transcription)}")

            if not streamed:
                await self.write_event(TranscriptStart().event())
            await self._write_transcript(text)
            transcript_sent = True
            await self.write_event(TranscriptStop().event())

        except Exception as e:
            _LOGGER.exception("Error during transcription: %s", e)
            # As on the Realtime path, the client is told instead of waiting for a transcript
            if not transcript_sent:
                await self._report_stt_failure()
                return False
        finally:
            self._discard_recording()
        return True

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

    def _is_openai_stt_model(self, model_name: str, prefixes: tuple[str, ...]) -> bool:
        """Check if an ASR model is one of a family of OpenAI models, named by prefix, on the OPENAI backend."""
        # Not limited to the official domain: proxies in front of OpenAI serve these models the same way
        if self._stt_client is None or self._stt_client.backend != OpenAIBackend.OPENAI:
            return False
        return model_name.startswith(prefixes)

    def _uses_plural_stt_languages(self, model_name: str) -> bool:
        """Check if an ASR model takes a `languages` list instead of the singular `language` field."""
        return self._is_openai_stt_model(model_name, OPENAI_PLURAL_LANGUAGE_STT_MODEL_PREFIXES)

    def _apply_plural_stt_languages(
        self, model_name: str, language: str | None, extra_body: dict[str, object] | None
    ) -> tuple[str | None, list[str] | None, dict[str, object]]:
        """Return the `language` and `languages` fields to send, and a copy of extra_body that does not conflict."""
        extra_body = dict(extra_body or {})
        if "languages" in extra_body:
            # On every model and server an extra_body `languages` list replaces the singular field, from the
            # request or the body, so the two are never sent together
            extra_body.pop("language", None)
            return None, None, extra_body
        if not self._uses_plural_stt_languages(model_name):
            return language, None, extra_body
        # These models reject `language` alongside `languages`, so a singular extra_body override is translated
        override = extra_body.pop("language", language)
        if not isinstance(override, str) or not override:
            # Without a usable language the model detects it
            return None, None, extra_body
        return None, [override], extra_body

    def _is_tts_voice_realtime(self, voice: TtsVoiceModel) -> bool:
        """Check if a TTS voice should be synthesized over OpenAI Realtime."""
        return voice.model_name in self._tts_realtime_models

    def _splits_plain_synthesize(self, voice: TtsVoiceModel) -> bool:
        """
        Check if a plain synthesize request is split into sentences like a streaming synthesis.
        Only a streamed Realtime voice needs it: long text is where a conversational model drifts from the
        script. /v1/audio/speech already streams the audio of one request and keeps its prosody across sentences.
        """
        return self._is_tts_voice_realtime(voice) and self._is_tts_voice_streaming(voice.name)

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

        segmenter = _get_boundary_detector(language)
        # preserve_whitespace=False strips leading/trailing whitespace from each sentence

        merged, current_chunk, current_words = self._merge_sentences(
            list(segmenter.segment(text)), min_words, max_chars
        )
        chunks = [chunk.strip() for chunk in merged]

        # A remainder below the minimum joins the chunk before it rather than being dropped
        if current_chunk:
            if chunks and min_words and current_words < min_words:
                chunks[-1] = f"{chunks[-1]} {current_chunk.strip()}"
            else:
                chunks.append(current_chunk.strip())

        return chunks if chunks else [text]  # Fallback to original text if no valid chunks

    def _merge_sentences(
        self, sentences: list[str], min_words: int | None, max_chars: int | None, joiner: str = " "
    ) -> tuple[list[str], str, int]:
        """
        Merge sentences into chunks within the configured limits.

        Returns the finished chunks, the chunk still open after the last sentence, and its word count. Without
        limits every sentence is its own chunk. A chunk below `min_words` keeps growing, past `max_chars` if
        need be, so text is never dropped; one that meets the minimum is closed once the next sentence would
        exceed `max_chars`, or right away when no maximum is set. Words are counted sentence by sentence, so a
        sentence in a language written without spaces is one word however the sentences are joined.
        """
        chunks: list[str] = []
        current_chunk = ""
        current_words = 0

        for sentence in sentences:
            potential_chunk = current_chunk + joiner + sentence if current_chunk else sentence
            meets_min = not min_words or current_words >= min_words
            if current_chunk and meets_min and (not max_chars or len(potential_chunk) > max_chars):
                chunks.append(current_chunk)
                current_chunk = sentence
                current_words = len(sentence.split())
            else:
                current_chunk = potential_chunk
                current_words += len(sentence.split())

        return chunks, current_chunk, current_words

    async def _process_ready_sentences(self, sentences: list[str]) -> bool:
        """
        Process complete sentences of a streamed synthesis for immediate TTS synthesis.

        API requests start concurrently for all sentences, limited by the configured number of concurrent
        TTS requests, with sequential playback to maintain correct audio order.

        Args:
            sentences (list[str]): Complete sentences ready for synthesis.

        Returns:
            bool: True if processing succeeded, False if synthesis was aborted.
        """
        voice = self._synthesis.voice
        valid_sentences = [s for s in sentences if s.strip()]
        if voice is None or not valid_sentences:
            _LOGGER.debug("No non-empty sentences available for incremental synthesis.")
            return True

        try:
            if not await self._play_sentences(valid_sentences, voice):
                return await self._abort_synthesis()
            return True
        except Exception as e:
            _LOGGER.exception("Error processing ready sentences: %s", e)
            return await self._abort_synthesis()

    async def _play_sentences(self, sentences: list[str], voice: TtsVoiceModel) -> bool:
        """
        Synthesize sentences concurrently and play them in order on this synthesis' audio stream.

        Args:
            sentences (list[str]): Non-empty text chunks, in playback order.
            voice (TtsVoiceModel): Voice to use for synthesis.

        Returns:
            bool: True if every sentence was played or skipped, False if playback failed. Pending sentences
                and an open audio stream are then left to `_stop_sentence_playback`, which the caller runs.
        """
        _LOGGER.info("Starting concurrent synthesis for %d sentences", len(sentences))
        # Kept across the awaits, like _stop_sentence_playback does, to see whether it ended this synthesis
        state = self._synthesis

        # Create ALL tasks with IDs - API calls start concurrently
        # Semaphore limits actual concurrency to the configured number of concurrent TTS requests
        synthesis_tasks = [
            (
                f"sentence_{i}",
                self._create_synthesis_task(
                    self._get_tts_audio_stream(sentence, voice, task_id=f"sentence_{i}"),
                    name=f"sentence_{i}",
                ),
            )
            for i, sentence in enumerate(sentences)
        ]

        # Await tasks IN ORDER for sequential playback
        # Enable streaming for whichever task we're currently awaiting
        for i, (task_id, task) in enumerate(synthesis_tasks):
            sentence_preview = truncate_for_log(sentences[i], 50)
            _LOGGER.debug("Processing sentence %d/%d: %s", i + 1, len(sentences), sentence_preview)

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
                return False
            except Exception as err:
                _LOGGER.exception(
                    "Unexpected error while synthesizing sentence %d (%s): %s",
                    i + 1,
                    sentence_preview,
                    err,
                )
                return False
            finally:
                self._allow_streaming_task_id = None

            # stop() may have ended the synthesis while this sentence finished; with nothing left to cancel it
            # cannot stop the sentences that are already buffered, so they are not played after its error
            if state.failed:
                return False

            if result.streamed:
                _LOGGER.debug("Sentence %d streamed directly with minimal latency", i + 1)
                continue

            # Otherwise, task completed and buffered - stream the buffered data now. Tracked like the requests,
            # so stop() ends a write the client is slow to take instead of the next sentence following it
            if not await self._create_synthesis_task(
                self._stream_audio_to_wyoming(result.audio or b"", self._get_tts_audio_format(voice), sentences[i]),
                name=f"{task_id}_playback",
            ):
                _LOGGER.error("Failed to stream sentence %d (%s) to Wyoming", i + 1, sentence_preview)
                return False

            _LOGGER.debug(
                "Successfully streamed buffered sentence %d, timestamp: %.2f",
                i + 1,
                self._synthesis.timestamp,
            )

        return True

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

    def _create_synthesis_task[T](self, coro: Coroutine[Any, Any, T], *, name: str) -> asyncio.Task[T]:
        """Start a synthesis task that is cancelled if the session aborts or the client disconnects."""
        # Its failure is reported by whoever awaits it
        return self._track_task(coro, name=name, tasks=self._synthesis_tasks, log_failure=False)

    def _run_in_background(self, coro: Any, *, name: str) -> None:
        """Run cleanup that should not delay the audio stream; awaited when the client disconnects."""
        self._track_task(coro, name=name, tasks=self._background_tasks, log_failure=True)

    def _cancel_cleanup_once(self, task: asyncio.Task[Any]) -> None:
        """
        Cancel a cleanup task unless it is done or was cancelled already: a second cancellation would cut its own
        cleanup short, as when the drains of stop() and disconnect() overlap.
        """
        if task.done() or task in self._cancelled_cleanup or task.cancelling():
            return
        self._cancelled_cleanup.add(task)
        task.cancel()

    async def _drain_background_tasks(self, timeout: float | None = None) -> None:
        """
        Wait for pending background cleanup, and for cleanup the tasks waited for start meanwhile, as the idle
        failure's websocket close, within the same time. What a timeout leaves unfinished is cancelled and given as
        long again to end, so a close that ignores its cancellation cannot hold up shutdown; what still runs then is
        left to the event loop. A drain that is itself cancelled cancels what it was waiting for, which is not left
        without one.
        """
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        tasks: set[asyncio.Task[Any]] = set()
        try:
            while self._background_tasks:
                remaining = None if deadline is None else deadline - loop.time()
                if remaining is not None and remaining <= 0:
                    break
                tasks = set(self._background_tasks)
                await asyncio.wait(tasks, timeout=remaining)
            pending = set(self._background_tasks)
            for task in pending:
                self._cancel_cleanup_once(task)
            if pending:
                _, pending = await asyncio.wait(pending, timeout=timeout)
                if pending:
                    _LOGGER.warning(
                        "%d background cleanup task(s) did not end once cancelled; leaving them to the event loop",
                        len(pending),
                    )
        except asyncio.CancelledError:
            for task in tasks | self._background_tasks:
                self._cancel_cleanup_once(task)
            raise

    async def _close_writer(self) -> None:
        """
        Wait for the closed writer to finish flushing, then abort a connection a client that stopped reading left
        open. Python 3.12 and later wait for every client transport at server shutdown, also one whose handler has
        ended, so one that never takes the rest of its data would hold the process up until it is forced shut.
        """
        # Never cancelled, as it waits on the close waiter of the stream protocol, which a cancellation would cancel
        # for every later wait_closed(), as that of a concurrent or repeated stop(); an abort completes it instead
        wait_closed = asyncio.create_task(self.writer.wait_closed(), name="wyoming_writer_close")
        # Its failure is read even when nobody waits for it any more, as after a cancellation
        wait_closed.add_done_callback(_retrieve_result)
        raised_by_close: BaseException | None = None
        try:
            async with asyncio.timeout(SHUTDOWN_CLOSE_TIMEOUT):
                try:
                    await asyncio.shield(wait_closed)
                except TimeoutError as err:
                    # The close's own, which leaves the connection closed: not the deadline's
                    raised_by_close = err
                    raise
        except TimeoutError as err:
            if err is raised_by_close:
                _LOGGER.debug("Waiting for the client connection to close failed: %s", err)
                return
            _LOGGER.warning(
                "Client did not take the rest of its data within %g seconds; aborting its connection",
                SHUTDOWN_CLOSE_TIMEOUT,
            )
            self.writer.transport.abort()
            # The abort completes the waiter; its result is not needed
            with contextlib.suppress(Exception):
                await asyncio.shield(wait_closed)
        except asyncio.CancelledError:
            # Interrupted, as at Ctrl+C: aborted rather than left open on data the client may never take, which
            # would hold up the server's shutdown, and the cancellation goes on
            self.writer.transport.abort()
            raise
        except Exception as err:
            _LOGGER.debug("Waiting for the client connection to close failed: %s", err)

    async def _write_event_or_fail(self, event: Event) -> None:
        """
        Write an event of a request that may hold a shared STT or TTS slot. A client that does not take it within
        CLIENT_WRITE_TIMEOUT fails the request with SlowClientError, so a client that stopped reading cannot keep the
        slot from other connections. For an abort or a shutdown, _write_event_bounded gives up quietly instead.
        """
        raised_by_write: TimeoutError | None = None
        try:
            async with asyncio.timeout(CLIENT_WRITE_TIMEOUT):
                try:
                    await self.write_event(event)
                except TimeoutError as err:
                    # The write's own, even one raised while this limit cancels it: passed on as it is
                    raised_by_write = err
                    raise
        except TimeoutError as err:
            if err is raised_by_write:
                raise
            # The limit's own. Named, so the log of the failed request gives the cause; the cancellation the limit
            # used is not part of it
            raise SlowClientError(
                f"Client did not take its {event.type} event within {CLIENT_WRITE_TIMEOUT:g} seconds"
            ) from None

    async def _write_event_bounded(self, event: Event) -> bool:
        """
        Write an event of an abort or a shutdown, which a client that stopped reading must not hold up.
        Returns whether the client took it; one that did not is not offered the events that would follow.
        """
        try:
            await asyncio.wait_for(self.write_event(event), TTS_STREAM_STOP_TIMEOUT)
        except Exception:
            return False
        return True

    async def _stop_sentence_playback(self, error: str | None = None) -> bool:
        """
        Cancel pending synthesis requests and close their audio stream without ending the text session.
        Playback only reports failure; this is the one place that tells the client. It runs when the handler
        of a synthesis event unwinds, and from stop() while the client is still connected. It does nothing
        after a synthesis that completed, which has no request pending and closed its own stream.

        With an `error` the synthesis failed, and the client gets an error event first. Home Assistant takes
        an audio stop as the end of a complete plain synthesis and caches the audio, and reads on past it in
        a streaming one, so a failure is only told apart by that event; it is sent once per synthesis. A
        failed plain synthesis then gets no audio stop: clients that do not know the error event see the
        connection close instead, which the caller does by returning False.

        Returns whether the client took what was written.
        """
        # Kept across the awaits: stop() runs beside the handler of the synthesis, which unwinds once its
        # requests are cancelled, ends the synthesis itself and leaves the state of the next one behind
        state = self._synthesis
        tasks = [task for task in self._synthesis_tasks if not task.done()]
        for task in tasks:
            task.cancel()
        taken = True
        if error is not None and not state.failed:
            state.failed = True
            # Written before the requests are awaited, so that handler closes the stream after it, not before
            taken = await self._write_event_bounded(Error(text=error).event())
        if tasks:
            # Awaited so they no longer hold backend connections
            await asyncio.gather(*tasks, return_exceptions=True)
        if state.audio_started:
            # Closed for good even if the client does not take the stop, so nobody sends it a second time
            state.audio_started = False
            if taken and not (state.plain and state.failed):
                taken = await self._write_event_bounded(AudioStop(timestamp=int(state.timestamp)).event())
        return taken

    async def _abort_synthesis(self) -> bool:
        """Abort the current synthesis session, emitting error and stop events and resetting state."""
        # The error comes first: a client stops reading at synthesize-stopped
        if await self._end_synthesis(TTS_FAILED_ERROR):
            await self._write_event_bounded(SynthesizeStopped().event())
        return False

    async def _end_synthesis(self, error: str | None = None) -> bool:
        """
        Leave nothing of the current synthesis behind: no pending request, no open stream, no state.
        `error` and the result are those of `_stop_sentence_playback`.
        """
        taken = await self._stop_sentence_playback(error)
        self._synthesis = _SynthesisState()
        self._allow_streaming_task_id = None
        return taken

    def _log_unsupported_asr_model(self, model_name: str | None = None):
        """Log an unsupported ASR model"""
        if model_name:
            _LOGGER.warning("Unsupported ASR model: %s", model_name)
        else:
            _LOGGER.warning("No ASR models specified")

    def _is_asr_language_supported(self, language: str | None, model: AsrModel) -> bool:
        """Check if a language is supported by an ASR model; regional tags match as they do for TTS voices"""
        return supports_language(language, model.languages)

    def _log_unsupported_asr_language(self, model_name: str | None, language: str | None):
        """Log an unsupported ASR language"""
        _LOGGER.error("Unsupported ASR model %s for language %s", model_name, language)

    def _get_voice(self, name: str) -> TtsVoiceModel | None:
        """Get a TTS voice by name or None"""
        for voice in self._iter_tts_voices():
            if voice.name == name:
                return voice
        return None

    def _get_default_voice(self, language: str | None = None) -> TtsVoiceModel | None:
        """
        Get the voice for a request that names none: one of the selected program, or else of the first program
        with voices, which is the only one Home Assistant shows. Within that program the speech API is preferred
        over Realtime among language-compatible voices, so a Realtime voice is chosen only when no speech API voice
        of the program supports the language. A language-only request stays within that program. The language
        selects the voice as wyoming 1.11 defines it ("selects a voice if name is not set"), even though every voice
        currently advertises the same WYOMING_LANGUAGES.
        """
        voices = self._get_default_program_voices()
        compatible = [voice for voice in voices if not language or self._is_tts_language_supported(language, voice)]
        return self._prefer_speech_api_voices(compatible)[0] if compatible else None

    def _get_default_program_voices(self) -> list[TtsVoiceModel]:
        """The voices of the selected program, or else of the first program with voices; empty without any."""
        programs = [self._selected_tts_program] if self._selected_tts_program else self._wyoming_info.tts
        for program in programs:
            if program.voices:
                return [cast(TtsVoiceModel, voice) for voice in program.voices]
        return []

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
        if self._selected_tts_program and any(
            voice.name == requested_voice for program in self._wyoming_info.tts for voice in program.voices
        ):
            # Still advertised, by another program than the one the client selected: refused, not moved to a model
            _LOGGER.warning(
                "Voice %s is offered by another program than the selected %s",
                requested_voice,
                self._selected_tts_program.name,
            )
            return None
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
        """Check if a language is supported by a TTS voice; a regional tag matches its bare language either way"""
        return supports_language(language, voice.languages)

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
        voice = self._get_voice(requested_voice) if requested_voice else self._get_default_voice(requested_language)
        if voice is None and requested_voice:
            backend_matches = self._get_voices_by_backend_name(requested_voice)
            if not backend_matches:
                voice = self._get_voice_by_suffixed_name(requested_voice)
            elif len(backend_matches) == 1:
                voice = backend_matches[0]
            else:
                compatible_matches = backend_matches
                if requested_language:
                    language_matches = [
                        candidate
                        for candidate in backend_matches
                        if self._is_tts_language_supported(requested_language, candidate)
                    ]
                    compatible_matches = language_matches or backend_matches
                voice = self._prefer_speech_api_voices(compatible_matches)[0]
                self._log_legacy_voice_fallback(requested_voice, voice, backend_matches)

        if voice is None:
            # Only a program that has voices can be short of one for the language
            program_voices = [] if requested_voice or not requested_language else self._get_default_program_voices()
            if program_voices:
                advertised = sorted({language for voice in program_voices for language in voice.languages})
                _LOGGER.error(
                    "No TTS voice supports language %s in the default or selected program. Available languages: %s "
                    "(add it to WYOMING_LANGUAGES)",
                    requested_language,
                    advertised,
                )
            else:
                self._log_unsupported_voice(requested_voice)
            return None
        if not self._validate_tts_language(requested_language, voice):
            return None
        return voice

    def _validate_tts_language(self, language: str | None, voice: TtsVoice) -> bool:
        """Validate if a language is supported by a TTS voice.

        Returns True if supported. If no language is specified, also returns True.

        A named voice is held to its advertised languages on purpose, as it has been since the first release; only
        wyoming before 1.11 dropping the language of a named request let one through. Do not relax this to a warning.
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
        _LOGGER.debug("Handling synthesize request %s", synthesize)

        # IMPORTANT: Ignore standalone synthesize events when streaming synthesis is already active
        # This prevents duplicate audio synthesis when both streaming events (synthesize-start/chunk/stop)
        # and standalone synthesize events are used together
        if self._synthesis.active:
            _LOGGER.debug("Ignoring standalone synthesize event - streaming synthesis is already active")
            return True

        self._synthesis = _SynthesisState(plain=True)
        succeeded = False
        try:
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

            if self._splits_plain_synthesize(voice):
                chunks = self._chunk_text_for_streaming(
                    text, self._tts_streaming_min_words, self._tts_streaming_max_chars, requested_language
                )
                succeeded = await self._play_sentences(chunks, voice) and await self._finish_tts_audio_stream(voice)
            else:
                succeeded = await self._synthesize_non_streaming(text, voice)
            return succeeded

        except Exception as e:
            _LOGGER.exception("Error during synthesis: %s", e)
            return False
        finally:
            # Also when cancelled by a shutdown, which has then reported it already
            await self._end_synthesis(None if succeeded else TTS_FAILED_ERROR)

    async def _handle_synthesize_start(self, synthesize_start: SynthesizeStart) -> bool:
        """Handle start of streaming synthesis"""
        _LOGGER.debug("Handling synthesize-start event: %s", synthesize_start)

        if self._synthesis.active:
            _LOGGER.warning("Received synthesize-start during an active synthesis, which is ended without its stop")
        # A synthesis the client never stopped must not leave its audio stream open under the new one
        await self._end_synthesis()

        requested_voice = synthesize_start.voice.name if synthesize_start.voice else None
        requested_language = synthesize_start.voice.language if synthesize_start.voice else None

        # Resolved here, also when no voice is named, so sentences are spoken as they arrive. They reuse it
        # instead of resolving, and warning about, the name again
        voice = self._validate_tts_voice_and_language(requested_voice, requested_language)
        if not voice:
            # Without it a client that reads until the connection closes takes this for a synthesis without audio
            await self._write_event_bounded(Error(text=TTS_FAILED_ERROR).event())
            return False

        self._synthesis = _SynthesisState(
            active=True,
            voice=voice,
            # Decided here for the whole synthesis: a program selected midway must not make the chunks and
            # the stop disagree about whether the sentences were already spoken
            streamed=self._is_tts_voice_streaming(voice.name),
            language=requested_language,
            ssml_transformer=SsmlTextTransformer() if self._is_ssml_format(synthesize_start.text_format) else None,
        )
        return True

    async def _handle_synthesize_chunk(self, synthesize_chunk: SynthesizeChunk) -> bool:
        """Handle text chunk during streaming synthesis with incremental sentence detection"""
        if not self._synthesis.active:
            _LOGGER.warning("Received synthesize-chunk without active synthesis")
            return False

        chunk_text = synthesize_chunk.text if synthesize_chunk.text else ""
        _LOGGER.debug("Received synthesis chunk: '%s' (length: %d)", truncate_for_log(chunk_text, 50), len(chunk_text))

        if self._synthesis.ssml_transformer is not None:
            chunk_text = self._synthesis.ssml_transformer.feed(chunk_text)

        # Keep the full-text buffer and sentence accumulator as one identical
        # projected stream. The transformer owns all SSML whitespace repair.
        self._synthesis.buffer.append(chunk_text)
        if not self._synthesis.streamed:
            # Synthesized as a whole at synthesize-stop, so there are no sentences to look for
            return True

        # Add to accumulator for sentence detection across chunks
        self._synthesis.text_accumulator += chunk_text

        segmenter = _get_boundary_detector(self._synthesis.language)

        # Segment the entire accumulated text. preserve_whitespace=True keeps every
        # character (the retained last segment is appended to by later events), so no
        # text is lost at sentence boundaries even when yasbd redistributes the space.
        sentences: list[str] = list(segmenter.segment(self._synthesis.text_accumulator, preserve_whitespace=True))

        # Process complete sentences (all but the last one)
        ready_sentences: list[str] = []
        if len(sentences) > 1:
            unfinished_sentence = sentences.pop()
            # The same limits as for a text that is split as a whole. The whitespace is already in the sentences
            ready_sentences, open_chunk, open_words = self._merge_sentences(
                sentences, self._tts_streaming_min_words, self._tts_streaming_max_chars, joiner=""
            )
            min_words = self._tts_streaming_min_words
            if open_chunk and (not min_words or open_words >= min_words):
                # Spoken now rather than held for later text to fill it up to the maximum
                ready_sentences.append(open_chunk)
                open_chunk = ""
            # A chunk below the minimum waits for more text; synthesize-stop speaks whatever is left
            self._synthesis.text_accumulator = open_chunk + unfinished_sentence

        if ready_sentences:
            _LOGGER.info(
                "Detected %d ready sentences for immediate synthesis: %s",
                len(ready_sentences),
                [truncate_for_log(s, 30) for s in ready_sentences],
            )
            if not await self._process_ready_sentences(ready_sentences):
                return False
        else:
            _LOGGER.debug(
                "No complete sentences ready yet, accumulator has: '%s'",
                truncate_for_log(self._synthesis.text_accumulator),
            )

        return True

    async def _handle_synthesize_stop(self) -> bool:
        """Handle end of streaming synthesis"""
        state = self._synthesis
        if not state.active:
            _LOGGER.warning("Received synthesize-stop without active synthesis")
            return False

        state.active = False
        try:
            return await self._finish_synthesis(state)
        finally:
            await self._end_synthesis()

    async def _finish_synthesis(self, state: _SynthesisState) -> bool:
        """Speak what a streaming synthesis has left and end it with synthesize-stopped."""
        voice = state.voice
        if voice is None:
            # Not reachable through synthesize-start, which always resolves a voice
            _LOGGER.error("Streaming synthesis has no voice")
            return await self._abort_synthesis()

        if state.ssml_transformer is not None:
            flushed = state.ssml_transformer.finish()
            state.text_accumulator += flushed
            state.buffer.append(flushed)

        full_text = "".join(state.buffer)
        _LOGGER.debug("Streaming synthesis completed with text: %s", truncate_for_log(full_text))

        try:
            if state.streamed:
                # Earlier sentences were spoken as they arrived. What is left is final, complete or not
                if state.text_accumulator.strip():
                    _LOGGER.info("Processing final remaining text: '%s'", truncate_for_log(state.text_accumulator))
                    if not await self._process_ready_sentences([state.text_accumulator]):
                        return False
                succeeded = await self._finish_tts_audio_stream(voice)
            else:
                _LOGGER.debug("Using non-streaming synthesis for voice: %s", voice.name)
                succeeded = await self._synthesize_non_streaming(full_text, voice)

            if not succeeded:
                return await self._abort_synthesis()

            await self.write_event(SynthesizeStopped().event())
            _LOGGER.info("Successfully completed streaming synthesis, final timestamp: %.2f", state.timestamp)
            return True

        except Exception as e:
            _LOGGER.exception("Error during streaming synthesis: %s", e)
            return await self._abort_synthesis()

    async def _get_tts_audio_stream(
        self, text: str, voice: TtsVoiceModel, task_id: str | None = None
    ) -> TtsStreamResult:
        """
        Get TTS audio stream from OpenAI for a text chunk (parallel-safe).

        Audio is buffered until task_id matches _allow_streaming_task_id, which is when playback reaches this
        chunk. From then on what was buffered and the rest go to Wyoming as they arrive (minimal latency).
        A chunk that finishes before its turn is returned whole.

        Args:
            text (str): Text chunk to synthesize.
            voice (TtsVoiceModel): Voice to use for synthesis.
            task_id (str | None): Optional task identifier for streaming coordination.

        Returns:
            TtsStreamResult: Container with streaming status and optional buffered audio.
        """
        chunk_preview = truncate_for_log(text, 50)

        def is_playing() -> bool:
            """Check if playback has reached this chunk, so nothing else is writing audio."""
            return task_id is not None and task_id == self._allow_streaming_task_id

        try:
            if self._tts_client is None:
                raise TtsStreamError("TTS client is not configured", chunk_preview, voice.name)

            buffered: list[bytes] = []
            async with self._tts_limit.slot(), contextlib.aclosing(self._iter_tts_audio(text, voice)) as audio_stream:
                # Checked again with every chunk rather than once up front: a request that waited for its
                # turn, or is still running when playback reaches it, is not buffered whole first
                if not is_playing():
                    async for chunk in audio_stream:
                        buffered.append(chunk)
                        if is_playing():
                            break
                    else:
                        # Whether it holds anything to play is decided when it is played, by the rule
                        # `_play_tts_audio` applies to streamed audio too
                        _LOGGER.debug("Completed buffered synthesis for chunk: %s", chunk_preview)
                        return TtsStreamResult(streamed=False, audio=b"".join(buffered))

                async def buffered_then_live() -> AsyncGenerator[bytes, None]:
                    for chunk in buffered:
                        yield chunk
                    async for chunk in audio_stream:
                        yield chunk

                _LOGGER.debug("Streaming chunk directly (task %s): %s", task_id, chunk_preview)
                if not await self._play_tts_audio(
                    buffered_then_live(),
                    audio_format=self._get_tts_audio_format(voice),
                    text=text,
                    streamed=True,
                    sentence=True,
                ):
                    raise TtsStreamError("streaming its audio failed, as logged above", chunk_preview, voice.name)
                _LOGGER.debug("Completed direct streaming for chunk: %s", chunk_preview)
                return TtsStreamResult(streamed=True)

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
        speed = get_realtime_tts_speed(self._tts_speed, extra_body)
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
            "format": dict(REALTIME_AUDIO_FORMAT),
        }
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
            # A conversational model handed only punctuation or emoji may improvise a reply instead of
            # staying silent, so it is never sent; playback accepts no audio for such text
            return
        # Playback takes the audio only as fast as the Wyoming client does. With the default receive queue the
        # socket is no longer read behind 16 unread frames, which holds back keepalive pongs and the reply to
        # a close as well, so the queue is unbounded: at most it holds one response, as a buffered sentence does
        connection_manager = self._tts_client.realtime.connect(
            model=voice.model_name, websocket_connection_options={"max_queue": None}
        )
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
            # The closing handshake must not hold up the end of the audio stream or an abort. The concurrency
            # limit therefore counts requests in flight: this socket may still be closing when the next opens
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
        if not requested or not spoken_text.strip() or _normalize_for_comparison(spoken_text) == requested:
            return
        # A transcript spells out numbers and symbols ("72%" as "seventy-two percent"), which is not drift. Numbers
        # by category, not by numeric value: ideographs such as 一 and 十 have one and are part of ordinary words
        spelled_out = any(unicodedata.category(char).startswith("N") or _is_spoken_symbol(char) for char in text)
        log = _LOGGER.debug if spelled_out else _LOGGER.warning
        log(
            "Realtime TTS spoke text that differs from the request. Requested: %s | Spoken: %s",
            truncate_for_log(text),
            truncate_for_log(spoken_text),
        )

    async def _stream_audio_to_wyoming(self, audio_data: bytes, audio_format: TtsAudioFormat, text: str = "") -> bool:
        """
        Stream the buffered TTS response of a sentence to Wyoming, after the audio sent so far.

        Args:
            audio_data (bytes): Complete audio data to stream.
            audio_format (TtsAudioFormat): How the audio is framed, from `_get_tts_audio_format`.
            text (str): The text that was synthesized.

        Returns:
            bool: True if the sentence was played or skipped, False on error.
        """

        async def buffered_response() -> AsyncGenerator[bytes, None]:
            yield audio_data

        return await self._play_tts_audio(
            buffered_response(), audio_format=audio_format, text=text, streamed=False, sentence=True
        )

    async def _synthesize_non_streaming(self, text: str, voice: TtsVoiceModel) -> bool:
        """
        Synthesize text as one unsplit request.

        Args:
            text (str): Text to synthesize.
            voice (TtsVoiceModel): Voice to use for synthesis.

        Returns:
            bool: True on success, False on error.
        """
        if not text.strip():
            # Nothing to send to the backend: an empty stream, as for text with nothing to say
            _LOGGER.warning("No text to synthesize")
            return await self._finish_tts_audio_stream(voice)

        if not await self._create_synthesis_task(self._stream_tts_audio(voice, text), name="unsplit_synthesis"):
            return False
        if self._synthesis.failed:
            # stop() ended it as the request finished and told the client; an audio stop would mark it complete
            return False

        # Send audio stop after streaming completes
        await self._write_tts_audio_stop()
        _LOGGER.info("Successfully synthesized: %s", truncate_for_log(text))
        return True

    async def _write_tts_audio_start(self, rate: int, width: int, channels: int) -> None:
        """Send the audio start for the selected format and mark the stream as open."""
        # Marked before the write, which waits for the client to take it: stop() meanwhile must see the stream
        self._synthesis.audio_started = True
        await self._write_event_or_fail(AudioStart(rate=rate, width=width, channels=channels).event())

    async def _write_tts_audio_stop(self) -> None:
        """Send the audio stop at the end of the audio sent so far and mark the stream as closed."""
        # Marked before the write, which waits for the client to take it, so stop() meanwhile does not close the
        # stream a second time
        self._synthesis.audio_started = False
        await self.write_event(AudioStop(timestamp=int(self._synthesis.timestamp)).event())

    async def _finish_tts_audio_stream(self, voice: TtsVoiceModel) -> bool:
        """Send the audio stop of a completed synthesis, opening the stream first if nothing had audio."""
        if self._synthesis.failed:
            # stop() ended it after its last request finished and told the client, so nothing more is sent
            return False
        if not self._synthesis.audio_started:
            if self._synthesis.skipped_speakable_sentence:
                _LOGGER.error("TTS synthesis returned no audio for any speakable sentence")
                return False
            # Clients build their output from audio-start, so a synthesis with nothing to say still ends as
            # an empty stream rather than without one
            await self._write_tts_audio_start(
                self._get_tts_audio_format(voice).rate, DEFAULT_AUDIO_WIDTH, DEFAULT_AUDIO_CHANNELS
            )
        await self._write_tts_audio_stop()
        return True

    async def _write_tts_audio_chunk(self, audio_data: bytes, framer: WavFramer) -> None:
        """Send PCM in a framer's format and move the stream's timestamp past it; empty audio sends nothing."""
        if not audio_data:
            return
        timestamp = self._synthesis.timestamp
        # Kept current chunk by chunk, so a sentence that fails or is cancelled midway is stopped at what was sent;
        # moved before the write, which waits for the client, so a stop() meanwhile counts this chunk too
        self._synthesis.timestamp = self._advance_audio_timestamp(
            timestamp,
            audio_data=audio_data,
            audio_rate=framer.rate,
            audio_width=framer.width,
            audio_channels=framer.channels,
        )
        await self._write_event_or_fail(
            AudioChunk(
                audio=audio_data,
                rate=framer.rate,
                width=framer.width,
                channels=framer.channels,
                timestamp=int(timestamp),
            ).event()
        )

    async def _stream_tts_audio(self, voice: TtsVoiceModel, text: str) -> bool:
        """
        Synthesize text as one whole request and stream its audio to Wyoming as it arrives.

        Args:
            voice (TtsVoiceModel): Voice to use for synthesis.
            text (str): Text to synthesize.

        Returns:
            bool: True on success, False on error.
        """
        if self._tts_client is None:
            _LOGGER.error("No TTS client configured for synthesis")
            return False

        async with self._tts_limit.slot():
            return await self._play_tts_audio(
                self._iter_tts_audio(text, voice),
                audio_format=self._get_tts_audio_format(voice),
                text=text,
                streamed=True,
                sentence=False,
            )

    async def _play_tts_audio(
        self,
        chunks: AsyncGenerator[bytes, None],
        *,
        audio_format: TtsAudioFormat,
        text: str,
        streamed: bool,
        sentence: bool,
    ) -> bool:
        """
        Send the audio of one TTS response to Wyoming, as it arrives or from a buffered response.
        It continues the audio stream of the synthesis: AudioStart is sent first if the stream is not open yet,
        and the audio is timestamped after what was sent so far.

        Args:
            chunks (AsyncGenerator[bytes, None]): The response bytes; closed when playback ends.
            audio_format (TtsAudioFormat): How the bytes are framed, from `_get_tts_audio_format`.
            text (str): The text that was synthesized.
            streamed (bool): Whether the bytes arrive from the backend right now, not from a buffered response.
            sentence (bool): Whether this is one sentence of a synthesis rather than a whole request. A sentence
                without audio is skipped and leaves the stream unopened for the next one, which may have
                another audio format; a whole request still opens the stream when there is nothing to say
                and fails without audio for speakable text.

        Returns:
            bool: True if the audio was played or skipped, False on error. A stream left open by a failure
                or a cancellation is closed by `_stop_sentence_playback`, not here.
        """
        try:
            framer = WavFramer(audio_format)
            wrote_audio = False
            # A backend that answers without a WAV header does so for every sentence; the one that streams
            # reports it, the buffered ones behind it do not repeat the warning
            log_missing_header = _LOGGER.warning if streamed else _LOGGER.debug

            async def write_audio(audio_data: bytes) -> None:
                """Send the audio start once the format is known, then the PCM."""
                nonlocal wrote_audio
                if not self._synthesis.audio_started:
                    await self._write_tts_audio_start(framer.rate, framer.width, framer.channels)
                if not audio_data:
                    return
                await self._write_tts_audio_chunk(audio_data, framer)
                wrote_audio = True

            async with contextlib.aclosing(chunks) as audio_stream:
                async for chunk in audio_stream:
                    awaiting_header = framer.awaiting_header
                    audio_data = framer.feed(chunk)
                    if awaiting_header and framer.header_missing:
                        log_missing_header(
                            "Could not parse WAV header after buffering %d bytes, falling back to raw PCM",
                            len(audio_data),
                        )
                    # A parsed header alone only opens the stream for a caller that wants it opened when
                    # empty; a sentence that turns out to have no samples leaves it to the next one
                    if audio_data or (not sentence and not framer.awaiting_header):
                        await write_audio(audio_data)

            if held_audio := framer.finish():
                log_missing_header(
                    "TTS response ended before a complete WAV header was available, falling back to raw PCM"
                )
                await write_audio(held_audio)

            if framer.missing_bytes:
                _LOGGER.warning("TTS WAV response ended with %d declared PCM bytes missing", framer.missing_bytes)

            if wrote_audio:
                return True

            speakable = _has_speakable_content(text)
            if framer.missing_bytes or (speakable and not sentence):
                _LOGGER.error("TTS backend returned no audio for: %s", truncate_for_log(text, 50))
                return False

            if speakable:
                _LOGGER.warning("Skipping sentence without audio: %s", truncate_for_log(text, 50))
                self._synthesis.skipped_speakable_sentence = True
            else:
                _LOGGER.debug("No audio for text with nothing to speak: %s", truncate_for_log(text, 50))
            if not sentence:
                await write_audio(b"")
            return True

        except Exception as e:
            _LOGGER.exception("Error streaming TTS audio: %s", e)
            return False

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
