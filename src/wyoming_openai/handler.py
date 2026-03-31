import asyncio
import io
import logging
import wave
from dataclasses import dataclass
from typing import cast

import pysbd
from openai import AsyncStream, omit
from openai.types.audio.transcription_create_response import TranscriptionCreateResponse
from wyoming.asr import Transcribe, Transcript, TranscriptChunk, TranscriptStart, TranscriptStop
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, Describe, Info, TtsVoice
from wyoming.server import AsyncEventHandler
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeStopped, SynthesizeVoice

from .compatibility import CustomAsyncOpenAI, OpenAIBackend, TtsVoiceModel
from .utilities import NamedBytesIO, get_extra_body_boolean_field, validate_stt_extra_body, validate_tts_extra_body

_LOGGER = logging.getLogger(__name__)

DEFAULT_AUDIO_WIDTH = 2
DEFAULT_AUDIO_CHANNELS = 1
DEFAULT_ASR_AUDIO_RATE = 16000
TTS_AUDIO_RATE = 24000
TTS_CHUNK_SIZE = 2048
TTS_CONCURRENT_REQUESTS = 3


def _truncate(text: str, n: int = 100) -> str:
    return text[:n] + "..." if len(text) > n else text


@dataclass(frozen=True)
class TtsStreamResult:
    streamed: bool
    audio: bytes | None = None


class TtsStreamError(Exception):
    def __init__(self, message: str, chunk_preview: str, voice: str):
        super().__init__(message)
        self.chunk_preview = chunk_preview
        self.voice = voice


class OpenAIEventHandler(AsyncEventHandler):
    def __init__(
        self, *args,
        info: Info,
        stt_client: CustomAsyncOpenAI,
        tts_client: CustomAsyncOpenAI,
        stt_temperature: float | None = None,
        stt_prompt: str | None = None,
        stt_extra_body: dict[str, object] | None = None,
        tts_speed: float | None = None,
        tts_instructions: str | None = None,
        tts_extra_body: dict[str, object] | None = None,
        tts_streaming_min_words: int | None = None,
        tts_streaming_max_chars: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._wyoming_info = info
        self._stt_client = stt_client
        self._stt_temperature = stt_temperature
        self._stt_prompt = stt_prompt
        self._stt_extra_body = dict(stt_extra_body) if stt_extra_body else None
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

        # ASR state
        self._wav_buffer: NamedBytesIO | None = None
        self._wav_write_buffer: wave.Wave_write | None = None
        self._is_recording = False
        self._current_asr_model: AsrModel | None = None
        self._current_language: str | None = None

        # TTS streaming state
        self._synthesis_buffer: list[str] = []
        self._synthesis_voice: SynthesizeVoice | None = None
        self._is_synthesizing = False
        self._text_accumulator = ""
        self._ready_chunks: list[str] = []
        self._pysbd_segmenters: dict[str, pysbd.Segmenter] = {}
        self._audio_started = False
        self._current_timestamp: float = 0
        self._tts_semaphore = asyncio.Semaphore(TTS_CONCURRENT_REQUESTS)
        self._allow_streaming_task_id: str | None = None

        # Event logging
        self._last_event_type: str | None = None
        self._event_counter = 0

    # --- Event dispatch ---

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            await self._handle_audio_chunk(AudioChunk.from_event(event))
            return True

        _LOGGER.debug("Incoming event: %s", event.type)

        if Transcribe.is_type(event.type):
            return await self._handle_transcribe(Transcribe.from_event(event))
        if AudioStart.is_type(event.type):
            d = event.data or {}
            await self._handle_audio_start(
                d.get("rate", DEFAULT_ASR_AUDIO_RATE),
                d.get("width", DEFAULT_AUDIO_WIDTH),
                d.get("channels", DEFAULT_AUDIO_CHANNELS),
            )
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

        _LOGGER.info("Ignoring unhandled event: %s", event.type)
        return True

    # --- ASR ---

    async def _handle_transcribe(self, transcribe: Transcribe) -> bool:
        model = self._get_asr_model(transcribe.name)
        self._current_asr_model = None
        self._current_language = None

        if not model:
            self._log_unsupported_asr_model(transcribe.name)
            return False
        if not self._is_asr_language_supported(transcribe.language, model):
            self._log_unsupported_asr_language(transcribe.name, transcribe.language)
            return False

        self._current_asr_model = model
        self._current_language = transcribe.language
        return True

    async def _handle_audio_start(self, rate: int, width: int, channels: int) -> None:
        self._is_recording = True
        self._wav_buffer = NamedBytesIO(name="recording.wav")
        self._wav_write_buffer = wave.open(self._wav_buffer, "wb")
        self._wav_write_buffer.setnchannels(channels)
        self._wav_write_buffer.setsampwidth(width)
        self._wav_write_buffer.setframerate(rate)
        _LOGGER.info("Recording started: %dHz %dch %dbps", rate, channels, width)

    async def _handle_audio_chunk(self, chunk: AudioChunk) -> None:
        if self._is_recording and chunk.audio and self._wav_write_buffer:
            self._wav_write_buffer.writeframes(chunk.audio)
        else:
            _LOGGER.warning("Problem handling audio chunk")

    async def _handle_audio_stop(self) -> None:
        if not self._is_recording or not self._wav_buffer:
            _LOGGER.warning("Audio stop without recording")
            return

        self._is_recording = False
        try:
            if self._wav_write_buffer:
                self._wav_write_buffer.close()
                self._wav_write_buffer = None
            self._wav_buffer.seek(0)

            if not self._current_asr_model:
                _LOGGER.warning("No ASR model set")
                return

            extra_body = self._get_stt_extra_body()
            use_streaming = get_extra_body_boolean_field(
                extra_body, field_name="stream",
                default=self._is_asr_model_streaming(self._current_asr_model.name),
                body_name="STT",
            )

            kwargs = {
                "file": self._wav_buffer,
                "model": self._current_asr_model.name,
                "language": self._current_language if self._current_language is not None else omit,
                "temperature": self._stt_temperature if self._stt_temperature is not None else omit,
                "prompt": self._stt_prompt if self._stt_prompt is not None else omit,
                "response_format": "json",
                "stream": use_streaming if use_streaming else omit,
            }
            if extra_body:
                kwargs["extra_body"] = extra_body

            transcription = await self._stt_client.audio.transcriptions.create(**kwargs)
            await self.write_event(TranscriptStart().event())

            if isinstance(transcription, AsyncStream):
                full_text = ""
                async for chunk in transcription:
                    if chunk.type == "transcript.text.delta" and chunk.delta:
                        full_text += chunk.delta
                        await self.write_event(TranscriptChunk(text=chunk.delta).event())
                if full_text:
                    _LOGGER.info("Transcribed (stream): %s", full_text)
                else:
                    _LOGGER.warning("Empty streaming transcription. Check STT_STREAMING_MODELS config.")
                await self.write_event(Transcript(text=full_text).event())
            elif isinstance(transcription, TranscriptionCreateResponse):
                if transcription.text:
                    _LOGGER.info("Transcribed: %s", _truncate(transcription.text))
                else:
                    _LOGGER.warning("Empty transcription result")
                await self.write_event(Transcript(text=transcription.text).event())
            else:
                _LOGGER.error("Unexpected transcription type: %s", type(transcription))

            await self.write_event(TranscriptStop().event())
        except Exception as e:
            _LOGGER.exception("Transcription error: %s", e)
        finally:
            if self._wav_buffer:
                self._wav_buffer.close()
                self._wav_buffer = None

    # --- ASR helpers ---

    def _get_asr_model(self, name: str | None = None) -> AsrModel | None:
        for prog in self._wyoming_info.asr:
            for model in prog.models:
                if model.name == name or not name:
                    return model
        return None

    def _has_asr_models(self) -> bool:
        return any(getattr(p, "models", None) for p in self._wyoming_info.asr)

    def _has_tts_voices(self) -> bool:
        return any(getattr(p, "voices", None) for p in self._wyoming_info.tts)

    def _get_stt_extra_body(self) -> dict[str, object] | None:
        extra = dict(self._stt_extra_body or {})
        if getattr(self._stt_client, "backend", None) == OpenAIBackend.SPEACHES:
            extra.setdefault("vad_filter", False)
        return extra or None

    def _get_tts_extra_body(self) -> dict[str, object] | None:
        return dict(self._tts_extra_body) if self._tts_extra_body else None

    def _is_asr_model_streaming(self, name: str) -> bool:
        for prog in self._wyoming_info.asr:
            for m in prog.models:
                if m.name == name:
                    return prog.supports_transcript_streaming
        return False

    def _is_tts_voice_streaming(self, name: str) -> bool:
        for prog in self._wyoming_info.tts:
            for v in prog.voices:
                if v.name == name:
                    return getattr(prog, "supports_synthesize_streaming", False)
        return False

    def _is_asr_language_supported(self, language: str | None, model: AsrModel) -> bool:
        return not language or not model.languages or language in model.languages

    def _log_unsupported_asr_model(self, name: str | None):
        _LOGGER.warning("Unsupported ASR model: %s", name or "(none)")

    def _log_unsupported_asr_language(self, model: str | None, lang: str | None):
        _LOGGER.error("Unsupported ASR model %s for language %s", model, lang)

    # --- TTS helpers ---

    def _get_voice(self, name: str | None = None) -> TtsVoiceModel | None:
        for prog in self._wyoming_info.tts:
            for voice in prog.voices:
                if not name or voice.name == name:
                    return cast(TtsVoiceModel, voice)
        return None

    def _is_tts_language_supported(self, lang: str, voice: TtsVoice) -> bool:
        return not voice.languages or lang in voice.languages

    def _validate_tts_voice_and_language(
        self, voice_name: str | None, language: str | None
    ) -> TtsVoiceModel | None:
        voice = self._get_voice(voice_name)
        if not voice:
            self._log_unsupported_voice(voice_name)
            return None
        if not self._validate_tts_language(language, voice):
            return None
        return voice

    def _validate_tts_language(self, language: str | None, voice: TtsVoice) -> bool:
        if language and not self._is_tts_language_supported(language, voice):
            _LOGGER.error("Language %s not supported for voice %s. Available: %s", language, voice.name, voice.languages)
            return False
        return True

    def _log_unsupported_voice(self, name: str | None) -> None:
        if name:
            available = [v.name for p in self._wyoming_info.tts for v in p.voices]
            _LOGGER.error("Voice %s not supported. Available: %s", name, available)
        else:
            _LOGGER.error("No TTS voices specified")

    def _build_tts_kwargs(self, voice: TtsVoiceModel, text: str) -> dict:
        kwargs = {
            "model": voice.model_name,
            "voice": voice.name,
            "input": text,
            "response_format": "wav",
            "speed": self._tts_speed if self._tts_speed is not None else omit,
            "instructions": self._tts_instructions if self._tts_instructions is not None else omit,
        }
        if extra := self._get_tts_extra_body():
            kwargs["extra_body"] = extra
        return kwargs

    # --- WAV parsing ---

    def _parse_wav_header(self, data: bytes) -> tuple[int, int, int, int] | None:
        """Returns (sample_rate, channels, sample_width, data_offset) or None."""
        try:
            buf = io.BytesIO(data)
            with wave.open(buf, "rb") as wf:
                return wf.getframerate(), wf.getnchannels(), wf.getsampwidth(), buf.tell()
        except Exception:
            return None

    # --- TTS: standalone synthesize ---

    async def _handle_synthesize(self, synthesize: Synthesize) -> bool:
        try:
            if self._is_synthesizing:
                _LOGGER.debug("Ignoring standalone synthesize - streaming active")
                return True

            voice_name = synthesize.voice.name if synthesize.voice else None
            voice_lang = synthesize.voice.language if synthesize.voice else None
            voice = self._validate_tts_voice_and_language(voice_name, voice_lang)
            if not voice:
                return False

            ts = await self._stream_tts_audio(voice, synthesize.text, send_audio_start=True)
            if ts is not None:
                await self.write_event(AudioStop(timestamp=int(ts)).event())
                _LOGGER.info("Synthesized: %s", _truncate(synthesize.text))
                return True
            return False
        except Exception as e:
            _LOGGER.exception("Synthesis error: %s", e)
            return False

    # --- TTS: streaming synthesize ---

    async def _handle_synthesize_start(self, ev: SynthesizeStart) -> bool:
        _LOGGER.debug("synthesize-start: %s", ev)
        self._synthesis_buffer = []
        self._is_synthesizing = True
        self._text_accumulator = ""
        self._ready_chunks = []
        self._pysbd_segmenters.clear()
        self._audio_started = False
        self._current_timestamp = 0

        if ev.voice:
            self._synthesis_voice = ev.voice
            voice = self._validate_tts_voice_and_language(ev.voice.name, ev.voice.language)
            if not voice:
                self._is_synthesizing = False
                return False
        else:
            self._synthesis_voice = None
        return True

    async def _handle_synthesize_chunk(self, ev: SynthesizeChunk) -> bool:
        if not self._is_synthesizing:
            _LOGGER.warning("synthesize-chunk without active synthesis")
            return False

        text = ev.text or ""
        self._synthesis_buffer.append(ev.text)
        self._text_accumulator += text

        lang = self._synthesis_voice.language if self._synthesis_voice else None
        pysbd_lang = self._get_pysbd_language(lang)
        if pysbd_lang not in self._pysbd_segmenters:
            self._pysbd_segmenters[pysbd_lang] = pysbd.Segmenter(language=pysbd_lang, clean=True)

        sentences = list(self._pysbd_segmenters[pysbd_lang].segment(self._text_accumulator))
        if len(sentences) > 1:
            ready = sentences[:-1]
            self._text_accumulator = sentences[-1]
            _LOGGER.info("Ready sentences: %s", [_truncate(s, 30) for s in ready])
            if not await self._process_ready_sentences(ready, lang):
                return False
        return True

    async def _handle_synthesize_stop(self) -> bool:
        if not self._is_synthesizing:
            _LOGGER.warning("synthesize-stop without active synthesis")
            return False

        self._is_synthesizing = False

        # Process remaining text
        if self._text_accumulator.strip():
            lang = self._synthesis_voice.language if self._synthesis_voice else None
            if not await self._process_ready_sentences([self._text_accumulator], lang):
                return False

        full_text = "".join(self._synthesis_buffer)
        voice_info = self._synthesis_voice
        self._synthesis_buffer = []
        self._synthesis_voice = None
        self._text_accumulator = ""
        self._ready_chunks = []
        self._pysbd_segmenters.clear()

        # If incremental synthesis happened, finish up
        if self._audio_started:
            await self.write_event(AudioStop(timestamp=int(self._current_timestamp)).event())
            await self.write_event(SynthesizeStopped().event())
            _LOGGER.info("Incremental synthesis done, ts=%.2f", self._current_timestamp)
            self._audio_started = False
            self._current_timestamp = 0
            self._pysbd_segmenters.clear()
            return True

        if not full_text.strip():
            _LOGGER.warning("No text to synthesize")
            await self.write_event(SynthesizeStopped().event())
            return True

        try:
            voice_name = voice_info.name if voice_info else None
            voice_lang = voice_info.language if voice_info else None
            voice = self._validate_tts_voice_and_language(voice_name, voice_lang)
            if not voice:
                await self.write_event(SynthesizeStopped().event())
                return False

            if self._is_tts_voice_streaming(voice.name):
                await self._synthesize_chunked(full_text, voice, voice_lang)
            else:
                if not await self._synthesize_non_streaming(full_text, voice):
                    await self.write_event(SynthesizeStopped().event())
                    return False

            await self.write_event(SynthesizeStopped().event())
            return True
        except Exception as e:
            _LOGGER.exception("Streaming synthesis error: %s", e)
            await self.write_event(SynthesizeStopped().event())
            return False

    # --- TTS: sentence processing ---

    def _get_pysbd_language(self, language: str | None) -> str:
        if not language:
            return "en"
        base = language[:2].lower() if len(language) >= 2 else "en"
        try:
            pysbd.Segmenter(language=base)
            return base
        except (ValueError, KeyError):
            _LOGGER.warning("Language '%s' not supported by pysbd, using English", base)
            return "en"

    def _chunk_text_for_streaming(
        self, text: str, min_words: int | None = None, max_chars: int | None = None, language: str | None = None
    ) -> list[str]:
        if not text.strip():
            return []
        segmenter = pysbd.Segmenter(language=self._get_pysbd_language(language), clean=True)
        sentences = segmenter.segment(text)

        chunks: list[str] = []
        current = ""
        for s in sentences:
            potential = f"{current} {s}" if current else s
            if max_chars and len(potential) > max_chars and current:
                if not min_words or len(current.split()) >= min_words:
                    chunks.append(current.strip())
                current = s
            elif not max_chars and not min_words:
                if current:
                    chunks.append(current.strip())
                current = s
            else:
                current = potential
        if current and (not min_words or len(current.split()) >= min_words):
            chunks.append(current.strip())
        return chunks or [text]

    def _meets_min_criteria(self, text: str, min_words: int) -> bool:
        return len(text.split()) >= min_words

    async def _process_ready_sentences(self, sentences: list[str], language: str | None = None) -> bool:
        if not sentences or not self._synthesis_voice:
            return True
        try:
            voice = self._validate_tts_voice_and_language(
                self._synthesis_voice.name, self._synthesis_voice.language
            )
            if not voice:
                return await self._abort_synthesis()

            if not self._is_tts_voice_streaming(voice.name):
                return True

            valid = [s for s in sentences if s.strip()]
            if not valid:
                return True

            _LOGGER.info("Concurrent synthesis for %d sentences", len(valid))
            tasks = [
                (f"sentence_{i}", asyncio.create_task(
                    self._get_tts_audio_stream(s, voice, task_id=f"sentence_{i}"),
                    name=f"inc_sentence_{i}",
                ))
                for i, s in enumerate(valid)
            ]

            for i, (tid, task) in enumerate(tasks):
                self._allow_streaming_task_id = tid
                try:
                    result = await task
                except (TtsStreamError, Exception) as err:
                    _LOGGER.exception("Failed sentence %d: %s", i + 1, err)
                    return await self._abort_synthesis()
                finally:
                    self._allow_streaming_task_id = None

                if result.streamed:
                    continue

                if not result.audio:
                    _LOGGER.error("No audio for sentence %d", i + 1)
                    return await self._abort_synthesis()

                ts = await self._stream_audio_to_wyoming(
                    result.audio, is_first_chunk=not self._audio_started,
                    start_timestamp=self._current_timestamp,
                )
                if ts is None:
                    return await self._abort_synthesis()
                self._current_timestamp = ts
                self._audio_started = True

            return True
        except Exception as e:
            _LOGGER.exception("Error processing sentences: %s", e)
            return await self._abort_synthesis()

    async def _synthesize_chunked(self, text: str, voice: TtsVoiceModel, language: str | None) -> None:
        """Chunked concurrent TTS synthesis (fallback path for streaming voices)."""
        chunks = self._chunk_text_for_streaming(
            text, self._tts_streaming_min_words, self._tts_streaming_max_chars, language
        )
        tasks = [
            (f"fallback_chunk_{i}", asyncio.create_task(
                self._get_tts_audio_stream(c, voice, task_id=f"fallback_chunk_{i}"),
                name=f"chunk_{i}",
            ))
            for i, c in enumerate(chunks)
        ]

        total_ts: float = 0
        for i, (tid, task) in enumerate(tasks):
            self._allow_streaming_task_id = tid
            try:
                result = await task
            except (TtsStreamError, Exception) as err:
                _LOGGER.exception("Failed chunk %d: %s", i + 1, err)
                await self._abort_synthesis()
                return
            finally:
                self._allow_streaming_task_id = None

            if result.streamed:
                total_ts = self._current_timestamp
                continue

            if not result.audio:
                _LOGGER.error("No audio for chunk %d", i + 1)
                await self._abort_synthesis()
                return

            ts = await self._stream_audio_to_wyoming(
                result.audio, is_first_chunk=(i == 0), start_timestamp=total_ts,
            )
            if ts is None:
                await self._abort_synthesis()
                return
            total_ts = ts

        await self.write_event(AudioStop(timestamp=int(total_ts)).event())
        _LOGGER.info("Chunked synthesis done: %s", _truncate(text))

    # --- TTS: audio streaming ---

    async def _get_tts_audio_stream(
        self, text: str, voice: TtsVoiceModel, task_id: str | None = None
    ) -> TtsStreamResult:
        preview = _truncate(text, 50)
        try:
            if task_id and task_id == self._allow_streaming_task_id:
                ts = await self._stream_tts_audio_incremental(text, voice)
                if ts is None:
                    raise TtsStreamError("No audio while streaming", preview, voice.name)
                return TtsStreamResult(streamed=True)

            # Buffer mode
            chunks: list[bytes] = []
            async with self._tts_semaphore:
                kwargs = self._build_tts_kwargs(voice, text)
                async with self._tts_client.audio.speech.with_streaming_response.create(**kwargs) as resp:
                    async for chunk in resp.iter_bytes(chunk_size=TTS_CHUNK_SIZE):
                        chunks.append(chunk)

            audio = b"".join(chunks)
            if not audio:
                raise TtsStreamError("Empty audio response", preview, voice.name)
            return TtsStreamResult(streamed=False, audio=audio)
        except TtsStreamError:
            raise
        except Exception as exc:
            raise TtsStreamError("TTS audio error", preview, voice.name) from exc

    async def _stream_tts_audio_incremental(self, text: str, voice: TtsVoiceModel) -> float | None:
        ts = await self._stream_tts_audio(
            voice, text, send_audio_start=not self._audio_started,
            start_timestamp=self._current_timestamp,
        )
        if ts is not None:
            self._current_timestamp = ts
            self._audio_started = True
        return ts

    async def _stream_tts_audio(
        self, voice: TtsVoiceModel, text: str, send_audio_start: bool = True, start_timestamp: float = 0,
    ) -> float | None:
        try:
            first_chunk = None
            rate, width, channels = TTS_AUDIO_RATE, DEFAULT_AUDIO_WIDTH, DEFAULT_AUDIO_CHANNELS
            ts = start_timestamp

            async with self._tts_semaphore:
                kwargs = self._build_tts_kwargs(voice, text)
                async with self._tts_client.audio.speech.with_streaming_response.create(**kwargs) as resp:
                    async for chunk in resp.iter_bytes(chunk_size=TTS_CHUNK_SIZE):
                        if first_chunk is None:
                            first_chunk = chunk
                            wav = self._parse_wav_header(chunk)
                            if wav:
                                rate, channels, width, offset = wav
                                audio_data = chunk[offset:]
                            else:
                                audio_data = chunk
                            if send_audio_start:
                                await self.write_event(AudioStart(rate=rate, width=width, channels=channels).event())
                                send_audio_start = False
                        else:
                            audio_data = chunk

                        if audio_data:
                            await self.write_event(AudioChunk(
                                audio=audio_data, rate=rate, width=width,
                                channels=channels, timestamp=int(ts),
                            ).event())
                            ts += (len(audio_data) // width / rate) * 1000

            return ts
        except Exception as e:
            _LOGGER.exception("TTS streaming error: %s", e)
            return None

    async def _stream_audio_to_wyoming(
        self, audio_data: bytes, is_first_chunk: bool, start_timestamp: float,
    ) -> float | None:
        try:
            rate, width, channels = TTS_AUDIO_RATE, DEFAULT_AUDIO_WIDTH, DEFAULT_AUDIO_CHANNELS
            ts = start_timestamp

            wav = self._parse_wav_header(audio_data)
            if wav:
                rate, channels, width, offset = wav
                audio_data = audio_data[offset:]

            if is_first_chunk:
                await self.write_event(AudioStart(rate=rate, width=width, channels=channels).event())
            if audio_data:
                await self.write_event(AudioChunk(
                    audio=audio_data, rate=rate, width=width,
                    channels=channels, timestamp=int(ts),
                ).event())
                ts += (len(audio_data) // width / rate) * 1000
            return ts
        except Exception as e:
            _LOGGER.exception("Error streaming to Wyoming: %s", e)
            return None

    async def _synthesize_non_streaming(self, text: str, voice: TtsVoiceModel) -> bool:
        ts = await self._stream_tts_audio(voice, text, send_audio_start=True)
        if ts is not None:
            await self.write_event(AudioStop(timestamp=int(ts)).event())
            _LOGGER.info("Synthesized (non-streaming): %s", _truncate(text))
            return True
        return False

    async def _abort_synthesis(self) -> bool:
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

    # --- Logging ---

    async def write_event(self, event: Event) -> None:
        if self._last_event_type != event.type:
            self._last_event_type = event.type
            self._event_counter = 1
        else:
            self._event_counter += 1

        if event.type == "audio-chunk":
            if self._event_counter <= 1:
                _LOGGER.debug("Outgoing: %s", event.type)
            elif self._event_counter == 2:
                _LOGGER.debug("Outgoing: %s (further chunks suppressed)", event.type)
        else:
            _LOGGER.debug("Outgoing: %s", event.type)

        await super().write_event(event)
