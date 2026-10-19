"""Async driver for Home Assistant's assist_pipeline/run websocket command."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from websockets.asyncio.client import connect as async_connect
from websockets.exceptions import ConnectionClosed

from .config import HaE2ESettings
from .ha_api import HaApiError, HaStack
from .helpers import WavData

_LOGGER = logging.getLogger(__name__)

AUDIO_CHUNK_BYTES = 3200  # 100 ms of 16 kHz mono 16-bit PCM
AUDIO_CHUNK_DELAY_SECONDS = 0.01
TTS_FIRST_AUDIO_BYTES = 64  # WAV header, then the first PCM samples


@dataclass
class TtsProgress:
    """Mutable progress of the TTS URL fetch, exposed to the failure lanes."""

    done: asyncio.Event = field(default_factory=asyncio.Event)
    data: bytearray = field(default_factory=bytearray)
    status_code: int | None = None
    first_byte_at: float | None = None
    first_audio_at: float | None = None
    error: BaseException | None = None
    http_body: bytes = b""

    @property
    def bytes_received(self) -> int:
        return len(self.data)


@dataclass
class AssistRunResult:
    """Everything collected from one assist_pipeline/run."""

    events: list[dict[str, Any]]
    run_start: dict[str, Any] | None
    tts_progress: TtsProgress | None
    run_started_at: float
    conversation_id: str

    @property
    def error_events(self) -> list[dict[str, Any]]:
        return [event for event in self.events if event.get("type") == "error"]

    def first_event(self, event_type: str) -> dict[str, Any] | None:
        return next((event for event in self.events if event.get("type") == event_type), None)

    def events_of_type(self, event_type: str) -> list[dict[str, Any]]:
        return [event for event in self.events if event.get("type") == event_type]


async def _send_audio(websocket: Any, handler_id: int, audio: WavData) -> None:
    """Stream PCM as binary frames, then send a handler-id-only stop frame."""

    prefix = bytes([handler_id])
    try:
        for start in range(0, len(audio.pcm), AUDIO_CHUNK_BYTES):
            chunk = audio.pcm[start : start + AUDIO_CHUNK_BYTES]
            await websocket.send(prefix + chunk)
            await asyncio.sleep(AUDIO_CHUNK_DELAY_SECONDS)
        await websocket.send(prefix)
    except ConnectionClosed:
        # The server closes the connection after an STT failure; the event
        # collector has already seen (or will see) the error event.
        return


async def _fetch_tts(client: httpx.AsyncClient, url: str, progress: TtsProgress) -> None:
    try:
        async with client.stream("GET", url) as response:
            progress.status_code = response.status_code
            if response.status_code >= 400:
                progress.http_body = await response.aread()
                response.raise_for_status()
            async for chunk in response.aiter_bytes():
                now = time.monotonic()
                if progress.first_byte_at is None:
                    progress.first_byte_at = now
                progress.data.extend(chunk)
                if progress.first_audio_at is None and len(progress.data) >= TTS_FIRST_AUDIO_BYTES:
                    progress.first_audio_at = now
    except BaseException as exc:  # noqa: BLE001 - surfaced to the caller, including cancellation
        progress.error = exc
    finally:
        progress.done.set()


async def _watch_tts_for_stop(
    progress: TtsProgress,
    stop_after_bytes: int,
    callback: Callable[[TtsProgress], Awaitable[None]],
) -> None:
    while progress.bytes_received < stop_after_bytes:
        if progress.done.is_set():
            return
        await asyncio.sleep(0.05)
    await callback(progress)


def _absolute_url(base_url: str, url: str) -> str:
    if url.startswith(("http://", "https://")):
        return url
    return f"{base_url.rstrip('/')}/{url.lstrip('/')}"


async def run_assist(
    settings: HaE2ESettings,
    stack: HaStack,
    pipeline_id: str,
    audio: WavData,
    *,
    start_stage: str = "stt",
    end_stage: str = "tts",
    conversation_id: str | None = None,
    event_timeout: float = 180.0,
    tts_timeout: float = 300.0,
    allow_tts_error: bool = False,
    stop_after_tts_bytes: int | None = None,
    on_tts_bytes: Callable[[TtsProgress], Awaitable[None]] | None = None,
) -> AssistRunResult:
    """Run one Assist pipeline turn and collect every event.

    The TTS URL is available in the run-start event, before the pipeline's tts
    stage finishes.  Fetching it immediately makes the measured time to first
    audio real, and lets the failure lane stop the proxy while audio is still
    streaming.
    """

    run_started_at = time.monotonic()
    events: list[dict[str, Any]] = []
    run_start: dict[str, Any] | None = None
    progress: TtsProgress | None = None
    fetch_task: asyncio.Task[None] | None = None
    audio_task: asyncio.Task[None] | None = None
    watcher_task: asyncio.Task[None] | None = None
    command_id = 1

    async with async_connect(
        settings.ws_url,
        max_size=None,
        open_timeout=10,
        ping_interval=20,
        ping_timeout=20,
        proxy=None,
    ) as websocket:
        greeting = json.loads(await websocket.recv())
        if greeting.get("type") != "auth_required":
            raise HaApiError(f"unexpected websocket greeting: {greeting}")
        await websocket.send(json.dumps({"type": "auth", "access_token": stack.session.access_token}))
        auth = json.loads(await websocket.recv())
        if auth.get("type") != "auth_ok":
            raise HaApiError(f"websocket authentication failed: {auth}")

        request: dict[str, Any] = {
            "id": command_id,
            "type": "assist_pipeline/run",
            "start_stage": start_stage,
            "end_stage": end_stage,
            "input": {"sample_rate": 16000},
            "pipeline": pipeline_id,
        }
        if conversation_id is not None:
            request["conversation_id"] = conversation_id
        await websocket.send(json.dumps(request))

        async with httpx.AsyncClient(timeout=httpx.Timeout(tts_timeout, connect=10.0)) as http:
            try:
                while True:
                    try:
                        frame = await asyncio.wait_for(websocket.recv(), timeout=event_timeout)
                    except TimeoutError as exc:
                        raise TimeoutError(
                            f"timed out after {event_timeout}s waiting for a pipeline event; collected "
                            f"{[event.get('type') for event in events]}"
                        ) from exc
                    if isinstance(frame, bytes):
                        continue
                    message = json.loads(frame)
                    if message.get("id") != command_id:
                        continue
                    if message.get("type") == "error":
                        raise HaApiError(f"assist_pipeline/run failed: {message.get('error')}")
                    if message.get("type") == "result":
                        if not message.get("success", True):
                            raise HaApiError(f"assist_pipeline/run was rejected: {message}")
                        continue
                    if message.get("type") != "event":
                        continue

                    event = message.get("event") or {}
                    events.append(event)
                    event_type = event.get("type")
                    data = event.get("data") or {}

                    if event_type == "run-start":
                        run_start = data
                        runner_data = data.get("runner_data") or {}
                        handler_id = runner_data.get("stt_binary_handler_id")
                        if not isinstance(handler_id, int):
                            raise HaApiError(f"run-start did not include stt_binary_handler_id: {data}")
                        tts_output = data.get("tts_output") or {}
                        tts_url = tts_output.get("url")
                        if end_stage == "tts" and tts_url:
                            progress = TtsProgress()
                            fetch_task = asyncio.create_task(
                                _fetch_tts(http, _absolute_url(settings.base_url, str(tts_url)), progress),
                                name="ha-e2e-tts-fetch",
                            )
                            if stop_after_tts_bytes is not None and on_tts_bytes is not None:
                                watcher_task = asyncio.create_task(
                                    _watch_tts_for_stop(progress, stop_after_tts_bytes, on_tts_bytes),
                                    name="ha-e2e-tts-stop-watcher",
                                )
                        audio_task = asyncio.create_task(
                            _send_audio(websocket, handler_id, audio),
                            name="ha-e2e-audio-upload",
                        )
                    elif event_type == "run-end":
                        break

                if audio_task is not None:
                    if not audio_task.done():
                        audio_task.cancel()
                    await asyncio.gather(audio_task, return_exceptions=True)

                # HA emits run-end before the TTS URL has finished streaming.
                # Keep the stop watcher alive while the audio download completes,
                # then wait for any callback it started.
                if fetch_task is not None and progress is not None:
                    fetch_timed_out = False
                    try:
                        await asyncio.wait_for(fetch_task, timeout=tts_timeout)
                    except TimeoutError:
                        fetch_timed_out = True
                        fetch_task.cancel()
                        await asyncio.gather(fetch_task, return_exceptions=True)

                    if watcher_task is not None:
                        if not watcher_task.done():
                            try:
                                await asyncio.wait_for(watcher_task, timeout=60)
                            except TimeoutError:
                                watcher_task.cancel()
                                await asyncio.gather(watcher_task, return_exceptions=True)
                                raise TimeoutError("TTS stop watcher did not finish") from None
                        else:
                            await watcher_task

                    if fetch_timed_out:
                        raise TimeoutError(f"TTS audio was not fully fetched within {tts_timeout}s")
                    if progress.error is not None and not allow_tts_error:
                        raise progress.error
                elif watcher_task is not None:
                    watcher_task.cancel()
                    await asyncio.gather(watcher_task, return_exceptions=True)
            finally:
                for task in (audio_task, watcher_task, fetch_task):
                    if task is not None and not task.done():
                        task.cancel()
                await asyncio.gather(
                    *(task for task in (audio_task, watcher_task, fetch_task) if task is not None),
                    return_exceptions=True,
                )

    resolved_conversation_id = str((run_start or {}).get("conversation_id", conversation_id or ""))
    return AssistRunResult(
        events=events,
        run_start=run_start,
        tts_progress=progress,
        run_started_at=run_started_at,
        conversation_id=resolved_conversation_id,
    )


async def transcribe_with_speaches(settings: HaE2ESettings, wav_bytes: bytes) -> str:
    """Transcribe audio directly with Speaches, independently of the proxy."""

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        response = await client.post(
            f"{settings.speaches_url}/audio/transcriptions",
            files={"file": ("tts.wav", wav_bytes, "audio/wav")},
            data={"model": settings.stt_model, "language": "en", "response_format": "json"},
        )
    if not response.is_success:
        raise RuntimeError(
            f"Speaches transcription returned HTTP {response.status_code}: {response.text[:2000]}"
        )
    text = response.json().get("text")
    if not isinstance(text, str):
        raise RuntimeError(f"Speaches transcription returned no text: {response.text[:2000]}")
    _LOGGER.debug("Speaches transcript: %s", text)
    return text
