"""Smoke tests of the shared STT and TTS limits with a real Home Assistant controlling devices.

Opt in with the smoke stack up (tests/ha_e2e/smoke/docker-compose.yml), and for the OpenAI Realtime lane also
tests/ha_e2e/smoke/docker-compose.openai.yml with HA_SMOKE_OPENAI=1:

    pytest -m ha_smoke -v

Only an explicit `-m ha_smoke` selects the suite: it carries no `integration` marker, so neither the default run nor
`-m integration` or `-m ha_e2e`, which CI runs without this stack, picks it up.

Home Assistant's built-in conversation agent controls the demo integration's devices from speech that Kokoro
synthesizes. Each pipeline run opens its own Wyoming connections to the proxy, so turns run together are what
several satellites speaking at once look like to it. A summary of every turn is written to the run's artifacts.
"""

from __future__ import annotations

import array
import asyncio
import json
import logging
import os
import re
import secrets
import statistics
import subprocess
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from rapidfuzz import fuzz

from ha_e2e.config import PACKAGE_DIR, HaE2ESettings, load_settings
from ha_e2e.docker_control import (
    container_exit_code,
    container_logs,
    start_container,
    stop_container,
    wait_for_tcp,
)
from ha_e2e.generate_fixtures import TRAILING_SILENCE_SECONDS, _decode_wav, _resample, _to_mono, _wav_bytes
from ha_e2e.ha_api import (
    HaRestClient,
    HaSession,
    HaStack,
    HaWsClient,
    PipelineInfo,
    _ensure_pipeline,
    _first_voice,
    _load_entry_state,
    _wait_for_entities,
    ensure_custom_entry,
    ensure_session,
    ensure_wyoming_entry,
)
from ha_e2e.helpers import (
    WavData,
    assert_reply_match,
    assert_transcript_match,
    decode_audio_bytes,
    normalize_numbers,
    parse_wav_bytes,
)
from ha_e2e.runner import AssistRunResult, run_assist, run_assist_text, transcribe_with_speaches
from ha_e2e.scenarios import LONG_REPLY

LOGGER = logging.getLogger(__name__)

pytestmark = [pytest.mark.ha_smoke]

SMOKE_ENV = {
    "HA_E2E_BASE_URL": "http://127.0.0.1:8124",
    "HA_E2E_SPEACHES_URL": "http://127.0.0.1:8010/v1",
    "HA_E2E_STT_MODEL": "Systran/faster-distil-whisper-large-v3",
    "HA_E2E_STATE_DIR": str(PACKAGE_DIR / ".state" / "smoke"),
    "HA_E2E_ARTIFACT_DIR": str(PACKAGE_DIR / "artifacts"),
    "HA_E2E_WYOMING_PORT": "10300",
}
FIXTURE_TTS_MODEL = "speaches-ai/Kokoro-82M-v1.0-ONNX"
VOICES = ("af_heart", "am_michael", "bf_emma", "am_adam", "af_bella", "bm_george", "af_sarah", "bm_lewis")
BUILT_IN_AGENT = "conversation.home_assistant"
STATE_TIMEOUT_SECONDS = 10.0

STT_WAIT_LOG = "Waiting for a free STT request slot"
SLOW_CLIENT_LOG = "Client did not take its"
IDLE_LOG = "No audio for"
REALTIME_RECORDING_LOG = "Realtime recording started"
STT_CANCEL_LOG = "Cancelling the STT request in flight"
TRANSCRIPTION_POST_LOG = "POST http://speaches:8000/v1/audio/transcriptions"
# How much later than a log line is written a poll of the logs may see it, so how much earlier than that a reply that
# waited for it may arrive
LOG_POLL_SLACK = 1.5


@dataclass(frozen=True)
class Proxy:
    """One proxy of the smoke stack; its container name is also its host name on the stack's network."""

    name: str
    container: str
    host_port: int
    stt_limit: int
    tts_limit: int
    openai: bool = False


DEFAULT = Proxy("default", "ha_smoke_proxy", 10400, 3, 3)
LIMITED = Proxy("limited", "ha_smoke_proxy_limited", 10401, 1, 1)
UNLIMITED = Proxy("unlimited", "ha_smoke_proxy_unlimited", 10402, 0, 0)
OPENAI = Proxy("openai", "ha_smoke_proxy_openai", 10403, 3, 3, openai=True)
OPENAI_LIMITED = Proxy("openai-limited", "ha_smoke_proxy_openai_limited", 10404, 1, 1, openai=True)
OPENAI_ENABLED = os.environ.get("HA_SMOKE_OPENAI", "").strip() == "1"


def _proxies(*proxies: Proxy) -> list[Any]:
    skip_openai = pytest.mark.skip(reason="set HA_SMOKE_OPENAI=1 with docker-compose.openai.yml up")
    return [
        pytest.param(proxy, id=proxy.name, marks=skip_openai if proxy.openai and not OPENAI_ENABLED else ())
        for proxy in proxies
    ]


@dataclass(frozen=True)
class Command:
    """A spoken request, the state its entity is put in first, and the states that count as done."""

    phrase: str
    entity_id: str | None = None
    prepare: tuple[str, str] | None = None
    expected: tuple[str, ...] = ()
    brightness: tuple[int, int] | None = None


CONVERSATION = (
    Command("Turn off the kitchen lights.", "light.kitchen_lights", ("light", "turn_on"), ("off",)),
    Command("Turn on the kitchen lights.", "light.kitchen_lights", None, ("on",)),
    Command("Set the kitchen lights to 30 percent.", "light.kitchen_lights", None, ("on",), brightness=(65, 90)),
    Command("Open the garage door.", "cover.garage_door", ("cover", "close_cover"), ("open", "opening")),
    Command("Is the garage door open?"),
    Command("Close the garage door.", "cover.garage_door", None, ("closed", "closing")),
)
SATELLITES = (
    Command("Turn off the ceiling lights.", "light.ceiling_lights", ("light", "turn_on"), ("off",)),
    Command("Turn on the bed light.", "light.bed_light", ("light", "turn_off"), ("on",)),
    Command("Close the hall window.", "cover.hall_window", ("cover", "open_cover"), ("closed", "closing")),
)
LOAD = (
    *SATELLITES,
    Command("Turn off the kitchen lights.", "light.kitchen_lights", ("light", "turn_on"), ("off",)),
    Command("Open the kitchen window.", "cover.kitchen_window", ("cover", "close_cover"), ("open", "opening")),
    Command(
        "Close the living room window.", "cover.living_room_window", ("cover", "open_cover"), ("closed", "closing")
    ),
    Command("Open the garage door.", "cover.garage_door", ("cover", "close_cover"), ("open", "opening")),
    Command("What time is it?"),
)
DEMO_ENTITIES = tuple(sorted({command.entity_id for command in (*CONVERSATION, *LOAD) if command.entity_id}))
LONG_TEXTS = (
    LONG_REPLY[:700],
    "Here is a summary of the household calendar for the coming week. On Monday the recycling goes out before "
    "seven, and the plumber arrives between ten and noon to look at the upstairs sink. Tuesday is quiet apart "
    "from the school concert at six thirty, so dinner should be early. On Wednesday the car is booked for its "
    "service at eight, and someone has to pick it up after four. Thursday brings the grocery delivery in the "
    "morning and book club in the evening. Friday is pizza night, and the weekend is free so far.",
    "The garden needs a little attention before the frost arrives. Bring the potted herbs inside and keep them "
    "near the south window. Cut the roses back by a third, and cover the strawberry bed with straw. Drain and "
    "coil the hoses so they do not split, and shut off the outdoor tap from inside. Rake the leaves off the lawn "
    "so it does not smother, and add them to the compost with a few shovels of soil. Finally, clean and oil the "
    "tools before you hang them up, and they will be ready in spring.",
)

STUCK_TTS_CLIENT = r"""
import asyncio, socket, sys
from wyoming.event import async_write_event
from wyoming.tts import Synthesize

async def main(text, seconds):
    sock = socket.socket()
    # A small receive buffer, so the proxy's writes back up quickly once nothing is read
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    sock.connect(("127.0.0.1", 10300))
    sock.setblocking(False)
    _, writer = await asyncio.open_connection(sock=sock)
    await async_write_event(Synthesize(text=text).event(), writer)
    await asyncio.sleep(seconds)  # Never reads

asyncio.run(main(sys.argv[1], float(sys.argv[2])))
"""

IDLE_STT_CLIENT = r"""
import asyncio, sys
from wyoming.asr import Transcribe
from wyoming.audio import AudioStart
from wyoming.client import AsyncTcpClient

async def main(seconds):
    client = AsyncTcpClient("127.0.0.1", 10300)
    await client.connect()
    await client.write_event(Transcribe(language="en").event())
    await client.write_event(AudioStart(rate=16000, width=2, channels=1).event())
    await asyncio.sleep(seconds)  # Sends no audio
    await client.disconnect()

asyncio.run(main(float(sys.argv[1])))
"""


@dataclass
class TurnOutcome:
    """What one turn did and how long it took, for the summary."""

    test: str
    proxy: str
    phrase: str
    passed: bool = False
    stt_text: str = ""
    speech: str = ""
    response_type: str = ""
    tts_transcript: str = ""
    error: str = ""
    total_seconds: float | None = None
    stt_seconds: float | None = None
    first_audio_seconds: float | None = None
    audio_seconds: float | None = None
    stt_end_at: float | None = None  # time.monotonic() when Home Assistant reported the end of its STT stage
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Smoke:
    """The configured smoke stack."""

    settings: HaE2ESettings
    session: HaSession
    rest: HaRestClient
    stack: HaStack
    devices: dict[str, str]
    scripted: dict[str, str]
    run_dir: Path
    outcomes: list[TurnOutcome] = field(default_factory=list)
    _audio: dict[tuple[str, str], WavData] = field(default_factory=dict)

    def audio(self, phrase: str, voice: str) -> WavData:
        """Speech for a phrase, synthesized with Kokoro once per phrase and voice."""

        key = (phrase, voice)
        if key not in self._audio:
            response = httpx.post(
                f"{self.settings.speaches_url}/audio/speech",
                json={"model": FIXTURE_TTS_MODEL, "voice": voice, "input": phrase, "response_format": "wav"},
                timeout=120.0,
            )
            response.raise_for_status()
            rate, _, channels, samples = _decode_wav(response.content)
            mono = _resample(_to_mono(samples, channels), rate)
            mono.extend(array.array("h", bytes(2 * round(16000 * TRAILING_SILENCE_SECONDS))))
            self._audio[key] = parse_wav_bytes(_wav_bytes(mono))
        return self._audio[key]


@pytest.fixture(scope="session")
def smoke() -> Iterator[Smoke]:
    with pytest.MonkeyPatch.context() as patch:
        for name, value in SMOKE_ENV.items():
            patch.setenv(name, os.environ.get(name) or value)
        settings = load_settings()
    session = ensure_session(settings)
    rest = HaRestClient(settings, session.access_token)
    ws = HaWsClient(settings, session.access_token)
    try:
        rest.wait_running()
        state = _load_entry_state(settings)
        custom_entry = ensure_custom_entry(rest, settings, state)
        scripted_agent = _wait_for_entities(rest, ws, custom_entry, ("conversation",))["conversation"]
        proxies = [DEFAULT, LIMITED, UNLIMITED, *((OPENAI, OPENAI_LIMITED) if OPENAI_ENABLED else ())]
        devices: dict[str, str] = {}
        scripted: dict[str, str] = {}
        first: PipelineInfo | None = None
        for proxy in proxies:
            entry = ensure_wyoming_entry(rest, settings, proxy.container, state)
            entities = _wait_for_entities(rest, ws, entry, ("stt", "tts"))
            voice = _first_voice(ws, entities["tts"])
            engines = {
                "stt_engine": entities["stt"],
                "stt_language": "en",
                "tts_engine": entities["tts"],
                "tts_language": "en",
                "tts_voice": voice,
            }
            devices[proxy.name] = _ensure_pipeline(
                ws, f"Smoke {proxy.name}", {"conversation_engine": BUILT_IN_AGENT, **engines}
            )
            scripted[proxy.name] = _ensure_pipeline(
                ws, f"Smoke {proxy.name} scripted", {"conversation_engine": scripted_agent, **engines}
            )
            if first is None:
                first = PipelineInfo(devices[proxy.name], entities["stt"], entities["tts"], voice)
        _wait_for_demo_entities(rest)
        ws.command(
            "homeassistant/expose_entity",
            assistants=["conversation"],
            entity_ids=list(DEMO_ENTITIES),
            should_expose=True,
        )
        ha_version = str(rest.get_config().get("version", "unknown"))
    finally:
        ws.close()

    assert first is not None
    stack = HaStack(settings, session, rest, first, first, BUILT_IN_AGENT, "", ha_version)
    run_dir = settings.artifact_dir / f"smoke-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    configured = Smoke(settings, session, rest, stack, devices, scripted, run_dir)
    LOGGER.info("Home Assistant %s ready with pipelines %s", ha_version, devices)
    yield configured
    _write_summary(configured, ha_version)
    rest.close()


def _wait_for_demo_entities(rest: HaRestClient) -> None:
    deadline = time.monotonic() + 120
    missing: list[str] = list(DEMO_ENTITIES)
    while time.monotonic() < deadline:
        missing = [entity_id for entity_id in DEMO_ENTITIES if rest.get_state(entity_id) is None]
        if not missing:
            return
        time.sleep(1.0)
    states = [state["entity_id"] for state in rest.get_json("/api/states")]
    pytest.fail(f"demo entities {missing} never appeared; lights and covers: "
                f"{[s for s in states if s.startswith(('light.', 'cover.'))]}")


def _write_summary(smoke: Smoke, ha_version: str) -> None:
    payload = {"ha_version": ha_version, "turns": [asdict(outcome) for outcome in smoke.outcomes]}
    (smoke.run_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# Smoke run, Home Assistant {ha_version}",
        "",
        "| test | proxy | phrase | ok | total s | STT s | first audio s | notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for outcome in smoke.outcomes:
        lines.append(
            f"| {outcome.test} | {outcome.proxy} | {outcome.phrase[:48]} | {'yes' if outcome.passed else 'NO'} | "
            f"{_fmt(outcome.total_seconds)} | {_fmt(outcome.stt_seconds)} | {_fmt(outcome.first_audio_seconds)} | "
            f"{json.dumps(outcome.notes) if outcome.notes else (outcome.error[:80] or '')} |"
        )
    (smoke.run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def _timestamp(event: dict[str, Any] | None) -> float | None:
    if not event or not event.get("timestamp"):
        return None
    return datetime.fromisoformat(str(event["timestamp"])).timestamp()


def _stage_seconds(result: AssistRunResult, start: str, end: str) -> float | None:
    started, ended = _timestamp(result.first_event(start)), _timestamp(result.first_event(end))
    return None if started is None or ended is None else ended - started


def _first_audio_seconds(result: AssistRunResult) -> float | None:
    progress = result.tts_progress
    first = None if progress is None else (progress.first_audio_at or progress.first_byte_at)
    return None if first is None else first - result.run_started_at


def _log_count(proxy: Proxy, needle: str) -> int:
    return container_logs(proxy.container).count(needle)


def _prepare(smoke: Smoke, commands: tuple[Command, ...] | list[Command]) -> None:
    for command in commands:
        if command.prepare and command.entity_id:
            domain, service = command.prepare
            smoke.rest.call_service(domain, service, {"entity_id": command.entity_id})
    for command in commands:
        if command.prepare and command.entity_id:
            # The demo covers move for a few seconds; wait until each has arrived where it was sent
            target = {"turn_on": "on", "turn_off": "off", "open_cover": "open", "close_cover": "closed"}[
                command.prepare[1]
            ]
            _wait_for_state(smoke, command.entity_id, (target,), timeout=30)
    smoke.rest.call_service("tts", "clear_cache", {})


def _wait_for_state(smoke: Smoke, entity_id: str, states: tuple[str, ...], timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    state: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        state = smoke.rest.get_state(entity_id)
        if state is not None and state.get("state") in states:
            return state
        time.sleep(0.25)
    raise AssertionError(f"{entity_id} is {(state or {}).get('state')!r}, expected one of {states}")


def _intent_response(result: AssistRunResult) -> tuple[str, str]:
    data = (result.first_event("intent-end") or {}).get("data") or {}
    response = (data.get("intent_output") or {}).get("response") or {}
    speech = str(((response.get("speech") or {}).get("plain") or {}).get("speech") or "")
    return str(response.get("response_type") or ""), speech


async def _check_tts(smoke: Smoke, result: AssistRunResult, text: str, outcome: TurnOutcome, *, long: bool) -> None:
    progress = result.tts_progress
    assert progress is not None, "TTS audio was never fetched"
    assert progress.status_code == 200, f"TTS URL returned HTTP {progress.status_code}"
    assert progress.error is None, f"TTS fetch failed: {progress.error!r}"
    assert progress.data, "TTS URL returned no audio"
    audio = decode_audio_bytes(bytes(progress.data))
    outcome.audio_seconds = audio.duration_seconds
    # Kept with the summary, to be listened to
    name = f"{smoke.outcomes.index(outcome):03d}-{outcome.test}-{outcome.proxy}.wav"
    (smoke.run_dir / name).write_bytes(audio.to_wav_bytes())
    outcome.notes["audio"] = name
    outcome.tts_transcript = await transcribe_with_speaches(smoke.settings, audio.to_wav_bytes())
    # Checked whether or not the reply as a whole matches, which it still does with a sentence missing
    problems: list[str] = []
    if gaps := _silences(audio):
        problems.append(
            f"silences of {SILENCE_SECONDS:g} s or more at {', '.join(gaps)}, as where a TTS model returns silence "
            "for a sentence instead of speaking it"
        )
    if missing := _missing_sentences(outcome.tts_transcript, text):
        problems.append("not spoken: " + " | ".join(sentence[:70] for sentence in missing))
    outcome.notes.update({"problems": problems} if problems else {})
    try:
        assert_reply_match(outcome.tts_transcript, text, long_reply=long)
    except AssertionError as exc:
        if not problems:
            raise
        raise AssertionError(f"{exc}\n  " + "\n  ".join(problems)) from None
    assert not problems, "; ".join(problems)


SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+")
SENTENCE_MIN_WORDS = 4  # A shorter one is left to the match of the reply as a whole
SENTENCE_MIN_SCORE = 75  # rapidfuzz partial_ratio; Whisper mishears a word or two of a long sentence


def _comparable(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", normalize_numbers(text).lower()).split())


def _missing_sentences(transcript: str, text: str) -> list[str]:
    """The sentences of the text, of SENTENCE_MIN_WORDS or more, that the transcript of its audio does not contain."""

    spoken = _comparable(transcript)
    missing = []
    for sentence in SENTENCE_RE.split(text.strip()):
        expected = _comparable(sentence)
        if len(expected.split()) >= SENTENCE_MIN_WORDS and fuzz.partial_ratio(expected, spoken) < SENTENCE_MIN_SCORE:
            missing.append(sentence.strip())
    return missing


SILENCE_SECONDS = 3.0
SILENCE_LEVEL = 150  # RMS of 16-bit samples below which a quarter second counts as silent


def _silences(audio: WavData) -> list[str]:
    """Stretches of near silence of at least SILENCE_SECONDS in 16-bit audio, as "7.5-13.8s"."""

    if audio.width != 2:
        return []
    samples = array.array("h", audio.pcm)
    step = max(1, int(audio.rate * 0.25) * audio.channels)
    quiet = []
    for start in range(0, len(samples), step):
        block = samples[start : start + step]
        quiet.append((sum(sample * sample for sample in block) / max(1, len(block))) ** 0.5 < SILENCE_LEVEL)
    gaps: list[str] = []
    begun: int | None = None
    for index, is_quiet in enumerate([*quiet, False]):
        if is_quiet and begun is None:
            begun = index
        elif not is_quiet and begun is not None:
            if (index - begun) * 0.25 >= SILENCE_SECONDS:
                gaps.append(f"{begun * 0.25:.1f}-{index * 0.25:.1f}s")
            begun = None
    return gaps


async def _device_turn(
    smoke: Smoke,
    test: str,
    proxy: Proxy,
    command: Command,
    voice: str,
    *,
    conversation_id: str | None = None,
) -> TurnOutcome:
    """Speak one command to a proxy's devices pipeline and check what Home Assistant did and said."""

    outcome = TurnOutcome(test=test, proxy=proxy.name, phrase=command.phrase)
    smoke.outcomes.append(outcome)
    audio = await asyncio.to_thread(smoke.audio, command.phrase, voice)
    started = time.monotonic()
    try:
        result = await run_assist(
            smoke.settings,
            smoke.stack,
            smoke.devices[proxy.name],
            audio,
            conversation_id=conversation_id,
            event_timeout=120,
            tts_timeout=120,
        )
        outcome.total_seconds = time.monotonic() - started
        outcome.stt_seconds = _stage_seconds(result, "stt-start", "stt-end")
        outcome.first_audio_seconds = _first_audio_seconds(result)
        outcome.stt_end_at = result.received("stt-end")
        assert not result.error_events, f"pipeline reported errors: {result.error_events}"
        stt_data = (result.first_event("stt-end") or {}).get("data") or {}
        outcome.stt_text = str((stt_data.get("stt_output") or {}).get("text") or "")
        assert_transcript_match(outcome.stt_text, command.phrase)
        outcome.response_type, outcome.speech = _intent_response(result)
        assert outcome.response_type and outcome.response_type != "error", (
            f"Assist did not act on {command.phrase!r}: {outcome.response_type} {outcome.speech!r}"
        )
        assert outcome.speech.strip(), "Assist gave no spoken reply"
        if command.entity_id and command.expected:
            state = await asyncio.to_thread(
                _wait_for_state, smoke, command.entity_id, command.expected, STATE_TIMEOUT_SECONDS
            )
            if command.brightness is not None:
                brightness = int((state.get("attributes") or {}).get("brightness") or 0)
                low, high = command.brightness
                assert low <= brightness <= high, f"brightness {brightness}, expected {low}..{high}"
        await _check_tts(smoke, result, outcome.speech, outcome, long=False)
        outcome.passed = True
    except BaseException as exc:
        outcome.error = repr(exc)
        raise
    finally:
        if outcome.total_seconds is None:
            outcome.total_seconds = time.monotonic() - started
    return outcome


async def _devices_at_once(smoke: Smoke, test: str, proxy: Proxy, commands: tuple[Command, ...]) -> list[Any]:
    """Run the commands as turns at once, each in its own voice, once their devices are where they start."""

    await asyncio.to_thread(_prepare, smoke, commands)
    for index, command in enumerate(commands):
        await asyncio.to_thread(smoke.audio, command.phrase, VOICES[index % len(VOICES)])
    turns = [
        _device_turn(smoke, test, proxy, command, VOICES[index % len(VOICES)])
        for index, command in enumerate(commands)
    ]
    return await asyncio.wait_for(asyncio.gather(*turns, return_exceptions=True), timeout=300)


def _raise_first(results: list[Any]) -> None:
    errors = [result for result in results if isinstance(result, BaseException)]
    if errors:
        raise AssertionError(f"{len(errors)} of {len(results)} turns failed; first: {errors[0]!r}") from errors[0]


def _timing_notes(outcomes: list[TurnOutcome]) -> dict[str, Any]:
    totals = [outcome.total_seconds for outcome in outcomes if outcome.total_seconds is not None]
    stts = [outcome.stt_seconds for outcome in outcomes if outcome.stt_seconds is not None]
    return {
        "turns": len(outcomes),
        "total_median_s": round(statistics.median(totals), 2) if totals else None,
        "total_max_s": round(max(totals), 2) if totals else None,
        "stt_max_s": round(max(stts), 2) if stts else None,
    }


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.parametrize("proxy", _proxies(DEFAULT, OPENAI))
async def test_device_conversation(smoke: Smoke, proxy: Proxy) -> None:
    """A few turns of one conversation: lights on, off and dimmed, a garage door opened, asked about and closed."""

    await asyncio.to_thread(_prepare, smoke, CONVERSATION)
    conversation_id = f"smoke-{secrets.token_hex(6)}"
    for command in CONVERSATION:
        await _device_turn(smoke, "conversation", proxy, command, VOICES[0], conversation_id=conversation_id)


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.parametrize("proxy", _proxies(DEFAULT, OPENAI))
async def test_streamed_llm_style_reply(smoke: Smoke, proxy: Proxy) -> None:
    """A long reply streamed in deltas, as from an LLM agent, spoken sentence by sentence."""

    outcome = TurnOutcome(test="llm-reply", proxy=proxy.name, phrase="What's the weather like tomorrow in Berlin?")
    smoke.outcomes.append(outcome)
    smoke.rest.call_service("ha_e2e_scripted", "set_reply", {"reply": LONG_REPLY, "delta_delay_ms": 150})
    smoke.rest.call_service("tts", "clear_cache", {})
    audio = await asyncio.to_thread(smoke.audio, outcome.phrase, VOICES[1])
    started = time.monotonic()
    try:
        result = await run_assist(smoke.settings, smoke.stack, smoke.scripted[proxy.name], audio, tts_timeout=300)
        outcome.total_seconds = time.monotonic() - started
        outcome.stt_seconds = _stage_seconds(result, "stt-start", "stt-end")
        outcome.first_audio_seconds = _first_audio_seconds(result)
        assert not result.error_events, f"pipeline reported errors: {result.error_events}"
        await _check_tts(smoke, result, LONG_REPLY, outcome, long=True)
        outcome.passed = True
    except BaseException as exc:
        outcome.error = repr(exc)
        raise


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.parametrize("proxy", _proxies(DEFAULT, LIMITED, UNLIMITED, OPENAI, OPENAI_LIMITED))
async def test_satellites_at_once(smoke: Smoke, proxy: Proxy) -> None:
    """Three satellites speak at the same time, each controlling its own device; every turn goes through."""

    waits_before = _log_count(proxy, STT_WAIT_LOG)
    results = await _devices_at_once(smoke, "satellites", proxy, SATELLITES)
    turns = [result for result in results if isinstance(result, TurnOutcome)]
    note = {**_timing_notes(turns), "stt_waits_logged": _log_count(proxy, STT_WAIT_LOG) - waits_before}
    smoke.outcomes.append(TurnOutcome(test="satellites", proxy=proxy.name, phrase="(summary)", passed=True, notes=note))
    _raise_first(results)
    if proxy.stt_limit == 0:
        assert note["stt_waits_logged"] == 0, "a request waited for a slot on a proxy without a limit"
    elif proxy.openai and proxy.stt_limit == 1:
        # A Realtime session holds its slot from audio-start, and the three start together: two must wait
        assert note["stt_waits_logged"] >= 1, "no Realtime session waited for the only STT slot"


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.parametrize("proxy", _proxies(DEFAULT, LIMITED, UNLIMITED, OPENAI, OPENAI_LIMITED))
async def test_long_replies_at_once(smoke: Smoke, proxy: Proxy) -> None:
    """Three satellites read out long replies at the same time; each one's audio is complete and in order."""

    smoke.rest.call_service("tts", "clear_cache", {})

    async def one(text: str) -> TurnOutcome:
        outcome = TurnOutcome(test="long-replies", proxy=proxy.name, phrase=text[:60])
        smoke.outcomes.append(outcome)
        started = time.monotonic()
        try:
            result = await run_assist_text(
                smoke.settings, smoke.session.access_token, smoke.devices[proxy.name], text, tts_timeout=300
            )
            outcome.total_seconds = time.monotonic() - started
            outcome.first_audio_seconds = _first_audio_seconds(result)
            assert not result.error_events, f"pipeline reported errors: {result.error_events}"
            await _check_tts(smoke, result, text, outcome, long=True)
            outcome.passed = True
        except BaseException as exc:
            outcome.error = repr(exc)
            raise
        return outcome

    results = await asyncio.wait_for(asyncio.gather(*(one(text) for text in LONG_TEXTS), return_exceptions=True), 600)
    _raise_first(results)


async def _exec_in(proxy: Proxy, script: str, *args: str) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        "docker", "exec", proxy.container, "python", "-c", script, *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )


async def _finish(process: asyncio.subprocess.Process, timeout: float = 60) -> None:
    """Wait for a client started in a proxy's container to end on its own: killing `docker exec` would leave it."""

    try:
        await asyncio.wait_for(process.wait(), timeout)
    except TimeoutError:
        process.kill()
        await process.wait()


def _container_running(name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", name], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() == "true"


async def _ensure_running(proxy: Proxy) -> None:
    """Start a proxy a test stopped, and wait until it takes connections, so later tests have it."""

    if not await asyncio.to_thread(_container_running, proxy.container):
        await asyncio.to_thread(start_container, proxy.container)
    await asyncio.to_thread(wait_for_tcp, "127.0.0.1", proxy.host_port, 60)
    await asyncio.sleep(2)


async def _when_logged(proxy: Proxy, needle: str, previous_count: int, timeout: float) -> float:
    """time.monotonic() at which a poll first sees a new log line with `needle`, at most LOG_POLL_SLACK late."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await asyncio.to_thread(_log_count, proxy, needle) > previous_count:
            return time.monotonic()
        await asyncio.sleep(0.2)
    raise AssertionError(f"{proxy.container} logged no new {needle!r} within {timeout:g}s")


async def _restart(proxy: Proxy) -> float:
    """Stop a proxy as `docker stop` does, start it again, and return how long the stop took."""

    started = time.monotonic()
    exit_code: int | None = None
    try:
        await asyncio.to_thread(stop_container, proxy.container)
        stopped = time.monotonic() - started
        exit_code = await asyncio.to_thread(container_exit_code, proxy.container)
    finally:
        await _ensure_running(proxy)
    assert exit_code == 0, f"{proxy.container} exited with {exit_code} after {stopped:.1f}s"
    return stopped


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.parametrize("proxy", _proxies(LIMITED, OPENAI_LIMITED))
async def test_stuck_client_frees_the_tts_slot(smoke: Smoke, proxy: Proxy) -> None:
    """
    A client that stops reading while it holds the only TTS slot fails within the client write limit, and a
    satellite waiting for the slot is answered after it. With that client still connected, the proxy still stops
    promptly and cleanly.
    """

    slow_before = _log_count(proxy, SLOW_CLIENT_LOG)
    stuck = await _exec_in(proxy, STUCK_TTS_CLIENT, LONG_REPLY, "40")
    freed: asyncio.Task[float] | None = None
    try:
        # Long enough for its synthesis to take the slot and fill the buffers its client does not read
        await asyncio.sleep(3)
        assert stuck.returncode is None, "the stuck client ended before it could hold the slot"
        assert _log_count(proxy, SLOW_CLIENT_LOG) == slow_before, "the stuck client failed before the satellite asked"
        freed = asyncio.create_task(_when_logged(proxy, SLOW_CLIENT_LOG, slow_before, 40))
        smoke.rest.call_service("tts", "clear_cache", {})
        text = "The kitchen lights are on, and the garage door is closed."
        outcome = TurnOutcome(test="stuck-client", proxy=proxy.name, phrase=text)
        smoke.outcomes.append(outcome)
        started = time.monotonic()
        try:
            result = await run_assist_text(smoke.settings, smoke.session.access_token, smoke.devices[proxy.name], text)
            outcome.total_seconds = time.monotonic() - started
            outcome.first_audio_seconds = _first_audio_seconds(result)
            assert not result.error_events, f"pipeline reported errors: {result.error_events}"
            await _check_tts(smoke, result, text, outcome, long=False)
            freed_at = await freed
            progress = result.tts_progress
            first_audio = None if progress is None else (progress.first_audio_at or progress.first_byte_at)
            assert first_audio is not None, "the satellite got no audio"
            outcome.notes["freed_after_s"] = round(freed_at - started, 2)
            # Its audio waited for the slot, which the stuck client held until the write limit failed it
            assert first_audio >= freed_at - LOG_POLL_SLACK, (
                f"the satellite's audio came {freed_at - first_audio:.1f}s before the stuck client's slot was freed: "
                "it did not wait for the slot"
            )
            logs = container_logs(proxy.container)
            outcome.notes["log"] = next(line for line in logs.splitlines()[::-1] if SLOW_CLIENT_LOG in line)[-140:]
            # The stuck client is still connected, with audio it never takes left in its connection
            stop_seconds = await _restart(proxy)
            outcome.notes["stop_with_stuck_client_s"] = round(stop_seconds, 2)
            assert stop_seconds < 8, f"stopping the proxy took {stop_seconds:.1f}s with the stuck client connected"
            outcome.passed = True
        except BaseException as exc:
            outcome.error = repr(exc)
            raise
    finally:
        if freed is not None and not freed.done():
            freed.cancel()
        await _finish(stuck)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.parametrize("proxy", _proxies(OPENAI_LIMITED))
async def test_silent_realtime_client_frees_the_stt_slot(smoke: Smoke, proxy: Proxy) -> None:
    """
    A Realtime client that starts a stream and sends no audio holds the only STT slot until the idle limit fails
    it; a satellite that speaks meanwhile is transcribed after that.
    """

    idle_before = _log_count(proxy, IDLE_LOG)
    recording_before = _log_count(proxy, REALTIME_RECORDING_LOG)
    await asyncio.to_thread(_prepare, smoke, CONVERSATION[:1])
    await asyncio.to_thread(smoke.audio, CONVERSATION[0].phrase, VOICES[2])
    idle = await _exec_in(proxy, IDLE_STT_CLIENT, "30")
    freed: asyncio.Task[float] | None = None
    try:
        # Its session is open, and holds the only slot
        await _when_logged(proxy, REALTIME_RECORDING_LOG, recording_before, 20)
        assert _log_count(proxy, IDLE_LOG) == idle_before, "the silent client was failed before the satellite spoke"
        freed = asyncio.create_task(_when_logged(proxy, IDLE_LOG, idle_before, 30))
        outcome = await _device_turn(smoke, "silent-client", proxy, CONVERSATION[0], VOICES[2])
        freed_at = await freed
        assert outcome.stt_end_at is not None, "Home Assistant reported no end of the STT stage"
        assert outcome.stt_end_at >= freed_at - LOG_POLL_SLACK, (
            f"the satellite was transcribed {freed_at - outcome.stt_end_at:.1f}s before the silent client's slot was "
            "freed: it did not wait for the slot"
        )
        logs = container_logs(proxy.container)
        outcome.notes["log"] = next(line for line in logs.splitlines()[::-1] if IDLE_LOG in line)[-120:]
    finally:
        if freed is not None and not freed.done():
            freed.cancel()
        await _finish(idle)


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_proxy_restart_under_traffic(smoke: Smoke) -> None:
    """
    The proxy stops while it answers three satellites at its limits of 1: the stop interrupts at least one turn,
    every turn ends and none hangs, the proxy exits cleanly and promptly, and once it is back the next turn works.
    """

    proxy = LIMITED
    await asyncio.to_thread(_prepare, smoke, SATELLITES)
    # Synthesized before anything starts, so the turns reach the proxy together
    audio = [await asyncio.to_thread(smoke.audio, command.phrase, VOICES[i]) for i, command in enumerate(SATELLITES)]
    cancels_before = _log_count(proxy, STT_CANCEL_LOG)
    waits_before = _log_count(proxy, STT_WAIT_LOG)
    posts_before = _log_count(proxy, TRANSCRIPTION_POST_LOG)

    async def answering() -> bool:
        waits, posts = await asyncio.gather(
            asyncio.to_thread(_log_count, proxy, STT_WAIT_LOG),
            asyncio.to_thread(_log_count, proxy, TRANSCRIPTION_POST_LOG),
        )
        return waits > waits_before or posts > posts_before

    # A turn whose STT the stop fails never has its TTS URL filled, so its fetch is allowed to fail; the pipeline
    # itself must still end
    turns = [
        asyncio.create_task(
            run_assist(
                smoke.settings,
                smoke.stack,
                smoke.devices[proxy.name],
                clip,
                event_timeout=60,
                tts_timeout=15,
                allow_tts_error=True,
            )
        )
        for clip in audio
    ]
    stop_seconds: float | None = None
    exit_code: int | None = None
    results: list[Any] = []
    try:
        # Stopped once the proxy is answering them: a transcription has come back, or one waits for the slot
        deadline = time.monotonic() + 30
        while not await answering():
            assert time.monotonic() < deadline, "the turns never reached the proxy"
            await asyncio.sleep(0.1)
        started = time.monotonic()
        await asyncio.to_thread(stop_container, proxy.container)
        stop_seconds = time.monotonic() - started
        exit_code = await asyncio.to_thread(container_exit_code, proxy.container)
        results = await asyncio.wait_for(asyncio.gather(*turns, return_exceptions=True), timeout=120)
    finally:
        for turn in turns:
            if not turn.done():
                turn.cancel()
        await asyncio.gather(*turns, return_exceptions=True)
        # Back before anything is asserted, so a failure here does not leave the later tests without it
        await _ensure_running(proxy)

    ended = [result for result in results if isinstance(result, AssistRunResult)]
    interrupted = [result for result in ended if result.error_events]
    note = {
        "turns_ok": len(ended) - len(interrupted),
        "turns_interrupted": len(interrupted),
        "pipeline_errors": sorted(
            {str((event.get("data") or {}).get("code")) for result in interrupted for event in result.error_events}
        ),
        "turns_not_ended": len(results) - len(ended),
        "exit_code": exit_code,
        "stop_seconds": None if stop_seconds is None else round(stop_seconds, 2),
        "stt_requests_cancelled_by_stop": _log_count(proxy, STT_CANCEL_LOG) - cancels_before,
    }
    smoke.outcomes.append(TurnOutcome(test="restart", proxy=proxy.name, phrase="(summary)", passed=True, notes=note))
    assert exit_code == 0, f"the proxy exited with {exit_code}"
    assert stop_seconds is not None and stop_seconds < 8, f"stopping the proxy took {stop_seconds}s"
    not_ended = [result for result in results if not isinstance(result, AssistRunResult)]
    assert not not_ended, f"a pipeline did not end after the proxy stopped: {not_ended!r}"
    for result in ended:
        assert result.events[-1].get("type") == "run-end", f"a pipeline did not reach run-end: {result.events[-1]}"
    assert interrupted, "every turn finished before the stop: it did not hit their traffic"

    await asyncio.to_thread(_prepare, smoke, SATELLITES[:1])
    await _device_turn(smoke, "after-restart", proxy, SATELLITES[0], VOICES[3])


@pytest.mark.asyncio
@pytest.mark.timeout(900)
@pytest.mark.parametrize("proxy", _proxies(DEFAULT, UNLIMITED, LIMITED))
async def test_eight_satellites_at_once(smoke: Smoke, proxy: Proxy) -> None:
    """Eight turns at once on the GPU: all go through, and no Home Assistant connection meets the write limit."""

    waits_before = _log_count(proxy, STT_WAIT_LOG)
    slow_before = _log_count(proxy, SLOW_CLIENT_LOG)
    results = await _devices_at_once(smoke, "eight", proxy, LOAD)
    turns = [result for result in results if isinstance(result, TurnOutcome)]
    note = {
        **_timing_notes(turns),
        "stt_waits_logged": _log_count(proxy, STT_WAIT_LOG) - waits_before,
        "slow_client_failures": _log_count(proxy, SLOW_CLIENT_LOG) - slow_before,
    }
    smoke.outcomes.append(TurnOutcome(test="eight", proxy=proxy.name, phrase="(summary)", passed=True, notes=note))
    _raise_first(results)
    assert note["slow_client_failures"] == 0, "a Home Assistant connection met the client write limit"
    if proxy.stt_limit == 0:
        assert note["stt_waits_logged"] == 0, "a request waited for a slot on a proxy without a limit"
    elif proxy.stt_limit == 1:
        assert note["stt_waits_logged"] >= 1, "none of eight transcriptions at once waited for the only STT slot"
