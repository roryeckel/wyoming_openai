"""End-to-end tests that drive a real Home Assistant against wyoming_openai.

Run with the HA overlay stack up:

    docker build -t wyoming_openai:e2e .
    docker compose -f docker-compose.speaches-cpu.yml -f docker-compose.e2e.yml \
      -f docker-compose.ha-e2e.yml up -d
    pytest -m ha_e2e -v --timeout=600

The suite is skipped by default through the `integration` marker.  It is
test-only: the Home Assistant user and password are generated on a fresh
instance and stored under tests/ha_e2e/.state (gitignored).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
from pathlib import Path

import pytest
from e2e_helpers import fuzzy_match

from ha_e2e.config import HaE2ESettings, load_settings
from ha_e2e.docker_control import (
    container_exit_code,
    container_logs,
    start_container,
    stop_container,
    wait_for_new_log,
    wait_for_tcp,
)
from ha_e2e.ha_api import (
    HaSession,
    HaStack,
    HaUnavailableError,
    configure_ha,
    ensure_session,
)
from ha_e2e.helpers import (
    WavData,
    assert_plausible_duration,
    assert_reply_match,
    assert_transcript_match,
    decode_audio_bytes,
    normalize_numbers,
    read_wav,
    save_wav,
    short_text,
)
from ha_e2e.report import FailureRecord, RunReport, TurnRecord
from ha_e2e.runner import AssistRunResult, TtsProgress, run_assist, transcribe_with_speaches
from ha_e2e.scenarios import AudioFixture, Scenario, failure_scenario, load_manifest, load_turn_scenarios

LOGGER = logging.getLogger(__name__)

pytestmark = [pytest.mark.integration, pytest.mark.ha_e2e]

PROXY_STOP_AFTER_TTS_BYTES = 64 * 1024
PROXY_STOP_ERROR = "Server is shutting down"
STT_FAILURE_ERROR = "Speech-to-text transcription failed"


@pytest.fixture(scope="session")
def settings() -> HaE2ESettings:
    return load_settings()


@pytest.fixture(scope="session")
def ha_session(settings: HaE2ESettings) -> HaSession:
    try:
        return ensure_session(settings)
    except HaUnavailableError as exc:
        pytest.fail(str(exc), pytrace=False)


@pytest.fixture(scope="session")
def stack(settings: HaE2ESettings, ha_session: HaSession):
    configured = configure_ha(settings, ha_session)
    yield configured
    configured.rest.close()


@pytest.fixture(scope="session")
def report(settings: HaE2ESettings, stack: HaStack):
    run_report = RunReport(settings, stack.ha_version)
    yield run_report
    run_report.write()


@pytest.fixture(scope="session")
def scenarios(settings: HaE2ESettings) -> list[Scenario]:
    return load_turn_scenarios(settings)


def _fixture(settings: HaE2ESettings, scenario: Scenario) -> AudioFixture:
    fixture = load_manifest(settings)[scenario.fixture_id]
    digest = hashlib.sha256(fixture.path.read_bytes()).hexdigest()
    assert digest == fixture.sha256, f"fixture {fixture.path} does not match its manifest hash"
    return fixture


def _audio_for(fixture: AudioFixture) -> WavData:
    audio = read_wav(fixture.path)
    assert audio.rate == 16000, f"fixture must be 16 kHz, got {audio.rate}"
    assert audio.width == 2, f"fixture must be 16-bit, got {audio.width * 8}-bit"
    assert audio.channels == 1, f"fixture must be mono, got {audio.channels} channels"
    return audio


def _save_input(report: RunReport, scenario: Scenario, fixture: AudioFixture) -> Path:
    path = report.run_dir / f"turn-{scenario.turn:02d}-{scenario.id}-input.wav"
    shutil.copyfile(fixture.path, path)
    return path


def _save_events(report: RunReport, scenario: Scenario, result: AssistRunResult) -> Path:
    path = report.run_dir / f"turn-{scenario.turn:02d}-{scenario.id}-events.json"
    path.write_text(json.dumps(result.events, indent=2) + "\n", encoding="utf-8")
    return path


def _time_to_first_audio(result: AssistRunResult) -> float | None:
    progress = result.tts_progress
    if progress is None:
        return None
    first = progress.first_audio_at or progress.first_byte_at
    if first is None:
        return None
    return first - result.run_started_at


def _speech_from_intent_end(event: dict) -> str:
    intent_output = (event.get("data") or {}).get("intent_output") or {}
    response = intent_output.get("response") or {}
    speech = response.get("speech") or {}
    return str((speech.get("plain") or {}).get("speech") or "")


def _set_reply(stack: HaStack, reply: str, delay_ms: int) -> None:
    stack.rest.call_service(
        "ha_e2e_scripted",
        "set_reply",
        {"reply": reply, "delta_delay_ms": delay_ms},
    )


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "scenario_index",
    [0, 1, 2],
    ids=["multi-sentence-numbers", "long-paused-stream", "one-word"],
)
async def test_assist_conversation_turn(
    settings: HaE2ESettings,
    stack: HaStack,
    report: RunReport,
    scenarios: list[Scenario],
    scenario_index: int,
) -> None:
    scenario = scenarios[scenario_index]
    fixture = _fixture(settings, scenario)
    audio = _audio_for(fixture)
    input_path = _save_input(report, scenario, fixture)
    _set_reply(stack, scenario.reply, scenario.delta_delay_ms)

    record = TurnRecord(
        turn=scenario.turn,
        scenario=scenario.id,
        expected_text=scenario.expected_text,
        reply_chars=len(scenario.reply),
        artifacts=[input_path.name],
    )
    result: AssistRunResult | None = None
    error: BaseException | None = None
    try:
        result = await run_assist(
            settings,
            stack,
            stack.good.id,
            audio,
            conversation_id=stack.conversation_id,
            event_timeout=180,
            tts_timeout=300,
        )
        events_path = _save_events(report, scenario, result)
        record.artifacts.append(events_path.name)

        assert not result.error_events, f"pipeline reported errors: {result.error_events}"
        assert result.run_start is not None, "missing run-start event"
        assert result.run_start.get("conversation_id") == stack.conversation_id, (
            f"conversation_id changed: {result.run_start.get('conversation_id')!r}"
        )
        assert result.events[-1].get("type") == "run-end", result.events[-1]

        stt_end = result.first_event("stt-end")
        assert stt_end is not None, f"missing stt-end; events={[e.get('type') for e in result.events]}"
        stt_text = str(((stt_end.get("data") or {}).get("stt_output") or {}).get("text") or "")
        stt_match = assert_transcript_match(stt_text, scenario.expected_text)
        record.stt_text = stt_text
        record.stt_ok = True
        record.stt_score = stt_match.score
        record.stt_coverage = stt_match.coverage

        intent_start = result.first_event("intent-start")
        assert intent_start is not None, "missing intent-start"
        intent_data = intent_start.get("data") or {}
        assert intent_data.get("conversation_id") == stack.conversation_id
        intent_input = str(intent_data.get("intent_input") or "")
        assert_transcript_match(intent_input, scenario.expected_text)

        intent_end = result.first_event("intent-end")
        assert intent_end is not None, "missing intent-end"
        speech = _speech_from_intent_end(intent_end)
        assert speech.strip() == scenario.reply.strip(), (
            f"conversation result speech differs from the scripted reply: {speech[:200]!r}"
        )

        tts_start = result.first_event("tts-start")
        assert tts_start is not None, "missing tts-start"
        tts_input = str((tts_start.get("data") or {}).get("tts_input") or "")
        assert tts_input.strip() == scenario.reply.strip(), (
            f"tts-start input differs from the scripted reply: {tts_input[:200]!r}"
        )

        streaming = any(
            event.get("type") == "intent-progress"
            and bool((event.get("data") or {}).get("tts_start_streaming"))
            for event in result.events
        )
        assert streaming is scenario.expect_streaming, (
            f"expected tts_start_streaming={scenario.expect_streaming}, got {streaming}"
        )

        tts_end = result.first_event("tts-end")
        assert tts_end is not None, "missing tts-end"
        tts_output = (tts_end.get("data") or {}).get("tts_output") or {}
        tts_url = str(tts_output.get("url") or "")
        run_start_tts_url = str((result.run_start.get("tts_output") or {}).get("url") or "")
        assert tts_url, "tts-end did not include a URL"
        assert tts_url == run_start_tts_url, f"TTS URL changed between run-start and tts-end: {run_start_tts_url}"

        progress = result.tts_progress
        assert progress is not None, "TTS audio was never fetched"
        assert progress.status_code == 200, f"TTS URL returned HTTP {progress.status_code}"
        assert progress.error is None, f"TTS fetch failed: {progress.error}"
        assert progress.data, "TTS URL returned no audio"

        raw_output = bytes(progress.data)
        raw_extension = ".wav" if raw_output.startswith(b"RIFF") else ".mp3"
        raw_path = report.run_dir / f"turn-{scenario.turn:02d}-{scenario.id}-output-raw{raw_extension}"
        raw_path.write_bytes(raw_output)
        record.artifacts.append(raw_path.name)
        output_wav = decode_audio_bytes(raw_output)
        assert output_wav.rate > 0
        assert output_wav.width == 2
        assert output_wav.channels in (1, 2)
        duration = assert_plausible_duration(output_wav, scenario.reply)

        output_path = report.run_dir / f"turn-{scenario.turn:02d}-{scenario.id}-output.wav"
        save_wav(output_path, output_wav)
        record.artifacts.append(output_path.name)

        transcript = await transcribe_with_speaches(settings, output_wav.to_wav_bytes())
        reply_match = assert_reply_match(transcript, scenario.reply, long_reply=scenario.long_reply)
        record.transcript = transcript
        record.reply_score = reply_match.score
        record.reply_coverage = reply_match.coverage
        record.duration_seconds = duration
        record.time_to_first_audio = _time_to_first_audio(result)
        record.passed = True

        LOGGER.info(
            "Turn %d (%s): STT=%.2fs match, reply=%d chars, TTS=%.2fs, TTFA=%s, round-trip match",
            scenario.turn,
            scenario.id,
            stt_match.score,
            len(scenario.reply),
            duration,
            f"{record.time_to_first_audio:.2f}s" if record.time_to_first_audio is not None else "n/a",
        )
    except BaseException as exc:
        error = exc
        record.error = repr(exc)
        raise
    finally:
        if result is not None and not any(path.endswith("-events.json") for path in record.artifacts):
            events_path = _save_events(report, scenario, result)
            record.artifacts.append(events_path.name)
        record.passed = record.passed and error is None
        report.add_turn(record)


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_invalid_stt_model_reports_error(
    settings: HaE2ESettings,
    stack: HaStack,
    report: RunReport,
) -> None:
    scenario = load_turn_scenarios(settings)[2]
    fixture = _fixture(settings, scenario)
    audio = _audio_for(fixture)
    input_path = report.run_dir / "failure-invalid-stt-input.wav"
    shutil.copyfile(fixture.path, input_path)
    logs_before = await asyncio.to_thread(container_logs, settings.ha_container)
    error: BaseException | None = None
    result: AssistRunResult | None = None
    detail = ""
    passed = False
    report_artifacts = [input_path.name]
    try:
        result = await run_assist(
            settings,
            stack,
            stack.bad_stt.id,
            audio,
            start_stage="stt",
            end_stage="stt",
            conversation_id=stack.conversation_id,
            event_timeout=30,
            tts_timeout=30,
        )
        events_path = report.run_dir / "failure-invalid-stt-events.json"
        events_path.write_text(json.dumps(result.events, indent=2) + "\n", encoding="utf-8")
        assert result.error_events, (
            f"bad STT model produced no error event within the timeout; "
            f"events={[event.get('type') for event in result.events]}"
        )
        error_event = result.error_events[0]
        error_data = error_event.get("data") or {}
        assert error_data.get("code") == "stt-stream-failed", f"unexpected STT error: {error_event}"
        assert result.first_event("run-end") is not None, "pipeline did not emit run-end after the error"
        logs = await asyncio.to_thread(
            wait_for_new_log,
            settings.ha_container,
            STT_FAILURE_ERROR,
            logs_before.count(STT_FAILURE_ERROR),
            20,
        )
        assert STT_FAILURE_ERROR in logs
        passed = True
        detail = f"error={error_data.get('code')}: {error_data.get('message')}"
        LOGGER.info("Bad STT lane: %s", detail)
        report_artifacts = [input_path.name, events_path.name]
    except BaseException as exc:
        error = exc
        detail = repr(exc)
        raise
    finally:
        report.add_failure(
            FailureRecord(
                name="invalid STT model",
                passed=passed and error is None,
                detail=detail,
                artifacts=report_artifacts,
                error=repr(error) if error else None,
            )
        )


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_proxy_stop_during_streaming_reply(
    settings: HaE2ESettings,
    stack: HaStack,
    report: RunReport,
) -> None:
    scenario = failure_scenario(settings)
    fixture = _fixture(settings, scenario)
    audio = _audio_for(fixture)
    input_path = _save_input(report, scenario, fixture)
    _set_reply(stack, scenario.reply, scenario.delta_delay_ms)
    logs_before = await asyncio.to_thread(container_logs, settings.ha_container)
    stop_called = False
    result: AssistRunResult | None = None
    error: BaseException | None = None
    detail = ""
    passed = False
    report_artifacts = [input_path.name]

    async def stop_proxy(progress: TtsProgress) -> None:
        nonlocal stop_called
        stop_called = True
        LOGGER.info("Stopping proxy after %d bytes of TTS audio", progress.bytes_received)
        await asyncio.to_thread(stop_container, settings.proxy_container)

    try:
        result = await run_assist(
            settings,
            stack,
            stack.good.id,
            audio,
            conversation_id=stack.conversation_id,
            event_timeout=180,
            tts_timeout=120,
            allow_tts_error=True,
            stop_after_tts_bytes=PROXY_STOP_AFTER_TTS_BYTES,
            on_tts_bytes=stop_proxy,
        )
        events_path = _save_events(report, scenario, result)
        report_artifacts.append(events_path.name)
        assert stop_called, (
            "proxy was not stopped; received "
            f"{result.tts_progress.bytes_received if result.tts_progress else 0} bytes"
        )
        assert result.first_event("run-end") is not None, "pipeline hung after the proxy stopped"

        exit_code = await asyncio.to_thread(container_exit_code, settings.proxy_container)
        assert exit_code == 0, f"proxy did not exit cleanly on SIGTERM (exit code {exit_code})"

        logs = await asyncio.to_thread(
            wait_for_new_log,
            settings.ha_container,
            PROXY_STOP_ERROR,
            logs_before.count(PROXY_STOP_ERROR),
            20,
        )
        assert PROXY_STOP_ERROR in logs

        progress = result.tts_progress
        assert progress is not None, "no TTS fetch progress"
        detail = f"HA saw shutdown error; HTTP {progress.status_code}, {progress.bytes_received} audio bytes"
        if progress.data:
            raw_path = report.run_dir / f"failure-{scenario.id}-output.bin"
            raw_path.write_bytes(bytes(progress.data))
            report_artifacts.append(raw_path.name)
            try:
                output_wav = decode_audio_bytes(bytes(progress.data))
            except ValueError as exc:
                detail += f"; not decodable audio ({exc})"
            else:
                duration = output_wav.duration_seconds
                detail += f", {duration:.2f}s"
                output_path = report.run_dir / f"failure-{scenario.id}-output.wav"
                save_wav(output_path, output_wav)
                report_artifacts.append(output_path.name)
                minimum = max(0.35, len(scenario.reply.split()) * 0.08)
                if duration >= minimum:
                    transcript = await transcribe_with_speaches(settings, output_wav.to_wav_bytes())
                    detail += f"; transcript={short_text(transcript, 120)}"
                    normalized = normalize_numbers(transcript)
                    match = fuzzy_match(
                        normalized,
                        normalize_numbers(scenario.reply),
                        threshold=0.6,
                        coverage_threshold=0.85,
                    )
                    assert not match.matched, (
                        "HA returned the complete reply after the proxy was stopped "
                        f"(score={match.score:.1f}, coverage={match.coverage:.2f})"
                    )
        else:
            detail += "; no audio body (HA reported the stream error)"
        passed = True
        LOGGER.info("Proxy stop lane: %s", detail)
    except BaseException as exc:
        error = exc
        detail = repr(exc)
        raise
    finally:
        restart_error: BaseException | None = None
        if stop_called:
            try:
                await asyncio.to_thread(start_container, settings.proxy_container)
                await asyncio.to_thread(wait_for_tcp, "127.0.0.1", settings.wyoming_port, 90)
            except BaseException as exc:
                restart_error = exc
                if error is None:
                    error = exc
                    detail = f"{detail}; proxy restart failed: {exc}"
        report.add_failure(
            FailureRecord(
                name="proxy stop during streaming reply",
                passed=passed and error is None,
                detail=detail,
                artifacts=report_artifacts,
                error=repr(error) if error else None,
            )
        )
        if restart_error is not None and error is restart_error:
            raise restart_error
