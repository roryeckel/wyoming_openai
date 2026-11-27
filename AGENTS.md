# AGENTS.md

This file provides guidance to AI coding agents working with code in this repository. It is symlinked as `CLAUDE.md` for Claude Code compatibility.

## Project Overview

Wyoming OpenAI is a proxy middleware that bridges the Wyoming protocol with OpenAI-compatible endpoints for ASR (Automatic Speech Recognition) and TTS (Text-to-Speech) services. It enables Wyoming clients like Home Assistant to use various OpenAI-compatible STT/TTS services.

## Development Commands

### Testing
```bash
# Create and activate a repository virtual environment first

# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run tests with coverage
pytest --cov=wyoming_openai

# Run specific test file
pytest tests/test_handler.py
```

### Code Quality
```bash
# Run these commands with the repository virtual environment activated
# Run linting with Ruff
ruff check .

# Auto-fix linting issues
ruff check . --fix

# Run type checking with Pyright
pyright
```

If the virtual environment can't stay activated between commands (common in agent shells), point Pyright at it explicitly: `pyright --pythonpath <venv>/Scripts/python.exe` (`<venv>/bin/python` on Linux/macOS).

### Local Development Setup
```bash
# Install in editable mode
pip install -e .

# Run the server locally
python -m wyoming_openai --uri tcp://0.0.0.0:10300 --stt-models gpt-transcribe --tts-models gpt-4o-mini-tts
```

### Docker Development
```bash
# Build and run development container
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build

# With local services (e.g., Speaches)
docker compose -f docker-compose.speaches.yml -f docker-compose.dev.yml up -d --build
```

## Pull Requests

Every PR runs lint, tests, integration, Docker and CodeQL checks. The **Home
Assistant E2E** suite (`.github/workflows/ha-e2e.yml`) only runs on a PR with
the `ha-e2e` label; without it the check shows as skipped, which is not a pass.

Add the label (`gh pr edit <PR#> --add-label ha-e2e`) and wait for a passing
run before merging when a change could alter what Home Assistant sees over the
Wyoming connection, or when it sees it:

- which events are sent, their order or their content (`info`, transcripts,
  audio, errors), including new event types the handler answers
- when a connection is closed, or how a failure is reported
- the timing of a request: waiting, queuing, timeouts, concurrency limits,
  streaming versus buffering
- audio format handling or conversion
- startup, the Docker image or compose files

The label is usually not needed for docs, tests, logging, CLI parsing or
validation that doesn't change behavior, refactors the unit tests cover,
or request fields sent only to a backend. When unsure, label it: the run
costs a few minutes. It can also be started by hand with
`gh workflow run ha-e2e.yml --ref <branch>`.

## Smoke Testing

Before opening a PR, run the Home Assistant smoke suite (`pytest -m ha_smoke`, see
"Home Assistant Smoke Tests" in the README) when the change falls in one of the
first two classes below. The `ha-smoke` skill
(`.agents/skills/ha-smoke/SKILL.md`) has the steps. The suite needs an NVIDIA
GPU, and CI does not run it.

| Class | What counts | Before opening the PR |
|---|---|---|
| Request shape | What a backend receives changes: request fields or their values (`prompt`, `language`, `response_format`, voice, speed, instructions, the extra-body merge), model or voice resolution, which route or transport is used, Realtime session settings, the audio sent (format, rate, chunking), auth or headers, backend detection or a backend shim | The request-shape check below, then the suite on several providers |
| Proxy behavior | What Home Assistant sees changes, but not the request: concurrency, queueing, timeouts, event order or content, error reporting, connection lifecycle and shutdown, handling of response audio, streaming versus buffering, startup | The suite on one provider, free or not, that serves the service and request mode the change touches (see Providers) |
| Neither | Docs, tests, logging, CLI parsing without a behavior change, refactors the unit tests cover | No smoke run |

When unsure, take the stricter class. Adding names to the STT prompt (#133) is
a request-shape change; limiting concurrent requests (#135) is a proxy-behavior
change. For a proxy-behavior change, the local lane is enough when the change is
on STT not streamed or TTS sentence by sentence; `speaches-other-modes` runs
streamed STT and TTS in one request per reply for free, and the OpenAI lane
Realtime. These classes are separate from the `ha-e2e` label above: a change can
need both.

### Request-shape check

Ground it in the code, before anything runs:

1. List every field, value, route or transport the change adds or alters, and
   for which models and backends.
2. Trace what each backend receives: `transcription_kwargs` and
   `_get_realtime_transcription_session()` for STT, `_iter_tts_audio()` and
   `_get_realtime_tts_session()` for TTS (all in `handler.py`), the backend
   shims in `compatibility.py`, and the extra-body validators in
   `utilities.py`.
3. Compare that with each provider's documented request schema, with
   [OpenAI's API reference](https://platform.openai.com/docs/api-reference) as
   the contract, and record whether each provider accepts, ignores or rejects
   it.

Official OpenAI is the strictest requirement: a change that breaks it on any
transport it touches does not ship. A break elsewhere is fixed in the same
work, for example by sending a field only to the backends that take it. If that
fix is hard or outside the change's scope, raise it with the developer before
opening the PR rather than ship the break.

### Providers

The plan names the service the change touches (STT, TTS or both) and the
request mode its changed code is on: for STT, HTTP streamed, HTTP not streamed
or Realtime; for TTS, sentence by sentence, one request per reply, or Realtime.
For each of those, a request-shape change runs on:

- official OpenAI in that mode: `openai` (streamed STT, TTS sentence by
  sentence), `openai-unstreamed` (STT not streamed, TTS one request per reply),
  or the OpenAI lane (Realtime)
- every backend shim its code path reaches: `SPEACHES` (the local lane: STT not
  streamed, TTS sentence by sentence; `speaches-other-modes`: STT streamed, TTS
  one request per reply), `KOKORO_FASTAPI` (`kokoro-fastapi`), `LOCALAI`
  (`localai`)
- one third-party provider on the plain `OPENAI` backend that serves that
  service: `speaches-openai-backend` is free for both, and `groq` (both),
  `mistral` (STT) or `chatterbox` (TTS) count too. `openai-edge-tts` is known
  broken upstream (#139)

A provider counts only for the service and mode it runs, which its file in
`tests/ha_e2e/smoke/providers/` states in its `Serves:` line. A TTS-only
provider does not count for an STT change, and a streamed preset does not count
for a change on the unstreamed path.

Some combinations have no preset: no third-party preset streams STT or speaks
Realtime, for example. The plan names each such combination as not covered,
gives the request-shape check's finding for it, citing the provider's
documentation, and asks the developer whether that is enough before anything
runs. Where a provider does serve the mode, a new preset for it is the better
answer, and part of the same work.

Paid providers are not a hard requirement. Run one when its key is available;
otherwise say in the plan and in the PR which providers did not run, and rely on
the request-shape check for them. A key counts as available, and as permission
to use it for smoke testing, only when it is clearly set for this project: an
uncommented entry in this repository's `.env`, such as `STT_OPENAI_KEY`. A
commented-out key, or one found anywhere else, needs the developer's go-ahead
first. Never print a key or put it on a command line: pass it through the
environment, and remove the containers that hold it when the run is done.

### The plan comes first

Before running anything, show the developer the smoke plan and wait for them to
accept it:

- the change's class, and why
- the request-shape findings, per provider
- which lanes and providers will run, and which won't, and why
- which keys will be used, and where they come from
- what the run costs: paid requests, images to pull, roughly how long it takes

Report the results in the PR's Testing section: what ran, on which providers,
and what failed. Tell a model's fault (a misheard command, a sentence returned
as silence, a reply it made up) from the proxy's, by running the same case on
`main` when unsure.

## Releases

Releases off `main` go through the **Bump Version** GitHub Action
(`.github/workflows/bump-version.yml`); don't hand-edit the version strings or
create the tag. Dispatch it with `gh workflow run bump-version.yml -f
bump_type=<patch|minor|major>`: it runs the Home Assistant E2E gate, opens a
`release-bump` PR, and, once merged, the gated **Draft Release** workflow
creates the draft release. Publishing that draft fires PyPI and Docker
`latest`. See `.agents/skills/bump-version/SKILL.md` for the full flow.

## Architecture

### Core Components

- **`handler.py`**: Contains `OpenAIEventHandler` - the main Wyoming protocol event handler that processes ASR and TTS requests
- **`compatibility.py`**: Provides `CustomAsyncOpenAI` class with backend detection and OpenAI API compatibility layer
- **`__main__.py`**: Entry point with argument parsing and server initialization
- **`utilities.py`**: Helper functions for audio processing and data handling
- **`wav.py`**: WAV header parsing and `WavFramer`, which turns TTS response bytes into PCM
- **`concurrency.py`**: `ConcurrencyLimit`, the server-wide STT and TTS request limits
- **`const.py`**: Version constants and configuration

### Key Architecture Patterns

1. **Async Event Handling**: Uses Wyoming's `AsyncEventHandler` to process incoming protocol events
2. **Backend Abstraction**: `CustomAsyncOpenAI` wraps different backends (OpenAI, Speaches, LocalAI, etc.) with a unified interface
3. **Stream Processing**: Handles both streaming and non-streaming transcription modes
4. **Audio Buffer Management**: Collects audio chunks into complete files for processing
5. **Shared Limits and Shutdown**: `main()` creates one STT and one TTS `ConcurrencyLimit`, which the handler of every connection shares. Code that sends a request to a backend follows these rules:
   - The backend part of a Realtime `audio-start` and of an `audio-stop` runs through `_run_stt_request()`, so `stop()` can cancel it, and calls `_check_stt_not_stopping()` before each backend operation after the first. Realtime audio chunks are appended from the event handler directly.
   - It holds a slot with `async with limit.slot()`, or with `acquire()` when the slot outlives the event that took it, as a Realtime STT session's does.
   - While it holds a slot, it writes its events with `_write_event_or_fail()`, so a client that stops reading fails the request with `SlowClientError` and frees the slot. Final events are written after the slot is released, and the events of an abort or a shutdown with `_write_event_bounded()`.
   - An `audio-stop` is answered through `_write_transcript()` or `_report_stt_failure()`, which clear `_stt_answer_owed`, so `stop()` sends its shutdown error only to a request that is still owed an answer.
   - A Realtime STT session's cleanup detaches it with `_detach_realtime_session()`, synchronously before its first await, then closes it with `_close_realtime_session()`.

### Wyoming Protocol Flow

The handler processes these Wyoming events:
- `AudioStart/AudioChunk/AudioStop` → STT transcription
- `Transcribe` → Initiate transcription request  
- `Synthesize` → TTS audio generation

### Backend Support

The `OpenAIBackend` enum defines supported backends:
- `OPENAI`: Official OpenAI API
- `SPEACHES`: Local Speaches service
- `LOCALAI`: LocalAI service
- `KOKORO_FASTAPI`: Kokoro TTS service

## Configuration

The server accepts both command-line arguments and environment variables. Key configuration includes:
- STT/TTS API keys and URLs
- Model lists for STT and TTS
- Voice configurations
- Backend-specific settings (temperature, speed, etc.)
- STT/TTS extra request body fields (`--stt-extra-body` / `--stt-realtime-extra-body` / `--tts-extra-body` / `--tts-realtime-extra-body`)
- Realtime transports for OpenAI models (`--stt-realtime-models` / `--tts-realtime-models`)

## Testing Strategy

Tests are organized by module:
- `test_handler.py`: Event handler logic
- `test_compatibility.py`: Backend compatibility
- `test_utilities.py`: Helper functions
- `test_wav.py`: WAV header parsing and framing
- `test_concurrency.py`: Request concurrency limits
- `test_main.py`: CLI argument parsing and startup validation
- `test_integration.py`: End-to-end scenarios
- `ha_e2e/test_ha_e2e.py`: Home Assistant E2E suite, the CI gate (see Pull Requests)
- `ha_e2e/test_ha_smoke.py`: Home Assistant smoke suite, on a GPU (see Smoke Testing)
