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

Fill in every section of the PR template (`.github/PULL_REQUEST_TEMPLATE.md`).
CONTRIBUTING.md, "Fill in the PR template", says what each one takes; the
Summary sentence becomes the release-notes line.

Every PR runs lint, tests, integration, Docker and CodeQL checks. The **scripted
Home Assistant** suite (`.github/workflows/ha-scripted.yml`) only runs on a PR with
the `ha-scripted` label; without it the check shows as skipped, which is not a pass.

Add the label (`gh pr edit <PR#> --add-label ha-scripted`) and wait for a passing
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
`gh workflow run ha-scripted.yml --ref <branch>`.

## Live Testing

The live Home Assistant suite (`pytest -m ha_live`) runs Home Assistant's
voice pipeline through the proxy against real STT and TTS providers. It runs
on the CPU, faster on an NVIDIA GPU, and CI does not run it. Before opening a
PR, run it when a change alters:

- what a backend receives (a request-shape change): request fields or their
  values, model or voice resolution, the route or transport, Realtime session
  settings, the audio sent, auth or headers, backend detection or a backend
  shim. It runs on several providers. Official OpenAI is the strictest
  requirement: a change that breaks it does not ship. A break on another
  provider is fixed in the same change, or raised with the developer if the
  fix is hard.
- what Home Assistant sees, but not the request (a proxy-behavior change):
  concurrency, queueing, timeouts, event order or content, error reporting,
  connection lifecycle and shutdown, response audio, streaming versus
  buffering, startup. It runs on one provider that serves the request mode the
  change touches.

Docs, tests, logging, CLI parsing without a behavior change, and refactors the
unit tests cover need no run. When unsure, take the stricter class. This is
separate from the `ha-scripted` label above: a change can need both.

The `ha-live` skill (`.agents/skills/ha-live/SKILL.md`) has the rest: the
request-shape check, which providers count, the plan, the commands and the
report. Two of its rules hold before it is loaded:

- Show the developer the live-test plan, and wait for them to accept it, before
  running anything.
- Only an uncommented key in this repository's `.env` counts as permission to
  use it. Never print a key.

## Code Review

Before opening a PR, have the full diff from `main` reviewed independently,
and record the latest review in the PR's Code review section.
CONTRIBUTING.md, "Have the change reviewed independently", is the policy. Two
of its rules hold here:

- The review runs outside your own context: in a separate session or a
  subagent with a fresh context, on another model, or by a person. Give it the
  diff and the PR's accepted trade-offs, not your conclusions about the code.
- Aim for the latest review to be clean, with nothing changed after it. When it
  isn't, because a finding is still open or the change moved on afterwards,
  the PR says so and why.

## Releases

Releases off `main` go through the **Bump Version** GitHub Action
(`.github/workflows/bump-version.yml`); don't hand-edit the version strings or
create the tag. Dispatch it with `gh workflow run bump-version.yml -f
bump_type=<patch|minor|major>`: it runs the scripted Home Assistant gate, opens a
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

Unit tests run in the default run, one module per area. The main ones:
- `test_handler.py`: Event handler logic
- `test_compatibility.py`: Backend compatibility
- `test_utilities.py`: Helper functions
- `test_wav.py`: WAV header parsing and framing
- `test_concurrency.py`: Request concurrency limits
- `test_main.py`: CLI argument parsing and startup validation
- `test_handler_workflows.py`: Whole STT and TTS workflows through the handler, with the backend mocked
- `test_e2e_helpers.py`, `ha_e2e/test_ha_api.py`, `ha_e2e/test_helpers.py`: the Docker suites' helpers, tested
  without Docker

Each Docker suite has one marker, and nothing else selects it:
- `test_e2e.py`, `test_e2e_negative.py` (`-m integration`): the Wyoming suite, a real proxy against a real Speaches
- `ha_e2e/test_ha_scripted.py` (`-m ha_scripted`): scripted Home Assistant suite, the CI gate (see Pull Requests)
- `ha_e2e/test_ha_live.py` (`-m ha_live`): live Home Assistant suite, on real providers, never in CI (see Live Testing)

`tests/README.md` says how to run each suite and what it needs.
