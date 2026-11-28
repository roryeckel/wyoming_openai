# Testing

Every command here runs from the repository root, in a virtual environment with the development dependencies
installed, as [CONTRIBUTING.md](../CONTRIBUTING.md#development-setup) sets up.

| Suite | Select it with | Needs | In CI | Its checks |
|---|---|---|---|---|
| [Unit](#unit-tests) | `pytest`, the default run | Nothing | Every push, and PRs to `main` | Deterministic |
| [Wyoming integration](#wyoming-integration-tests) | `-m "integration and not ha_scripted"` | Docker | Pushes to and PRs against `main` | A real proxy and Speaches; TTS checked by transcribing it back |
| [Scripted Home Assistant](#scripted-home-assistant-tests) | `-m ha_scripted` | Docker, `ffmpeg` | PRs labeled `ha-scripted`, and every release | Deterministic: a scripted conversation agent |
| [Live Home Assistant](#live-home-assistant-tests) | `-m ha_live` | Docker, `ffmpeg`; an NVIDIA GPU makes it faster | Never | Fuzzy, on real models: a failure needs reading |

[`smoke`](#the-smoke-marker) marks the quick subset of a suite.

## Unit Tests

```bash
pytest
pytest --cov=wyoming_openai
```

All new code should include appropriate tests. The [Test](../.github/workflows/test.yml) workflow runs them on every
push, and on pull requests to `main`.

## Wyoming Integration Tests

`tests/test_e2e.py` drives a real `wyoming_openai` server over TCP and a real Speaches backend in Docker, with no Home
Assistant or microphone needed. TTS output is verified by feeding it back through STT and fuzzy-matching the
transcript against the original phrase, with tolerance for minor mishearing. The tests carry the `integration` marker
and are skipped by default:

```bash
pytest -m "integration and not ha_scripted" -v
```

The fixtures handle the backend automatically: a Speaches instance already healthy on `localhost:8000` is reused;
otherwise the Docker Compose stack (`docker-compose.speaches-cpu.yml` layered with `docker-compose.e2e.yml`, which
swaps in a smaller Whisper model for testing) is started and torn down; if Docker is unavailable, the suite skips.
Useful environment variables:

- `WYOMING_OPENAI_KEEP_SPEACHES=1` leaves the Speaches container running between invocations, for faster iteration
  (model load is the slow part).
- `WYOMING_E2E_URI=tcp://host:port` targets an already-running proxy instead of spawning a subprocess. The
  autodetection test skips in this mode, since it must control the server's environment.
- `WYOMING_E2E_ARTIFACT_DIR=<dir>` saves synthesized WAVs and server logs for debugging. CI uses it to upload
  artifacts on failure.

On Windows this works under Docker Desktop. Run pytest from a native Windows Python, not WSL, so subprocess spawning
and localhost port publishing line up.

Realtime models (`STT_REALTIME_MODELS` and `TTS_REALTIME_MODELS`) are not covered here: they use the OpenAI websocket
API, which Speaches does not implement. The live Home Assistant suite's OpenAI lane covers them.

The [Integration](../.github/workflows/integration.yml) workflow runs this suite on pushes and PRs against `main`, with
the proxy as a subprocess, and its [`smoke`](#the-smoke-marker) subset against the freshly built Docker image.

## The `smoke` Marker

`smoke` marks the quick subset of a suite: a few checks that a build starts and answers, not how well it works. Today
it marks three Wyoming integration tests (`describe`, a streamed synthesis, and one TTS round trip), which CI's
`container-smoke` job runs against the freshly built Docker image. Select it together with its suite's marker, since
each suite needs its own stack. To run what that job runs:

```bash
docker build -t wyoming_openai:e2e .
docker compose -f docker-compose.speaches-cpu.yml -f docker-compose.e2e.yml up -d
WYOMING_E2E_URI=tcp://127.0.0.1:10300 pytest -m "integration and smoke" -v
```

In PowerShell, set the variable first: `$env:WYOMING_E2E_URI = "tcp://127.0.0.1:10300"`.

## Home Assistant Tests

Two suites drive a real, headless Home Assistant against the proxy, with the harness in `tests/ha_e2e/`. Instead of a
hand-written Wyoming client, it onboards Home Assistant through the REST API, adds the Wyoming config entry through
the config-flow API, creates Assist pipelines over the websocket API, and runs each turn through `assist_pipeline/run`
from the STT stage to the TTS stage.

- The pipeline's TTS URL returns mp3, so the harness needs `ffmpeg` on the host to decode it. It is present on
  GitHub's Ubuntu runners; install it locally if `ffmpeg -version` fails.
- The harness generates a test-only Home Assistant user and password on a fresh instance and stores them, with the
  config-entry IDs it creates, in `tests/ha_e2e/.state/` (gitignored). Re-runs reuse the stored refresh token and
  reload those entries instead of adding duplicates. If Home Assistant is already onboarded but that state is
  missing, take the stack down with `down -v`.
- Each run writes its artifacts under `tests/ha_e2e/artifacts/` (gitignored, so nothing from a run is committed):
  `scripted-<timestamp>/` or `live-<timestamp>/`. Delete old runs when you no longer need them. `HA_E2E_ARTIFACT_DIR`
  moves them.

### Scripted Home Assistant Tests

A test-only custom conversation agent streams scripted replies through Home Assistant's ChatLog delta API, so
streaming TTS is exercised deterministically, without an LLM. The suite covers a multi-sentence reply with numbers, a
~2000-character reply with pauses between deltas, a one-word reply, an invalid STT model, and a proxy `docker stop`
during a long streaming reply. TTS output is fetched from Home Assistant's `tts_output` URL and transcribed directly
with Speaches, independently of the proxy. The same commands work in PowerShell:

```bash
docker build -t wyoming_openai:e2e .
docker compose -f docker-compose.speaches-cpu.yml -f docker-compose.e2e.yml -f docker-compose.ha-scripted.yml up -d
pytest -m ha_scripted -v --timeout=600
```

`HA_VERSION` pins the Home Assistant image (default `2026.10.1`). Switching versions, or resetting onboarding, needs
`docker compose -f docker-compose.speaches-cpu.yml -f docker-compose.e2e.yml -f docker-compose.ha-scripted.yml down -v`
first.

Each run's folder holds the input WAVs, the TTS output WAVs, the event JSON and `report.md`. Other useful variables
are `HA_E2E_BASE_URL`, `HA_E2E_SPEACHES_URL`, `HA_E2E_STT_MODEL`, `HA_E2E_PROXY_CONTAINER` and `HA_E2E_HA_CONTAINER`.

The fixture WAVs are committed under `tests/ha_e2e/fixtures/audio/`. They are generated once with Speaches Kokoro, not
by the proxy under test, and are 16 kHz mono with trailing silence for Home Assistant's VAD. Regenerate them with
Speaches running:

```bash
docker compose -f docker-compose.speaches-cpu.yml -f docker-compose.e2e.yml up -d speaches
python tests/ha_e2e/generate_fixtures.py
```

The failure lanes require the Docker CLI: the invalid-STT proxy runs next to the good one, and the streaming-reply
lane stops the proxy container to exercise the SIGTERM path, then restarts it for later runs.

The [Home Assistant Scripted](../.github/workflows/ha-scripted.yml) workflow runs on `workflow_dispatch` and on pull
requests labeled `ha-scripted`, not on every push; [AGENTS.md](../AGENTS.md#pull-requests) says which PRs need the
label. The release workflows call it too: the version bump cannot open a release PR until it passes, and
`draft-release.yml` will not create a draft release until it passes against the exact release commit, so publishing
the draft cannot ship a release that skipped it. The repository variable `HA_SCRIPTED_VERSION` overrides the Home
Assistant image tag that release runs use. The workflow keeps an unwired hook for a later OpenAI HTTP/Realtime lane
in CI, run only by `workflow_dispatch` with an API key secret.

### Live Home Assistant Tests

Home Assistant's own conversation agent controls the demo integration's devices through the proxy, on real STT and TTS
providers. Each turn speaks a command synthesized with Kokoro, then checks the transcript, what Assist did, the
device's new state, and the spoken reply, transcribed back. The models are real, so the reply check is fuzzy: each
sentence of four words or more must be in the transcript, the reply may not gain or lose a negation such as "not",
and it may not have a silence of 3 seconds or more. A different word elsewhere can still pass, and a failure can be
the model's rather than the proxy's, so the results need reading. [AGENTS.md](../AGENTS.md#live-testing) says which
changes need a live run, and the [`ha-live` skill](../.agents/skills/ha-live/SKILL.md) which providers they run on.

The scenarios run against proxies at the default limits (3), at 1, and with no limit (0). Each test's docstring in
`tests/ha_e2e/test_ha_live.py` says what it checks.

| Scenario | Local lane | OpenAI lane | Provider lane |
|---|---|---|---|
| A six-turn conversation: lights off, on and dimmed, a garage door opened, asked about and closed | default | default | default |
| A long reply streamed in deltas, as from an LLM agent | default | default | default |
| Three satellites at once, each controlling its own device | default, 1, none | default, 1 | default |
| Three long replies read out at once | default, 1, none | default, 1 | default |
| A client that stops reading while it holds the only TTS slot, then `docker stop` with it still connected | 1 | 1 | |
| A Realtime client that sends no audio while it holds the only STT slot | | 1 | |
| The proxy stopping while it answers three satellites, then the next turn | 1 | | |
| Eight satellites at once | default, 1, none | | |

Only `-m ha_live` selects the suite. It has no `integration` marker, so neither the default run nor `-m integration`
or `-m ha_scripted` picks it up. Its stack publishes its ports on `127.0.0.1` only, and reuses the scripted suite's
model cache volume.

**On the CPU or a GPU.** The stack and the local engines run on the CPU. With an NVIDIA GPU, layer
`tests/ha_e2e/live/docker-compose.gpu.yml` on the stack, and `tests/ha_e2e/live/docker-compose.engines.gpu.yml` on the
engines, in every command that brings them up; otherwise a later `up` moves them back to the CPU. `HA_LIVE_GPU`
picks the GPU (default `0`). Run Chatterbox with the GPU overlay: on the CPU its long replies time out.

**The local lane** runs every scenario on the stack's own Speaches:

```bash
docker build -t wyoming_openai:live .
docker volume create wyoming_openai_huggingface-hub   # once, if the scripted suite's stack never ran
docker compose -f tests/ha_e2e/live/docker-compose.yml up -d --wait
pytest -m ha_live -v
```

With a GPU, the third command is
`docker compose -f tests/ha_e2e/live/docker-compose.yml -f tests/ha_e2e/live/docker-compose.gpu.yml up -d --wait`.

**The OpenAI lane** runs on `gpt-live-transcribe` and `gpt-realtime-2.1-mini`, and every request is billed to the
key. Its proxies refuse to start without `OPENAI_API_KEY`. `HA_LIVE_OPENAI=1` adds its cases to the run, and
`-k openai` runs only them. It also tests the Realtime models themselves, so a misheard command, a sentence returned
as silence, or a reply the model makes up in place of the text fails a turn.

```bash
docker compose -f tests/ha_e2e/live/docker-compose.openai.yml up -d --wait
HA_LIVE_OPENAI=1 pytest -m ha_live -v -k openai
```

**The provider lane** runs the first four scenarios against one other provider, defined in a file under
[`tests/ha_e2e/live/providers/`](ha_e2e/live/providers): official OpenAI over HTTP, streamed (`openai`) or not
(`openai-unstreamed`), GroqCloud, Mistral, Kokoro-FastAPI, LocalAI, Chatterbox, openai-edge-tts (known broken
upstream, #139), or the stack's own Speaches, on the plain `OPENAI` backend (`speaches-openai-backend`) or in the
modes the local lane doesn't run (`speaches-other-modes`). Each file's header says which service and request mode it
serves, whether it needs a key, and, for a provider that runs locally, how to start its engine from
`docker-compose.engines.yml`. The engines run in a project of their own, without host ports, so they don't clash with
an engine you already run. A preset that serves only one side gets the other from
[`providers/defaults/speaches.env`](ha_e2e/live/providers/defaults/speaches.env), the stack's own Speaches.
`HA_LIVE_PROVIDER` names the provider for both compose and pytest, and `HA_LIVE_STT_KEY` and `HA_LIVE_TTS_KEY` carry
the key for the side or sides a paid provider serves:

```bash
export HA_LIVE_PROVIDER=kokoro-fastapi   # $env:HA_LIVE_PROVIDER = "kokoro-fastapi" in PowerShell
docker compose -f tests/ha_e2e/live/docker-compose.engines.yml --profile kokoro-fastapi up -d --wait kokoro
docker compose -f tests/ha_e2e/live/docker-compose.provider.yml up -d --wait
pytest -m ha_live -v -k provider
docker compose -p ha_live_provider down
```

The tests check that the provider proxy is running, and is the one `HA_LIVE_PROVIDER` names.

**Reading the results.** Each run's folder holds `summary.md`, `summary.json`, and every reply's audio. To tell
whether a failure is the change's, run the same case on `main`: build it as `wyoming_openai:live-main` in a worktree of
`main`, recreate the proxies with it, and run the case again. `--no-deps` leaves Speaches as it is, on the CPU or the
GPU. A case of the OpenAI or provider lane needs its own proxy recreated the same way, from that lane's compose file.

```bash
WYOMING_OPENAI_IMAGE=wyoming_openai:live-main docker compose -f tests/ha_e2e/live/docker-compose.yml up -d --wait --no-deps --force-recreate proxy proxy_limited proxy_unlimited proxy_limited_debug
```

**Tearing down.** None of these commands needs a key. The engines' model and backend volumes stay, so the next run
skips those downloads; add `-v` to the engines command to remove them:

```bash
docker compose -p ha_live_openai down
docker compose -p ha_live_provider down
docker compose -p ha_live_engines down
docker compose -f tests/ha_e2e/live/docker-compose.yml down -v
```
