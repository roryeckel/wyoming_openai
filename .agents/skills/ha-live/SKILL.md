---
name: ha-live
description: Plan and run the live Home Assistant suite (pytest -m ha_live) before opening a pull request. Use for any change that alters what a backend receives (request fields, models, voices, routes or transports, Realtime session settings, the audio sent, backend shims) or what Home Assistant sees (concurrency, timeouts, event order, errors, shutdown, streaming). Classifies the change, checks the request shape against each provider in the code, proposes a plan for the developer to accept, runs it, and reports the results for the PR.
---

# Home Assistant Live Testing

AGENTS.md, "Live Testing", says when a change needs a live run. This skill
is how to plan it, run it and report it. `tests/README.md`, "Home Assistant
Live Tests", describes the suite itself and the commands for each lane.

## 1. Classify the change

Read the diff, not only its description, and put it in one class:

| Class | What counts | Before opening the PR |
|---|---|---|
| Request shape | What a backend receives changes: request fields or their values (`prompt`, `language`, `response_format`, voice, speed, instructions, the extra-body merge), model or voice resolution, which route or transport is used, Realtime session settings, the audio sent (format, rate, chunking), auth or headers, backend detection or a backend shim | The request-shape check (step 2), then the suite on several providers (step 3) |
| Proxy behavior | What Home Assistant sees changes, but not the request: concurrency, queueing, timeouts, event order or content, error reporting, connection lifecycle and shutdown, handling of response audio, streaming versus buffering, startup | The suite on one provider, free or not, that serves the service and request mode the change touches (step 3) |
| Neither | Docs, tests, logging, CLI parsing without a behavior change, refactors the unit tests cover | No live run |

A change that does both is a request-shape change, and when unsure, take the
stricter class. Adding names to the STT prompt (#133) is a request-shape
change; limiting concurrent requests (#135) is a proxy-behavior change. For
"neither", say so with the reason, and stop. These classes are separate from
the `ha-scripted` label: a change can need both.

Then name the service the change touches (STT, TTS or both) and the request
mode its changed code is on: for STT, HTTP streamed, HTTP not streamed or
Realtime; for TTS, sentence by sentence, one request per reply, or Realtime.

## 2. Check the request shape

For a request-shape change only. Ground it in the code, before anything runs:

1. List every field, value, route or transport the change adds or alters, and
   for which models and backends.
2. Trace what each backend receives: `transcription_kwargs` and
   `_get_realtime_transcription_session()` for STT, `_iter_tts_audio()` and
   `_get_realtime_tts_session()` for TTS (all in `handler.py`), the backend
   shims in `compatibility.py`, and the extra-body validators in
   `utilities.py`.
3. Compare that with each provider's documented request schema, with
   [OpenAI's API reference](https://platform.openai.com/docs/api-reference) as
   the contract.

Write the result as a table, one row per provider:

| Provider | Backend | What it now receives | Accepts / ignores / rejects | Source |
|---|---|---|---|---|

Cite the code (`file:line`) and the provider's documentation for each row. A
row you could not establish says so: it is not a pass.

Official OpenAI is the strictest requirement: a change that breaks it on any
transport it touches does not ship. A break elsewhere is fixed in the same
work, for example by sending a field only to the backends that take it. If that
fix is hard or outside the change's scope, raise it with the developer before
anything runs, rather than ship the break.

## 3. Choose the providers

Each file in `tests/ha_e2e/live/providers/` is a preset, and its `Serves:`
line says which service and request mode it runs. A provider counts only for
those: a TTS-only provider does not count for an STT change, and a streamed
preset does not count for a change on the unstreamed path.

For a proxy-behavior change, one provider that serves the mode is enough. The
local lane runs STT not streamed and TTS sentence by sentence;
`speaches-other-modes` runs streamed STT and TTS in one request per reply, both
for free; the OpenAI lane runs Realtime.

For a request-shape change, each mode it touches runs on:

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

Some combinations have no preset: no third-party preset streams STT or speaks
Realtime, for example. The plan names each such combination as not covered,
gives the request-shape check's finding for it, citing the provider's
documentation, and asks the developer whether that is enough. Where a provider
does serve the mode, a new preset for it is the better answer, and part of the
same work.

## 4. Find out what can run

- **Docker, and a GPU:** `docker info`, and `nvidia-smi`. With an NVIDIA GPU,
  use the GPU overlays tests/README.md names. Without one, the run is on the
  CPU: slower, and Chatterbox can't run. Say which in the plan. Without
  Docker, offer the developer the commands to run it themselves.
- **Keys:** paid providers are not a hard requirement. Run one when its key is
  available; otherwise say in the plan and in the PR which providers did not
  run, and rely on the request-shape check for them. A key counts as
  available, and as permission to use it for live testing, only when it is
  clearly set for this project: an uncommented entry in this repository's
  `.env`. Read the entries' names, never their values. `STT_OPENAI_KEY` and
  `TTS_OPENAI_KEY` there are OpenAI keys unless the file says otherwise. A
  commented-out key, or one found anywhere else, needs the developer's
  go-ahead first.
- **Engines and images:** a local provider's header names its engine (its
  `Engine:` line) and how to start it. Note the images that would have to be
  pulled, and their size, and the models a local engine downloads on its first
  start.
- **Ports:** the stack uses 127.0.0.1 ports 8010, 8124 and 10400-10406. Local
  engines run from `tests/ha_e2e/live/docker-compose.engines.yml`, in their own
  project, with `ha_live_*` container names and no host ports, so they don't
  clash with an engine the developer runs.

## 5. Propose the plan, and wait

Show the developer the plan, and wait for them to accept it, change it, or
decline. Nothing runs before that. The plan says:

- the change's class, and why
- the request-shape findings, per provider
- which lanes and providers will run, and which won't, and why
- which keys will be used, and where they come from
- what the run costs: paid requests, images to pull, models to download,
  roughly how long it takes

For example:

> **Class:** request shape. The change adds names to `prompt` on
> `/v1/audio/transcriptions` and in Realtime transcription sessions.
> **Check:** OpenAI, Speaches and Groq accept `prompt`; Mistral ignores it.
> **Runs:** the local lane; the provider lane on `openai` (HTTP) and
> `speaches-openai-backend`; the OpenAI lane (Realtime).
> **Won't run:** `groq`, whose key in `.env` is commented out.
> **Keys:** `STT_OPENAI_KEY` from `.env`. **Cost:** about 40 short OpenAI
> requests; no images to pull; about 15 minutes.

## 6. Run it

Build the image from the branch, and run each lane in the plan with the
commands in tests/README.md, "Home Assistant Live Tests". For each provider,
start its engine if it runs locally (its file's `Engine:` line), then its
proxy, run the lane, and take the proxy down before the next one.

A key goes into `HA_LIVE_STT_KEY` and `HA_LIVE_TTS_KEY` (provider lane) or
`OPENAI_API_KEY` (OpenAI lane) through the environment of the one command that
starts the proxy:

- Read only that entry from `.env`, in the same process, and never echo it,
  log it or put it on a command line. Don't source `.env` as a script. In a
  worktree, `.env` is in the main checkout.
- Set only the keys for the sides the provider's `Serves:` line names, and
  clear the others first. A key left over from another provider would go to
  the backend serving the other side.
- Turn shell tracing off before reading a key: `set -x` would print it.

In bash, for example:

```bash
( set +x
  unset HA_LIVE_STT_KEY HA_LIVE_TTS_KEY
  ENV_FILE=/path/to/the/main/checkout/.env
  # The last uncommented entry, without quotes or a trailing comment
  key() { sed -n "s/^$1 *= *//p" "$ENV_FILE" | tail -n 1 | sed 's/[[:space:]]*#.*$//; s/[[:space:]]*$//' | tr -d "\"'"; }
  export HA_LIVE_STT_KEY="$(key STT_OPENAI_KEY)" HA_LIVE_TTS_KEY="$(key TTS_OPENAI_KEY)"
  [ -n "$HA_LIVE_STT_KEY" ] && [ -n "$HA_LIVE_TTS_KEY" ] || { echo "a key is missing from .env" >&2; exit 1; }
  HA_LIVE_PROVIDER=openai docker compose -f tests/ha_e2e/live/docker-compose.provider.yml up -d --wait )
```

This reads uncommented entries only. A commented-out key is used only with
the developer's go-ahead, and read from its own line the same way.

In PowerShell, set the variables from the parsed entries in a script block,
and remove them when it ends. Whatever runs the command must not print its
environment.

## 7. Read the results

Each run writes `tests/ha_e2e/artifacts/live-<timestamp>/`, which stays on the
machine: the folder is gitignored, and neither its audio nor its summaries go
in a commit. A failure is either the proxy's or the model's:

- **The model's:** a misheard command, a sentence returned as silence, a reply
  it made up instead of reading the text, a provider's rate limit. The reply
  check names silences and missing sentences.
- **The proxy's:** anything else, and any doubt. Run the same case on `main`,
  as tests/README.md, "Reading the results", says. A failure on the branch but
  not on `main` is the change's.

A failure caused by the change is fixed before the PR opens, and the run
repeated. One that is the model's or the provider's is reported, not hidden.

## 8. Tear down

Take down what the run started, with the commands in tests/README.md,
"Tearing down": the OpenAI and provider proxies always, since they hold a key,
and the engines and the stack unless the developer wants them kept. Each of
these touches only what the live run started. Pulled images and the engines'
volumes stay unless the developer asks to remove them.

## 9. Report

In the PR's Testing section:

- the class, and the request-shape table for a request-shape change
- each lane and provider that ran, with its pass and fail counts
- each failure, as the proxy's or the model's, and why
- the providers that did not run, and why
