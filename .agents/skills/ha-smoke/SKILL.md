---
name: ha-smoke
description: Plan and run the Home Assistant smoke suite before opening a pull request. Use for any change that alters what a backend receives (request fields, models, voices, routes or transports, Realtime session settings, the audio sent, backend shims) or what Home Assistant sees (concurrency, timeouts, event order, errors, shutdown, streaming). Classifies the change, checks the request shape against each provider in the code, proposes a smoke plan for the developer to accept, runs it, and reports the results for the PR.
---

# Home Assistant Smoke Testing

The rules are in AGENTS.md, "Smoke Testing": which changes need a run, on which
providers, and when a key may be used. This skill is how to follow them. Read
that section first, and the README's "Home Assistant Smoke Tests" for the
suite itself.

## 1. Classify the change

Read the diff, not only its description, and put it in one of the classes in
AGENTS.md: request shape, proxy behavior, or neither. A change that does both is
a request-shape change. For "neither", say so with the reason, and stop.

## 2. Check the request shape

For a request-shape change only. Trace each field, value, route or transport
the change adds or alters, for each backend, in the code AGENTS.md names, and
compare it with the provider's documented request schema. Write the result as
a table, one row per provider:

| Provider | Backend | What it now receives | Accepts / ignores / rejects | Source |
|---|---|---|---|---|

Cite the code (`file:line`) and the provider's documentation for each row. A
row you could not establish says so: it is not a pass. A rejection by official
OpenAI blocks the change. A rejection elsewhere is fixed in the same work, or
raised with the developer before anything runs if the fix is hard or out of
scope.

## 3. Find out what can run

- **GPU and Docker:** `nvidia-smi` and `docker info`. Without an NVIDIA GPU the
  suite can't run here: say so in the plan, and offer the developer the
  commands to run it themselves.
- **Keys:** look only for uncommented entries in this repository's `.env`, and
  read their names, never their values. `STT_OPENAI_KEY` and `TTS_OPENAI_KEY`
  there are OpenAI keys unless the file says otherwise. A commented-out key, or
  one anywhere else, needs the developer's go-ahead.
- **Providers:** each file in `tests/ha_e2e/smoke/providers/` says in its
  `Serves:` line which service and request mode it runs; count it only for
  those. A local provider's header names its engine and how to start it. Note
  the images that would have to be pulled, and their size.
- **Ports:** the stack uses 127.0.0.1 ports 8010, 8124 and 10400-10406. A local
  engine publishes the port its compose file names (for example 8880 for
  Kokoro-FastAPI).

## 4. Propose the plan, and wait

Show the developer the plan AGENTS.md describes, and wait for them to accept it,
change it, or decline. Nothing runs before that. For example:

> **Class:** request shape. The change adds names to `prompt` on
> `/v1/audio/transcriptions` and in Realtime transcription sessions.
> **Check:** OpenAI, Speaches and Groq accept `prompt`; Mistral ignores it.
> **Runs:** the local lane; the provider lane on `openai` (HTTP) and
> `speaches-openai-backend`; the OpenAI lane (Realtime).
> **Won't run:** `groq`, whose key in `.env` is commented out.
> **Keys:** `STT_OPENAI_KEY` from `.env`. **Cost:** about 40 short OpenAI
> requests; no images to pull; about 15 minutes.

## 5. Run it

Build the image from the branch, and bring up the stack:

```bash
docker build -t wyoming_openai:smoke .
docker volume create wyoming_openai_huggingface-hub   # only if it does not exist yet
docker compose -f tests/ha_e2e/smoke/docker-compose.yml up -d --wait
pytest -m ha_smoke -v                                  # the local lane
```

For each provider in the plan, start its engine if it runs locally, then its
proxy, run the lane, and take the proxy down before the next one:

```bash
SMOKE_PROVIDER=<name> docker compose -f tests/ha_e2e/smoke/docker-compose.provider.yml up -d --wait
HA_SMOKE_PROVIDER=<name> pytest -m ha_smoke -v -k provider
docker compose -p ha_smoke_provider down
```

For Realtime, the OpenAI lane:

```bash
docker compose -f tests/ha_e2e/smoke/docker-compose.openai.yml up -d --wait
HA_SMOKE_OPENAI=1 pytest -m ha_smoke -v -k openai
docker compose -p ha_smoke_openai down
```

A key goes into `SMOKE_STT_KEY` and `SMOKE_TTS_KEY` (provider lane) or
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
  unset SMOKE_STT_KEY SMOKE_TTS_KEY
  key() { sed -n "s/^$1 *= *//p" "$ENV_FILE" | tr -d "\"'"; }
  SMOKE_STT_KEY="$(key STT_OPENAI_KEY)" SMOKE_TTS_KEY="$(key TTS_OPENAI_KEY)" SMOKE_PROVIDER=openai \
    docker compose -f tests/ha_e2e/smoke/docker-compose.provider.yml up -d --wait )
```

This reads uncommented entries only. A commented-out key is used only with
the developer's go-ahead, and read from its own line the same way.

In PowerShell, set the variables from the parsed entries in a script block,
and remove them when it ends. Whatever runs the command must not print its
environment.

## 6. Read the results

Each run writes `tests/ha_e2e/artifacts/smoke-<timestamp>/`: `summary.md`,
`summary.json`, and every reply's audio. A failure is either the proxy's or the
model's:

- **The model's:** a misheard command, a sentence returned as silence, a reply
  it made up instead of reading the text, a provider's rate limit. The reply
  check names silences and missing sentences.
- **The proxy's:** anything else, and any doubt. Build `main` as
  `wyoming_openai:smoke-main` (in a worktree), and run the same case with
  `WYOMING_OPENAI_IMAGE=wyoming_openai:smoke-main` after recreating the
  proxies. A failure on the branch but not on `main` is the change's.

A failure caused by the change is fixed before the PR opens, and the run
repeated. One that is the model's or the provider's is reported, not hidden.

## 7. Tear down

Take down what the run started: `docker compose -p ha_smoke_provider down` and
`docker compose -p ha_smoke_openai down` (both need no key, and remove the
containers that hold one), each local engine's compose project, and the stack
with `docker compose -f tests/ha_e2e/smoke/docker-compose.yml down -v` unless
the developer wants it kept. Pulled images stay unless the developer asks to
remove them.

## 8. Report

In the PR's Testing section:

- the class, and the request-shape table for a request-shape change
- each lane and provider that ran, with its pass and fail counts
- each failure, as the proxy's or the model's, and why
- the providers that did not run, and why
