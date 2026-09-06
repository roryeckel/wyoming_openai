# Contributing to Wyoming OpenAI

Thanks for contributing.

This project is an OpenAI-compatible proxy for the Wyoming protocol. Contributions are most useful when they keep that contract clear, stay small, and come with enough evidence to show where a problem belongs.

## Scope

Wyoming OpenAI targets OpenAI-compatible request and response behavior.

That means contributions are generally in scope when they improve or fix:

- OpenAI-compatible `/v1/audio/transcriptions`
- OpenAI-compatible `/v1/audio/speech`
- OpenAI Realtime transcription sessions used through `/v1/realtime`
- Wyoming protocol handling and event flow
- configuration, docs, tests, and packaging around those surfaces

This project also includes a small backend compatibility layer for known backends such as `SPEACHES`, `KOKORO_FASTAPI`, and `LOCALAI`.

Those backend shims are limited compatibility conveniences. They are not a general policy of supporting arbitrary provider-specific APIs.

## Out of Scope

The following are usually out of scope unless explicitly approved by the maintainer first:

- provider-specific custom endpoints such as `/tts`
- provider-specific streaming flags or transport semantics outside the OpenAI-compatible surface
- custom request or response schemas for a single upstream wrapper
- speculative compatibility code without a concrete supported use case
- new provider-specific health-check or autodetection endpoints (such as custom `/health`, `/readyz`, or `/test` routes) when explicit backend configuration or the OpenAI-compatible surface itself can identify the backend

The one accepted exception to "no provider-specific routes" is a helper that fills a gap the OpenAI spec does not cover — for example, listing available voices, which OpenAI has no endpoint for. A provider-specific route is justifiable only when it provides a capability the OpenAI-compatible surface lacks. Pure identity or health probes do not meet that bar.

If an upstream project exposes both an OpenAI-compatible route and a custom route, this project targets the OpenAI-compatible route.

For more detail, see [COMPATIBILITY.md](COMPATIBILITY.md).

## Before Opening an Issue

Please first:

- search existing issues and discussions
- confirm the exact endpoint or behavior involved
- gather a minimal reproduction if possible
- check whether the issue is on an OpenAI-compatible surface or on a custom provider API

Use GitHub Discussions for:

- setup help
- backend questions
- "is this supported?"
- general usage questions

Use GitHub Issues for:

- reproducible bugs
- well-scoped feature requests
- documentation errors
- compatibility reports on the supported OpenAI-compatible surface

## Good Bug Reports

The most helpful bug reports include:

- exact version or commit
- provider or upstream server name and version
- exact endpoint or behavior involved
- minimal reproduction steps
- expected behavior
- actual behavior
- relevant logs or configuration snippets
- links to upstream issues if the behavior may come from another project

When a report depends on another server or wrapper, include a link to that project and the exact route being used.

## Good Feature Requests

Good feature requests:

- explain the user problem first
- stay within the project's supported OpenAI-compatible boundary
- explain why existing behavior is insufficient
- avoid asking this project to learn a custom API from another provider or wrapper

If the request depends on a provider-specific custom route, it is unlikely to be accepted here.

## Pull Requests

### Keep changes narrow

Prefer the smallest correct change.

- one concern per PR
- avoid unrelated refactors
- avoid adding fallback behavior for undocumented responses
- do not expand scope to custom provider APIs without prior maintainer approval

### Add or update tests when behavior changes

If you change supported behavior, add or update tests.

### Update docs when user-facing behavior changes

If the CLI, compatibility story, or visible behavior changes, update the relevant docs.

### Preserve boundaries

PRs that mainly add support for one provider's custom route, schema, or transport are likely to be declined even if the code works.

## Development Setup

Create and activate a virtual environment, then install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Run linting:

```bash
ruff check .
pyright
```

Useful targeted commands:

```bash
pytest tests/test_handler.py
ruff check . --fix
```

## Naming and URL Conventions

Different surfaces use different name forms, each matching the form its platform renders as canonical.

- **GitHub (underscore: `wyoming_openai`)** — canonical repository URL is `https://github.com/roryeckel/wyoming_openai`. Use the underscore form for all GitHub web links, `git clone` URLs, and badge URLs (image src and href).
- **PyPI (hyphen: `wyoming-openai`)** — PyPI normalizes the project name and renders it with a hyphen on `https://pypi.org/project/wyoming-openai/`. Use the hyphen form for PyPI URLs and `pip install` commands. Both `pip install wyoming-openai` and `pip install wyoming_openai` work.
- **Python package, module, and Docker image (underscore: `wyoming_openai`)** — the importable module (`python -m wyoming_openai`), the `pyproject.toml` project name, and the published Docker image (`ghcr.io/roryeckel/wyoming_openai`) all use the underscore form.

## Review Expectations

Community PRs are reviewed primarily for:

- correctness
- scope fit
- maintenance cost
- regression risk
- test coverage on supported behavior

Even correct code may be declined if it expands the project's scope beyond OpenAI-compatible behavior.

## Upstream vs Local Issues

Not every compatibility problem belongs in this repo.

If another project advertises OpenAI compatibility but diverges on the actual behavior of `/v1/audio/speech`, `/v1/audio/transcriptions`, or `/v1/realtime`, the fix may belong upstream.

In general:

- if the problem is on the supported OpenAI-compatible surface and this repo mishandles it, open an issue here
- if the problem exists only on a custom upstream route or wrapper-specific transport, raise it upstream instead

## Tone

Please be direct, factual, and respectful.

Community issues and PRs are welcome, but maintaining a narrow, predictable contract is a project goal, not a rejection of contributors.
