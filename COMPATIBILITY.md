# Compatibility and Scope

This document explains what compatibility means in Wyoming OpenAI.

## Primary Contract

Wyoming OpenAI is an OpenAI-compatible proxy for the Wyoming protocol.

The primary contract is the OpenAI-compatible surface exposed by upstream providers or wrappers, not every custom API those projects may also offer.

## Supported Surfaces

These are the core surfaces this project intentionally targets.

| Surface | Status | Notes |
| --- | --- | --- |
| OpenAI-compatible `/v1/audio/transcriptions` | Supported | Includes standard transcription handling and response-streaming support for configured models. |
| OpenAI-compatible `/v1/audio/speech` | Supported | Includes provider-side audio byte streaming when the backend actually returns incremental bytes. |
| OpenAI Realtime transcription via `/v1/realtime` | Supported | Used for configured realtime STT models. |
| Wyoming `synthesize-start/chunk/stop` input flow | Supported | Wyoming-side incremental text input is supported, while still targeting OpenAI-compatible TTS upstream. |
| Wyoming transcription and synthesis event handling | Supported | Core project behavior. |

## Limited Backend Shims

This project includes a small backend compatibility layer for some known backends.

| Backend enum | Status | Notes |
| --- | --- | --- |
| `OPENAI` | First-class | Official OpenAI-compatible behavior. |
| `SPEACHES` | Limited shim | Small compatibility helpers such as backend detection and voice/model handling. |
| `KOKORO_FASTAPI` | Limited shim | Small compatibility helpers such as backend detection and voice discovery. |
| `LOCALAI` | Limited shim | Small compatibility helpers such as backend detection and voice handling. |

These shims exist to smooth known interoperability gaps. They are not a general policy of implementing every provider-specific route or dialect.

## Out of Scope by Default

The following are out of scope unless explicitly approved by the maintainer:

- provider-specific custom routes such as `/tts`
- provider-specific `stream=true` semantics outside the OpenAI-compatible surface
- custom request or response schemas for a single wrapper project
- bespoke streaming event formats or transports that are not part of the supported OpenAI-compatible contract

## How Wrapper Compatibility Is Evaluated

Projects often describe themselves as OpenAI-compatible. That label alone is not enough.

Compatibility is evaluated on the actual behavior of the route this project uses.

Questions that matter:

- Does the wrapper implement the OpenAI-compatible route used here?
- Does it accept the expected request schema?
- Does it return the expected response format?
- If streaming is claimed, does it stream on the OpenAI-compatible route or only on a custom route?

Example:

- If a wrapper streams on a custom `/tts` endpoint but buffers on its OpenAI-compatible `/v1/audio/speech` endpoint, Wyoming OpenAI will still treat the custom `/tts` route as out of scope.

## Community-Reported Compatibility

Some providers, wrappers, and deployment guides may work in practice without being explicitly tested in CI.

That usually means:

- they appear compatible on the OpenAI-compatible surface
- they may be covered by docs or examples
- they are not guaranteed to support every advanced feature or every future version

Community-reported compatibility is not the same thing as a commitment to support provider-specific custom APIs.

## When To File an Issue Here

File an issue here when:

- the failing behavior is on a supported OpenAI-compatible surface
- you can show a reproduction or concrete evidence
- this repo appears to mishandle that supported behavior

## When To File Upstream

File upstream when:

- the failure exists only on a custom provider route
- the upstream OpenAI-compatible route buffers, diverges, or otherwise does not behave like the surface this project expects
- the wrapper only provides the requested capability through a non-OpenAI-compatible endpoint

## Related Docs

- [README.md](README.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- GitHub Discussions for support and setup questions
