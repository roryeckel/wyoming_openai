---
name: community-triage
description: GitHub issues, pull requests, bug reports, scope questions, and support threads. Use when handling community reports or PRs to separate supported OpenAI-compatible behavior from out-of-scope provider-specific requests, request missing evidence, route upstream, and close or escalate politely.
---

# Community Triage

Handle community issues and PRs with respect, evidence, and firm scope discipline.

This skill exists to protect maintainer time, avoid speculative changes, and preserve the project's intended boundary.

## Project Boundary

- Wyoming OpenAI is an OpenAI-compatible proxy/middleware.
- Treat OpenAI-compatible request/response behavior as the primary contract.
- Existing backend enum/autodetection exceptions are compatibility shims, not a license to support arbitrary provider-specific transports, routes, or schemas.
- Do not expand scope to custom endpoints such as `/tts`, provider-specific `stream=true` semantics, or bespoke response formats unless the maintainer explicitly chooses to widen scope.

## Goals

- Be helpful without overcommitting.
- Separate local defects from upstream limitations and unsupported integrations.
- Prefer minimal actionable next steps: fix locally, request reproduction details, route upstream, or close with clear reasoning.
- Maintain a respectful tone even when issue quality is low.

## Default Posture

- Assume good faith.
- Do not mirror vague or incorrect technical framing.
- Translate the report into the actual technical question the project can answer.
- Use evidence from code, tests, docs, and upstream behavior before concluding.
- Do not let conversation momentum expand the project's scope by accident.

## Triage Workflow

1. Classify the thread.
2. Identify the supported surface.
3. Gather evidence.
4. Decide ownership.
5. Respond with a clear outcome.

### 1. Classify the thread

Choose the closest category:

- bug/regression on a supported surface
- feature request inside scope
- feature request outside scope
- support question or configuration issue
- upstream compatibility issue
- low-fidelity report with insufficient reproduction
- community PR

### 2. Identify the supported surface

Pin down the exact surface before deciding anything:

- endpoint path
- Wyoming event flow
- CLI/config flag
- backend compatibility path
- testable request/response behavior

Explicitly determine whether the report concerns an OpenAI-compatible path or instead depends on a custom provider API.

### 3. Gather evidence

Use the repo and upstream references, not assumptions.

- inspect the relevant implementation and tests in this repo
- read linked upstream docs/issues/PRs, not just their titles or summaries
- verify the exact route, request schema, and response transport involved
- distinguish between "supports streaming somewhere" and "streams on the exact supported integration surface used here"

### 4. Decide ownership

Assign the issue to one bucket:

- local defect in this repo
- upstream defect or limitation
- out-of-scope custom integration request
- uncertain because key evidence is still missing

### 5. Respond with a clear outcome

Pick one outcome and be explicit:

- implement/fix
- ask for concrete reproduction details
- route upstream
- close politely with scope reasoning

## Evidence Standard

Treat these as strong evidence:

- code paths
- tests
- official docs
- concrete reproductions
- exact request/response behavior
- linked upstream implementation details

Treat these as weak evidence:

- issue titles
- second-hand descriptions
- marketing claims
- "OpenAI-compatible" labels without route-level confirmation
- "supports streaming" claims without transport details

Do not conclude a behavior is supported just because a provider advertises compatibility. Confirm the specific surface this project actually uses.

## Minimum Missing Details To Request

When the report is ambiguous, ask for only what is necessary to decide ownership.

Good examples:

- exact endpoint path
- sample request/response behavior
- provider or server project link
- version or commit
- relevant logs
- minimal reproduction steps

Do not start by asking for everything. Ask for the smallest missing fact that would change the decision.

## Low-Fidelity Issues

When a report is vague, technically confused, or incomplete:

- extract the likely underlying question
- ask at most a small number of precise follow-ups
- request only the missing details needed to decide whether this repo owns the problem
- avoid speculative fixes or speculative roadmap commitments

If the issue remains unsupported by evidence after reasonable follow-up, close it politely rather than leaving it open indefinitely.

## Scope Guardrails

- Do not add support for provider-specific custom endpoints when an OpenAI-compatible endpoint already exists but behaves differently.
- Do not treat "it works in another server's `/tts` route" as evidence that this repo should adopt that route.
- Do not add speculative compatibility code for transports that are not part of the supported contract.
- Do not widen the compatibility layer beyond small, clearly bounded backend exceptions without explicit maintainer intent.
- Prefer "upstream should make their OpenAI-compatible surface behave correctly" over "this proxy should learn every custom dialect."

## Boundary Example

If an upstream Chatterbox server streams on a custom `/tts` endpoint but buffers on its OpenAI-compatible `/v1/audio/speech` endpoint:

- do not add `/tts` support here
- do not reopen scope just because the custom route exists
- explain that Wyoming OpenAI targets the OpenAI-compatible surface
- route streaming complaints about `/v1/audio/speech` upstream unless there is evidence this repo mishandles a truly incremental OpenAI-compatible response

## Issue Decision Matrix

### Keep Open Or Fix Here

Keep the issue open or fix it when:

- there is evidence the failure occurs on a supported OpenAI-compatible surface used by this repo
- the behavior regressed from prior repo behavior
- a minimal fix exists without expanding scope
- the report includes enough detail to reproduce or the code clearly shows a defect

### Ask For Details

Ask for more detail when:

- the failure might be in this repo, but one or two key facts are missing
- the report mixes supported and unsupported surfaces and needs disambiguation
- an upstream implementation claims compatibility but the actual route/response behavior is still unknown

### Route Upstream

Route upstream when:

- the failing behavior lives in another project's OpenAI-compatible server or provider
- the linked implementation buffers where true streaming is expected
- the only path that provides the requested capability is a custom upstream endpoint outside this repo's target contract

### Close

Close when:

- the request depends on a custom provider-specific API outside project scope
- there is no evidence of a defect in this repo after reasonable triage
- the issue asks for a nonexistent or irrelevant API surface
- the request would materially expand scope or maintenance burden beyond the maintainer's intent

## PR Review Guidance

Review community PRs primarily for:

- scope fit
- correctness
- maintenance cost
- tests on supported behavior
- regression risk

Be especially skeptical of PRs that:

- add provider-specific routes or bespoke schemas
- teach the proxy custom behavior for one server in a way that weakens the OpenAI-compatible boundary
- include broad refactors alongside compatibility changes
- add fallback code without evidence of a real supported use case

Prefer minimal changes that keep the contract clear.

If a PR changes supported behavior, ask for tests.

If asked to review a PR, present findings first with file/line references. If the PR's main value is widening scope to cover a custom upstream dialect, recommend narrowing or closing the PR instead of merging it.

## Response Style

- Thank the reporter or contributor for concrete links, repros, or code references.
- Be direct and factual.
- Avoid blame, sarcasm, or dismissal.
- Avoid phrases like "you don't know what you're talking about" even if the original framing is wrong.

Prefer phrasing such as:

- "I do not currently see evidence that the limitation is in this project."
- "This appears to be upstream of Wyoming OpenAI."
- "That endpoint is outside the OpenAI-compatible surface this project targets."
- "If you can show this failing on `/v1/audio/speech`, that would be worth revisiting."

Avoid phrasing such as:

- "Not my problem."
- "Works for me."
- speculative roadmap commitments
- over-apologizing for enforcing scope

## Closure Pattern

When closing, include all of the following:

- what was checked
- why the issue is out of scope or upstream
- the specific supported surface this project targets
- the condition under which the issue would become actionable here

Template:

> Thanks for the report and the additional reference.
>
> I checked the relevant code path and the linked implementation. Wyoming OpenAI already supports [supported behavior] on its OpenAI-compatible path, and I do not currently see evidence of a defect in this project.
>
> The behavior you are pointing to depends on [custom endpoint / upstream server behavior], which is outside the OpenAI-compatible surface this project targets.
>
> If there is evidence of the same problem on [exact supported surface], feel free to share reproduction details and it can be revisited.

## Upstream Escalation Pattern

When routing upstream:

- name the exact upstream repo, issue, or route
- describe the boundary cleanly
- do not volunteer to mirror custom APIs locally

Template:

> This looks like an upstream compatibility issue rather than a Wyoming OpenAI bug.
>
> The key question is whether [upstream project] behaves correctly on its OpenAI-compatible [route]. If that surface is buffered or otherwise diverges from OpenAI behavior, the fix belongs there rather than in this proxy.

## Re-entry Criteria

Reopen or continue when:

- a reporter provides a concrete reproduction on a supported OpenAI-compatible surface
- a community PR narrows itself to supported behavior and includes tests
- upstream changes make the supported surface behave differently and this repo needs a bounded compatibility adjustment

## Anti-Patterns

- implementing custom provider transports to salvage one wrapper
- keeping vague issues open indefinitely
- conflating "streaming exists somewhere in the upstream project" with "the supported integration surface streams here"
- adding fallback behavior for undocumented responses
- treating backend enum exceptions as permission for general vendor lock-in

## Maintainer Intent Distilled

- Be kind to community contributors.
- Keep the contract narrow.
- Favor evidence over momentum.
- Push upstream when the real defect lives upstream.
- Close unsupported or low-fidelity issues cleanly so maintainers are not left carrying indefinite support debt.
