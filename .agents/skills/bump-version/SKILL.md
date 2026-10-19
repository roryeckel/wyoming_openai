---
name: bump-version
description: Bump the version number of wyoming-openai.
argument-hint: "[patch|minor|major|estimate]"
disable-model-invocation: true
---

Bump version $ARGUMENTS.

## Releasing off `main` — dispatch the workflow

A release is prepared by the **Bump Version** GitHub Action, not by hand.
Do not edit `pyproject.toml`/`README.md` locally, and do not create the tag.

1. Dispatch it (or Actions → Bump Version → Run workflow):
   ```bash
   gh workflow run bump-version.yml -f bump_type=<patch|minor|major>
   ```
2. The Action runs the Home Assistant E2E gate, updates `pyproject.toml` and
   `README.md`, and opens a PR from `release/v<new>` labeled `release-bump`.
3. Optionally set the GitHub release title on the PR body's `Release title:`
   line (the version is appended automatically; blank defaults to
   `Release <new>`), then merge the PR. The **Draft Release** workflow re-runs
   the E2E gate against the merge commit and creates the draft release with
   generated notes.
4. Publishing that draft release creates the `v<new>` tag and fires the PyPI
   and Docker `latest` workflows.

The Action fails if `release/v<new>` already exists on origin; delete a stale
release branch before re-dispatching. Use `gh run watch <run-id> --exit-status`
to follow it, then check the new PR with `gh pr list`.

## Local fallback

Use the steps below only when working off a non-`main` branch, when the Action
is unavailable, or for `estimate` mode.

1. Read current version from `pyproject.toml` (line containing `version = "x.y.z"`)
2. If no argument provided, default to minor bump
3. If argument is `estimate`, analyze changes since last tag to determine bump type:
   - Use `git log` and `git diff` to review recent changes
   - **patch**: bug fixes, documentation, minor tweaks
   - **minor**: new features, non-breaking enhancements
   - **major**: breaking changes, API modifications
   - Present recommendation to user for confirmation before proceeding
4. Calculate new version:
   - **patch**: x.y.z → x.y.(z+1)
   - **minor**: x.y.z → x.(y+1).0
   - **major**: x.y.z → (x+1).0.0
5. Update version in:
   - `pyproject.toml`: Update `version = "x.y.z"` line
   - `README.md`: Update both Docker tagging examples (the `x.y.z` and `x.y`
     tags in section "Docker Tags")
6. Display summary: old version → new version, files updated
7. Commit the edits. Tag `vx.y.z` only if you are building release artifacts
   outside the workflow.
