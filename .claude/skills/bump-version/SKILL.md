---
name: bump-version
description: Bump the version number of wyoming-openai.
argument-hint: "[patch|minor|major|estimate]"
disable-model-invocation: true
allowed-tools: Read, Edit, AskUserQuestion, Bash, Grep
---

Bump version $ARGUMENTS.

Steps:
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
   - `README.md`: Update version in Docker tagging section
6. Display summary: old version → new version, files updated
7. Remind user to commit and tag: `git tag vx.y.z`
