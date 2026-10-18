"""Artifacts and the short markdown report for the HA end-to-end harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from .config import HaE2ESettings
from .helpers import short_text


@dataclass
class TurnRecord:
    """One completed or failed conversation turn."""

    turn: int
    scenario: str
    expected_text: str
    stt_text: str = ""
    stt_ok: bool = False
    stt_score: float | None = None
    stt_coverage: float | None = None
    reply_chars: int = 0
    reply_score: float | None = None
    reply_coverage: float | None = None
    transcript: str = ""
    duration_seconds: float | None = None
    time_to_first_audio: float | None = None
    artifacts: list[str] = field(default_factory=list)
    passed: bool = False
    error: str | None = None


@dataclass
class FailureRecord:
    """One failure-lane result."""

    name: str
    passed: bool
    detail: str
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None


class RunReport:
    """Write a run directory and its markdown summary."""

    def __init__(self, settings: HaE2ESettings, ha_version: str) -> None:
        self.settings = settings
        self.ha_version = ha_version
        self.started_at = datetime.now(UTC)
        stamp = self.started_at.strftime("%Y%m%d-%H%M%S")
        self.run_dir = settings.artifact_dir / f"run-{stamp}"
        suffix = 1
        while self.run_dir.exists():
            suffix += 1
            self.run_dir = settings.artifact_dir / f"run-{stamp}-{suffix}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "report.md"
        self.turns: list[TurnRecord] = []
        self.failures: list[FailureRecord] = []

    def add_turn(self, record: TurnRecord) -> None:
        self.turns.append(record)
        self.write()

    def add_failure(self, record: FailureRecord) -> None:
        self.failures.append(record)
        self.write()

    def write(self) -> None:
        lines = [
            "# Home Assistant E2E report",
            "",
            f"- Started: {self.started_at.isoformat()}",
            f"- Home Assistant: {self.ha_version}",
            f"- HA base URL: {self.settings.base_url}",
            f"- Speaches STT model: `{self.settings.stt_model}`",
            f"- Proxy container: `{self.settings.proxy_container}`",
            "",
            "## Conversation turns",
            "",
        ]
        if self.turns:
            lines.extend(
                [
                    "| Turn | Scenario | STT match | Reply chars | Audio (s) | TTFA (s) | Round-trip | Verdict |",
                    "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
                ]
            )
            for record in self.turns:
                stt = f"{record.stt_score:.1f}/{record.stt_coverage:.2f}" if record.stt_score is not None else "-"
                round_trip = (
                    f"{record.reply_score:.1f}/{record.reply_coverage:.2f}" if record.reply_score is not None else "-"
                )
                duration = f"{record.duration_seconds:.2f}" if record.duration_seconds is not None else "-"
                ttfa = f"{record.time_to_first_audio:.2f}" if record.time_to_first_audio is not None else "-"
                verdict = "PASS" if record.passed else "FAIL"
                lines.append(
                    f"| {record.turn} | {record.scenario} | {stt} | {record.reply_chars} | {duration} | "
                    f"{ttfa} | {round_trip} | {verdict} |"
                )
        else:
            lines.append("_No conversation turns recorded._")

        lines.extend(["", "## Failure lanes", ""])
        if self.failures:
            lines.extend(["| Case | Verdict | Detail |", "| --- | --- | --- |"])
            for failure in self.failures:
                verdict = "PASS" if failure.passed else "FAIL"
                lines.append(f"| {failure.name} | {verdict} | {short_text(failure.detail, 160)} |")
        else:
            lines.append("_No failure lanes recorded._")

        lines.extend(["", "## Details", ""])
        for record in self.turns:
            lines.append(f"### Turn {record.turn}: {record.scenario}")
            lines.append("")
            lines.append(f"- Expected STT: {short_text(record.expected_text, 300)}")
            lines.append(f"- STT returned: {short_text(record.stt_text, 300) or '(none)'}")
            lines.append(f"- Reply: {record.reply_chars} characters")
            lines.append(f"- Round-trip transcript: {short_text(record.transcript, 500) or '(none)'}")
            lines.append(f"- Artifacts: {', '.join(record.artifacts) or '(none)'}")
            if record.error:
                lines.append(f"- Error: {short_text(record.error, 1000)}")
            lines.append("")

        for failure in self.failures:
            lines.append(f"### Failure lane: {failure.name}")
            lines.append("")
            lines.append(f"- Detail: {short_text(failure.detail, 1000)}")
            lines.append(f"- Artifacts: {', '.join(failure.artifacts) or '(none)'}")
            if failure.error:
                lines.append(f"- Error: {short_text(failure.error, 1000)}")
            lines.append("")

        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
