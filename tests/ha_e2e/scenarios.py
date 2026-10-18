"""Scripted conversation turns and the fixture manifest contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import HaE2ESettings


@dataclass(frozen=True)
class AudioFixture:
    """One committed fixture WAV."""

    id: str
    file: str
    text: str
    sha256: str
    path: Path


@dataclass(frozen=True)
class Scenario:
    """One scripted conversation turn."""

    turn: int
    id: str
    fixture_id: str
    expected_text: str
    reply: str
    delta_delay_ms: int
    expect_streaming: bool
    long_reply: bool = False


LONG_REPLY = (
    "Here are several practical ways to reduce energy use at home without making the house uncomfortable. "
    "Start by sealing drafts around doors and windows, because even a small gap can waste a surprising amount "
    "of heat over a winter. Next, replace the oldest bulbs with efficient ones and put lamps on timers so empty "
    "rooms are not lit all evening. Wash only full loads in the dishwasher and the washing machine, and use the "
    "coldest water setting that still gets the job done. Lower the water heater thermostat a few degrees and "
    "insulate the first few feet of hot and cold pipes; that usually saves more than it costs. When you replace "
    "an appliance, compare its yearly energy estimate rather than only its purchase price. A heat pump dryer or "
    "a modern refrigerator can pay back the difference over its lifetime. Use smart plugs for televisions, game "
    "consoles, and chargers that draw power in standby, and put them on a schedule that matches your routine. "
    "Close blinds on hot afternoons in summer and open them on sunny winter mornings to let free heat in. Keep "
    "the refrigerator coils clean and leave space around the unit so it does not work harder than necessary. "
    "If you have central heating, seal and insulate the ducts in unconditioned spaces, and have the system "
    "serviced annually. A programmable thermostat with a modest setback overnight is often the easiest saving "
    "of all, especially when the house is empty during the day. Consider a home energy audit if your utility "
    "offers one; a technician with a thermal camera can find insulation gaps that are hard to see. Finally, "
    "track your usage for a month before and after each change, so you know which improvements actually helped "
    "and which ones are worth repeating. Small, consistent changes add up faster than one expensive upgrade, "
    "and they make the house quieter and more comfortable at the same time. Thank you for listening to this "
    "long explanation; the important part is to start with the cheapest fixes and measure the results."
)

FAILURE_REPLY = (
    LONG_REPLY + " This final sentence is deliberately added so the proxy has time to stop during a stream."
)


def load_manifest(settings: HaE2ESettings) -> dict[str, AudioFixture]:
    """Load and validate the committed fixture manifest."""

    manifest_path = settings.manifest_path
    try:
        payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"cannot read fixture manifest {manifest_path}: {exc}. "
            "Run tests/ha_e2e/generate_fixtures.py with Speaches running."
        ) from exc

    fixtures: dict[str, AudioFixture] = {}
    for item in payload.get("fixtures", []):
        fixture = AudioFixture(
            id=str(item["id"]),
            file=str(item["file"]),
            text=str(item["text"]),
            sha256=str(item["sha256"]),
            path=settings.fixture_dir / str(item["file"]),
        )
        fixtures[fixture.id] = fixture
    if not fixtures:
        raise RuntimeError(f"fixture manifest {manifest_path} contains no fixtures")
    return fixtures


def load_turn_scenarios(settings: HaE2ESettings) -> list[Scenario]:
    """Build the three conversation turns from the fixture manifest."""

    fixtures = load_manifest(settings)
    return [
        Scenario(
            turn=1,
            id="multi-sentence-numbers",
            fixture_id="kitchen_lights",
            expected_text=fixtures["kitchen_lights"].text,
            reply=(
                "I turned on the kitchen lights. There are 3 apples, 42 oranges, and 7 pears in the basket. "
                "The oven is set to 350 degrees."
            ),
            delta_delay_ms=0,
            expect_streaming=True,
        ),
        Scenario(
            turn=2,
            id="long-paused-stream",
            fixture_id="weather_berlin",
            expected_text=fixtures["weather_berlin"].text,
            reply=LONG_REPLY,
            delta_delay_ms=150,
            expect_streaming=True,
            long_reply=True,
        ),
        Scenario(
            turn=3,
            id="one-word",
            fixture_id="timer_five_minutes",
            expected_text=fixtures["timer_five_minutes"].text,
            reply="Done.",
            delta_delay_ms=0,
            expect_streaming=False,
        ),
    ]


def failure_scenario(settings: HaE2ESettings) -> Scenario:
    """A long reply used while the proxy container is stopped."""

    fixtures = load_manifest(settings)
    return Scenario(
        turn=99,
        id="proxy-stop",
        fixture_id="timer_five_minutes",
        expected_text=fixtures["timer_five_minutes"].text,
        reply=FAILURE_REPLY,
        delta_delay_ms=150,
        expect_streaming=True,
        long_reply=True,
    )
