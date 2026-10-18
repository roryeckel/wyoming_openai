"""Constants for the test-only HA E2E scripted conversation agent."""

from __future__ import annotations

from dataclasses import dataclass

DOMAIN = "ha_e2e_scripted"
SERVICE_SET_REPLY = "set_reply"
ATTR_REPLY = "reply"
ATTR_DELTA_DELAY_MS = "delta_delay_ms"
DATA_NEXT_REPLY = "next_reply"


@dataclass
class ScriptedReply:
    """The reply the agent should stream on its next turn."""

    text: str
    delta_delay_ms: int = 0
