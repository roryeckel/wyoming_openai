"""Unit tests for Home Assistant API helpers that do not need a live stack."""

from __future__ import annotations

from typing import Any

import pytest

from ha_e2e import ha_api


class _FakeConfigFlowRest:
    """Return scripted responses from the config-flow REST endpoints."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.flow_starts = 0
        self.submissions = 0

    def start_config_flow(self, handler: str) -> dict[str, Any]:
        self.flow_starts += 1
        return {"type": "form", "flow_id": f"flow-{self.flow_starts}"}

    def submit_config_flow(self, flow_id: str, data: dict[str, Any]) -> dict[str, Any]:
        self.submissions += 1
        return self._responses.pop(0)


def test_create_entry_via_flow_retries_connection_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ha_api.time, "sleep", lambda _seconds: None)
    rest: Any = _FakeConfigFlowRest(
        [
            {"type": "form", "errors": {"base": "cannot_connect"}},
            {"type": "create_entry", "result": {"entry_id": "entry-1"}},
        ]
    )

    result = ha_api._create_entry_via_flow(rest, "wyoming", {"host": "proxy", "port": 10300})

    assert result["result"]["entry_id"] == "entry-1"
    assert rest.submissions == 2
