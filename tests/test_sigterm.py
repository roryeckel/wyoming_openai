"""Tests for graceful shutdown on SIGTERM (wyoming v1.10.1).

wyoming's AsyncServer now registers a loop signal handler while run() is
active: SIGTERM stops the event handlers, closes the server, and lets run()
return normally, so the process exits with code 0 instead of dying via the
default signal action (exit code 143). Home Assistant's Supervisor flags
add-ons that do not trap SIGTERM, so this locks the fix in for the proxy.

The signal handler is only supported on POSIX event loops; on Windows the
library leaves behavior unchanged, so the test skips there.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time

import pytest
from conftest import (
    WYOMING_SHUTDOWN_GRACE_SECONDS,
    _preserve_log,
    _start_wyoming_server,
    _terminate,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="SIGTERM graceful shutdown is only supported on POSIX event loops",
)


def test_server_exits_zero_on_sigterm() -> None:
    """SIGTERM stops the server cleanly with exit code 0, not 143."""
    proc, port, log_path = _start_wyoming_server(
        env_overrides={
            # Point at the official OpenAI domain so backend autodetection
            # skips its network probes, and provide voices so the backend
            # voice listing is skipped too: the server must start fully
            # offline for this lifecycle test.
            "STT_OPENAI_URL": "https://api.openai.com/v1",
            "TTS_OPENAI_URL": "https://api.openai.com/v1",
            "TTS_VOICES": "alloy",
        }
    )
    try:
        # The TCP listener accepts connections a moment before run() registers
        # the signal handler; settle so SIGTERM cannot hit the default action.
        time.sleep(0.5)
        proc.send_signal(signal.SIGTERM)
        try:
            returncode = proc.wait(timeout=WYOMING_SHUTDOWN_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=WYOMING_SHUTDOWN_GRACE_SECONDS)
            pytest.fail("server did not exit within the grace period after SIGTERM")
        assert returncode == 0
    finally:
        _terminate(proc)
        _preserve_log(log_path, "wyoming-subprocess-sigterm")
