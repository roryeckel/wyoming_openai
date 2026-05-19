"""Shared pytest fixtures.

Most fixtures here are dedicated to the integration suite (`-m integration`).
The default `pytest` invocation skips that marker (see `addopts` in
pyproject.toml) and these fixtures do not load any heavy dependencies at
import time, so unit tests are unaffected.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Generator
from pathlib import Path

import pytest
import pytest_asyncio
from e2e_helpers import WyomingTestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = REPO_ROOT / "docker-compose.speaches-cpu.yml"

SPEACHES_HEALTH_URL = "http://127.0.0.1:8000/health"
SPEACHES_BASE_URL = "http://127.0.0.1:8000/v1"
SPEACHES_HEALTH_TIMEOUT_SECONDS = 240
SPEACHES_HEALTH_POLL_INTERVAL_SECONDS = 2

STT_MODEL = "Systran/faster-distil-whisper-large-v3"
TTS_MODEL = "speaches-ai/Kokoro-82M-v1.0-ONNX"

WYOMING_STARTUP_TIMEOUT_SECONDS = 30
WYOMING_SHUTDOWN_GRACE_SECONDS = 5
WYOMING_CONFIG_ENV_PREFIXES = ("STT_", "TTS_", "WYOMING_")


def _is_speaches_healthy() -> bool:
    try:
        with urllib.request.urlopen(SPEACHES_HEALTH_URL, timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return False


def _wait_for_speaches(timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(SPEACHES_HEALTH_URL, timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            last_error = exc
        time.sleep(SPEACHES_HEALTH_POLL_INTERVAL_SECONDS)
    detail = f"; last error: {last_error}" if last_error else ""
    raise TimeoutError(f"Speaches did not become healthy within {timeout}s{detail}")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_tcp(host: str, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.2)
    detail = f"; last error: {last_error}" if last_error else ""
    raise TimeoutError(f"Wyoming server did not accept connections at {host}:{port} within {timeout}s{detail}")


@pytest.fixture(scope="session")
def speaches_backend() -> Generator[str, None, None]:
    """Ensure a Speaches backend is reachable on localhost:8000.

    If one is already running, reuse it. Otherwise bring up the CPU
    docker-compose stack and tear it down on session exit (unless
    WYOMING_OPENAI_KEEP_SPEACHES=1).
    """

    if _is_speaches_healthy():
        yield SPEACHES_BASE_URL
        return

    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("docker is not available; cannot start Speaches for integration tests")

    if not COMPOSE_FILE.exists():
        pytest.skip(f"docker-compose file missing: {COMPOSE_FILE}")

    up_cmd = [docker, "compose", "-f", str(COMPOSE_FILE), "up", "-d", "speaches"]
    subprocess.run(up_cmd, check=True, cwd=REPO_ROOT)

    try:
        _wait_for_speaches(SPEACHES_HEALTH_TIMEOUT_SECONDS)
        yield SPEACHES_BASE_URL
    finally:
        if os.environ.get("WYOMING_OPENAI_KEEP_SPEACHES") != "1":
            down_cmd = [docker, "compose", "-f", str(COMPOSE_FILE), "down"]
            subprocess.run(down_cmd, check=False, cwd=REPO_ROOT)


def _start_wyoming_server(env_overrides: dict[str, str]) -> tuple[subprocess.Popen[bytes], int]:
    port = _find_free_port()
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(WYOMING_CONFIG_ENV_PREFIXES)
    }
    env.update(
        {
            "WYOMING_URI": f"tcp://127.0.0.1:{port}",
            "WYOMING_LOG_LEVEL": "INFO",
            "WYOMING_LANGUAGES": "en",
            "STT_OPENAI_URL": SPEACHES_BASE_URL,
            "STT_MODELS": STT_MODEL,
            "TTS_OPENAI_URL": SPEACHES_BASE_URL,
            "TTS_MODELS": TTS_MODEL,
        }
    )
    env.update(env_overrides)
    proc = subprocess.Popen(
        [sys.executable, "-m", "wyoming_openai"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=REPO_ROOT,
    )
    try:
        _wait_for_tcp("127.0.0.1", port, WYOMING_STARTUP_TIMEOUT_SECONDS)
    except Exception:
        _terminate(proc)
        raise
    return proc, port


def _terminate(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=WYOMING_SHUTDOWN_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=WYOMING_SHUTDOWN_GRACE_SECONDS)


@pytest.fixture(scope="session")
def wyoming_server(speaches_backend: str) -> Generator[int, None, None]:
    """Run wyoming_openai as a subprocess pointed at the Speaches backend.

    Yields the TCP port the server is listening on.
    """
    proc, port = _start_wyoming_server(env_overrides={"STT_BACKEND": "SPEACHES", "TTS_BACKEND": "SPEACHES"})
    try:
        yield port
    finally:
        _terminate(proc)


@pytest.fixture(scope="session")
def wyoming_server_autodetect(speaches_backend: str) -> Generator[int, None, None]:
    """Run wyoming_openai without explicit backend env vars to exercise autodetection."""
    proc, port = _start_wyoming_server(env_overrides={})
    try:
        yield port
    finally:
        _terminate(proc)


@pytest_asyncio.fixture
async def wyoming_client(wyoming_server: int):
    async with WyomingTestClient(host="127.0.0.1", port=wyoming_server) as client:
        yield client


@pytest_asyncio.fixture
async def wyoming_client_autodetect(wyoming_server_autodetect: int):
    async with WyomingTestClient(host="127.0.0.1", port=wyoming_server_autodetect) as client:
        yield client
