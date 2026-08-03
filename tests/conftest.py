"""Shared pytest fixtures.

Most fixtures here are dedicated to the integration suite (`-m integration`).
The default `pytest` invocation skips that marker (see `addopts` in
pyproject.toml) and these fixtures do not load any heavy dependencies at
import time, so unit tests are unaffected.

Two modes:

- Default: spawn `python -m wyoming_openai` as a subprocess pointed at a
  Speaches backend (reused if already healthy on localhost:8000, otherwise
  started via docker compose, otherwise the suite skips).
- External: set WYOMING_E2E_URI=tcp://host:port to target an already-running
  proxy (e.g. the containerized smoke stack). No docker or subprocess
  management happens; tests that must control the server's environment skip.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Generator
from pathlib import Path

import pytest
import pytest_asyncio
from e2e_helpers import ARTIFACT_DIR_ENV_VAR, WyomingTestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
# The e2e overlay swaps in a smaller Whisper model; the base file keeps the
# general-purpose default for real deployments.
COMPOSE_FILES = [
    REPO_ROOT / "docker-compose.speaches-cpu.yml",
    REPO_ROOT / "docker-compose.e2e.yml",
]

EXTERNAL_URI_ENV_VAR = "WYOMING_E2E_URI"

SPEACHES_HEALTH_URL = "http://127.0.0.1:8000/health"
SPEACHES_BASE_URL = "http://127.0.0.1:8000/v1"
SPEACHES_HEALTH_TIMEOUT_SECONDS = 240
SPEACHES_HEALTH_POLL_INTERVAL_SECONDS = 2

STT_MODEL = os.environ.get("WYOMING_E2E_STT_MODEL", "Systran/faster-distil-whisper-small.en")
TTS_MODEL = os.environ.get("WYOMING_E2E_TTS_MODEL", "speaches-ai/Kokoro-82M-v1.0-ONNX")

WYOMING_STARTUP_TIMEOUT_SECONDS = 30
WYOMING_SHUTDOWN_GRACE_SECONDS = 5
# Deliberately strips every proxy-related var a dev shell might export
# (including e.g. STT_REALTIME_MODELS - realtime uses the OpenAI websocket API,
# which Speaches does not implement, so it is out of scope for this suite).
# Non-prefixed vars (PATH, SystemRoot, ...) are kept so the subprocess also
# works on Windows.
WYOMING_CONFIG_ENV_PREFIXES = ("STT_", "TTS_", "WYOMING_")


def _external_endpoint() -> tuple[str, int] | None:
    uri = os.environ.get(EXTERNAL_URI_ENV_VAR)
    if not uri:
        return None
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "tcp" or not parsed.hostname or not parsed.port:
        raise ValueError(f"{EXTERNAL_URI_ENV_VAR} must look like tcp://host:port, got {uri!r}")
    return parsed.hostname, parsed.port


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


def _compose_command(docker: str, *args: str) -> list[str]:
    cmd = [docker, "compose"]
    for compose_file in COMPOSE_FILES:
        cmd.extend(["-f", str(compose_file)])
    cmd.extend(args)
    return cmd


@pytest.fixture(scope="session")
def speaches_backend() -> Generator[str, None, None]:
    """Ensure a Speaches backend is reachable on localhost:8000.

    If one is already running, reuse it. Otherwise bring up the CPU
    docker-compose stack (with the e2e model overlay) and tear it down on
    session exit (unless WYOMING_OPENAI_KEEP_SPEACHES=1).
    """

    if _is_speaches_healthy():
        yield SPEACHES_BASE_URL
        return

    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("docker is not available; cannot start Speaches for integration tests")

    missing = [str(f) for f in COMPOSE_FILES if not f.exists()]
    if missing:
        pytest.skip(f"docker-compose file(s) missing: {', '.join(missing)}")

    subprocess.run(_compose_command(docker, "up", "-d", "speaches"), check=True, cwd=REPO_ROOT)

    try:
        _wait_for_speaches(SPEACHES_HEALTH_TIMEOUT_SECONDS)
        yield SPEACHES_BASE_URL
    finally:
        if os.environ.get("WYOMING_OPENAI_KEEP_SPEACHES") != "1":
            subprocess.run(_compose_command(docker, "down"), check=False, cwd=REPO_ROOT)


def _start_wyoming_server(env_overrides: dict[str, str]) -> tuple[subprocess.Popen[bytes], int, Path]:
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

    # Log to a file rather than a PIPE: nothing drains a PIPE, so a chatty
    # server would eventually fill the 64KB buffer and block mid-test.
    log_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
        mode="wb", prefix=f"wyoming-e2e-{port}-", suffix=".log", delete=False
    )
    log_path = Path(log_file.name)
    proc = subprocess.Popen(
        [sys.executable, "-m", "wyoming_openai"],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=REPO_ROOT,
    )
    log_file.close()
    try:
        _wait_for_tcp("127.0.0.1", port, WYOMING_STARTUP_TIMEOUT_SECONDS)
    except Exception:
        # The server log is the most relevant diagnostic when startup fails;
        # the fixture teardown that would normally preserve it never runs.
        _terminate(proc)
        _preserve_log(log_path, "wyoming-subprocess-startup-failure")
        raise
    return proc, port, log_path


def _terminate(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=WYOMING_SHUTDOWN_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=WYOMING_SHUTDOWN_GRACE_SECONDS)


def _preserve_log(log_path: Path, label: str) -> None:
    """Copy the server log into $WYOMING_E2E_ARTIFACT_DIR (if set), then remove it.

    The temp file is always deleted so local runs don't accumulate logs
    indefinitely; the artifact copy (CI) is what survives.
    """
    artifact_dir = os.environ.get(ARTIFACT_DIR_ENV_VAR)
    if artifact_dir and log_path.exists():
        dest = Path(artifact_dir)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(log_path, dest / f"{label}.log")
    # Best-effort: on Windows the file can stay locked briefly after the
    # process exits; never let cleanup fail teardown.
    for _ in range(5):
        try:
            log_path.unlink(missing_ok=True)
        except PermissionError:
            time.sleep(0.2)
        else:
            break


@pytest.fixture(scope="session")
def wyoming_endpoint(request: pytest.FixtureRequest) -> Generator[tuple[str, int], None, None]:
    """(host, port) of the proxy under test.

    External mode (WYOMING_E2E_URI set): just wait for the port and yield.
    Otherwise: ensure Speaches, spawn the proxy subprocess, yield, terminate.
    """
    external = _external_endpoint()
    if external is not None:
        _wait_for_tcp(*external, WYOMING_STARTUP_TIMEOUT_SECONDS)
        yield external
        return

    request.getfixturevalue("speaches_backend")
    proc, port, log_path = _start_wyoming_server(
        env_overrides={"STT_BACKEND": "SPEACHES", "TTS_BACKEND": "SPEACHES"}
    )
    try:
        yield ("127.0.0.1", port)
    finally:
        _terminate(proc)
        _preserve_log(log_path, "wyoming-subprocess")


@pytest.fixture(scope="session")
def wyoming_endpoint_autodetect(request: pytest.FixtureRequest) -> Generator[tuple[str, int], None, None]:
    """Proxy spawned without explicit backend env vars to exercise autodetection.

    Skips in external mode: an external server's environment cannot be controlled.
    """
    if _external_endpoint() is not None:
        pytest.skip("autodetection test requires a locally spawned server (WYOMING_E2E_URI is set)")

    request.getfixturevalue("speaches_backend")
    proc, port, log_path = _start_wyoming_server(env_overrides={})
    try:
        yield ("127.0.0.1", port)
    finally:
        _terminate(proc)
        _preserve_log(log_path, "wyoming-subprocess-autodetect")


@pytest_asyncio.fixture
async def wyoming_client(wyoming_endpoint: tuple[str, int]):
    host, port = wyoming_endpoint
    async with WyomingTestClient(host=host, port=port) as client:
        yield client


@pytest_asyncio.fixture
async def wyoming_client_autodetect(wyoming_endpoint_autodetect: tuple[str, int]):
    host, port = wyoming_endpoint_autodetect
    async with WyomingTestClient(host=host, port=port) as client:
        yield client
