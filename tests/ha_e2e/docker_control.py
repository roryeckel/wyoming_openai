"""Small, synchronous Docker helpers used by the failure lanes."""

from __future__ import annotations

import shutil
import socket
import subprocess
import time

DOCKER_COMMAND_TIMEOUT_SECONDS = 90


class DockerUnavailableError(RuntimeError):
    """Docker CLI is not installed."""


def docker_binary() -> str:
    docker = shutil.which("docker")
    if docker is None:
        raise DockerUnavailableError("docker is not available; the failure lanes need the Docker CLI")
    return docker


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [docker_binary(), *args],
        capture_output=True,
        text=True,
        timeout=DOCKER_COMMAND_TIMEOUT_SECONDS,
        check=False,
    )


def stop_container(name: str) -> None:
    """Send SIGTERM to a container and wait for it to exit."""

    result = _run("stop", "-t", "10", name)
    if result.returncode != 0:
        raise RuntimeError(f"docker stop {name} failed: {result.stderr.strip() or result.stdout.strip()}")


def start_container(name: str) -> None:
    """Start a stopped container."""

    result = _run("start", name)
    if result.returncode != 0:
        raise RuntimeError(f"docker start {name} failed: {result.stderr.strip() or result.stdout.strip()}")


def container_exit_code(name: str) -> int:
    """Return the last exit code of a container."""

    result = _run("inspect", "--format", "{{.State.ExitCode}}", name)
    if result.returncode != 0:
        raise RuntimeError(f"docker inspect {name} failed: {result.stderr.strip()}")
    return int(result.stdout.strip())


def container_logs(name: str) -> str:
    """Return the complete logs of a container."""

    result = _run("logs", name)
    if result.returncode != 0:
        raise RuntimeError(f"docker logs {name} failed: {result.stderr.strip()}")
    return (result.stdout or "") + (result.stderr or "")


def wait_for_log(name: str, needle: str, timeout: float = 20.0) -> str:
    """Wait until a container's logs contain ``needle``, then return the logs."""

    deadline = time.monotonic() + timeout
    last_logs = ""
    while time.monotonic() < deadline:
        last_logs = container_logs(name)
        if needle in last_logs:
            return last_logs
        time.sleep(0.5)
    snippet = "\n".join(last_logs.splitlines()[-40:])
    raise AssertionError(f"container {name} logs did not contain {needle!r} within {timeout}s\n{snippet}")


def wait_for_new_log(
    name: str,
    needle: str,
    previous_count: int,
    timeout: float = 20.0,
) -> str:
    """Wait for a new occurrence of ``needle`` after a previous log snapshot."""

    deadline = time.monotonic() + timeout
    last_logs = ""
    while time.monotonic() < deadline:
        last_logs = container_logs(name)
        if last_logs.count(needle) > previous_count:
            return last_logs
        time.sleep(0.5)
    snippet = "\n".join(last_logs.splitlines()[-40:])
    raise AssertionError(
        f"container {name} logs did not gain a new {needle!r} within {timeout}s\n{snippet}"
    )


def wait_for_tcp(host: str, port: int, timeout: float) -> None:
    """Wait until a TCP port accepts connections."""

    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.25)
    detail = f"; last error: {last_error}" if last_error else ""
    raise TimeoutError(f"TCP port {host}:{port} did not open within {timeout}s{detail}")
