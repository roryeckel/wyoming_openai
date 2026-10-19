"""Configuration for the Home Assistant end-to-end harness.

All values can be overridden with environment variables so the same suite can
target a local Docker Desktop stack or a CI runner.  Nothing here starts or
stops containers; the caller is expected to bring the compose stack up first.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent
FIXTURE_DIR = PACKAGE_DIR / "fixtures" / "audio"
HA_CONFIG_DIR = PACKAGE_DIR / "ha_config"
CUSTOM_COMPONENT_DIR = PACKAGE_DIR / "custom_components" / "ha_e2e_scripted"

DEFAULT_BASE_URL = "http://127.0.0.1:8123"
DEFAULT_SPEACHES_URL = "http://127.0.0.1:8000/v1"
DEFAULT_STT_MODEL = "Systran/faster-distil-whisper-small.en"
DEFAULT_HA_VERSION = "2026.10.1"
DEFAULT_WYOMING_HOST = "wyoming_openai"
DEFAULT_BAD_STT_HOST = "wyoming_openai_bad_stt"
DEFAULT_WYOMING_PORT = 10300
DEFAULT_PROXY_CONTAINER = "wyoming_openai_e2e"
DEFAULT_BAD_PROXY_CONTAINER = "wyoming_openai_bad_stt_e2e"
DEFAULT_HA_CONTAINER = "homeassistant_ha_e2e"

COMPOSE_FILES = (
    REPO_ROOT / "docker-compose.speaches-cpu.yml",
    REPO_ROOT / "docker-compose.e2e.yml",
    REPO_ROOT / "docker-compose.ha-e2e.yml",
)

RESET_COMMAND = (
    "docker compose -f docker-compose.speaches-cpu.yml -f docker-compose.e2e.yml "
    "-f docker-compose.ha-e2e.yml down -v"
)


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    return Path(value).expanduser().resolve()


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    return int(value)


@dataclass(frozen=True)
class HaE2ESettings:
    """Resolved harness settings."""

    base_url: str
    ws_url: str
    client_id: str
    redirect_uri: str
    state_dir: Path
    artifact_dir: Path
    fixture_dir: Path
    speaches_url: str
    stt_model: str
    ha_version: str
    wyoming_host: str
    bad_stt_host: str
    wyoming_port: int
    proxy_container: str
    bad_proxy_container: str
    ha_container: str
    compose_files: tuple[Path, ...]

    @property
    def credentials_path(self) -> Path:
        return self.state_dir / "credentials.json"

    @property
    def manifest_path(self) -> Path:
        return self.fixture_dir / "manifest.json"


def load_settings() -> HaE2ESettings:
    """Build settings from the environment."""

    base_url = _env_str("HA_E2E_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    client_id = _env_str("HA_E2E_CLIENT_ID", f"{base_url}/")
    redirect_uri = _env_str("HA_E2E_REDIRECT_URI", f"{base_url}/?auth_callback=1")
    return HaE2ESettings(
        base_url=base_url,
        ws_url=base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1) + "/api/websocket",
        client_id=client_id,
        redirect_uri=redirect_uri,
        state_dir=_env_path("HA_E2E_STATE_DIR", PACKAGE_DIR / ".state"),
        artifact_dir=_env_path("HA_E2E_ARTIFACT_DIR", PACKAGE_DIR / "artifacts"),
        fixture_dir=_env_path("HA_E2E_FIXTURE_DIR", FIXTURE_DIR),
        speaches_url=_env_str("HA_E2E_SPEACHES_URL", DEFAULT_SPEACHES_URL).rstrip("/"),
        stt_model=_env_str("HA_E2E_STT_MODEL", DEFAULT_STT_MODEL),
        ha_version=_env_str("HA_VERSION", DEFAULT_HA_VERSION),
        wyoming_host=_env_str("HA_E2E_WYOMING_HOST", DEFAULT_WYOMING_HOST),
        bad_stt_host=_env_str("HA_E2E_BAD_STT_HOST", DEFAULT_BAD_STT_HOST),
        wyoming_port=_env_int("HA_E2E_WYOMING_PORT", DEFAULT_WYOMING_PORT),
        proxy_container=_env_str("HA_E2E_PROXY_CONTAINER", DEFAULT_PROXY_CONTAINER),
        bad_proxy_container=_env_str("HA_E2E_BAD_PROXY_CONTAINER", DEFAULT_BAD_PROXY_CONTAINER),
        ha_container=_env_str("HA_E2E_HA_CONTAINER", DEFAULT_HA_CONTAINER),
        compose_files=COMPOSE_FILES,
    )
