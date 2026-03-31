import importlib.metadata
import logging

_LOGGER = logging.getLogger(__name__)

try:
    __version__ = importlib.metadata.version("wyoming_openai")
except importlib.metadata.PackageNotFoundError:
    _LOGGER.warning("Could not determine package version.")
    __version__ = "unknown"

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
ATTRIBUTION_NAME = "OpenAI-Compatible Wyoming Proxy"
ATTRIBUTION_URL = "https://github.com/roryeckel/wyoming_openai"

# Aliases used by tests and compatibility module
ATTRIBUTION_NAME_MODEL = ATTRIBUTION_NAME
ATTRIBUTION_NAME_PROGRAM = ATTRIBUTION_NAME
ATTRIBUTION_NAME_PROGRAM_STREAMING = f"{ATTRIBUTION_NAME} (Streaming)"
