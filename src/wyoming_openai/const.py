import importlib.metadata
import logging

_LOGGER = logging.getLogger(__name__)

try:
    __version__ = importlib.metadata.version("wyoming_openai")
except importlib.metadata.PackageNotFoundError:
    _LOGGER.warning("Could not determine package version. Using 'unknown'.")
    __version__ = "unknown"

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

# Attribution names for different Wyoming info levels
ATTRIBUTION_NAME_MODEL = "OpenAI-Compatible Wyoming Proxy"
ATTRIBUTION_NAME_PROGRAM = "OpenAI-Compatible Proxy"
ATTRIBUTION_NAME_PROGRAM_STREAMING = "OpenAI-Compatible Proxy (Streaming)"
ATTRIBUTION_URL = "https://github.com/roryeckel/wyoming_openai"
