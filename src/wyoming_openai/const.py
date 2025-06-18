import importlib.metadata
import logging

_LOGGER = logging.getLogger(__name__)

try:
    __version__ = importlib.metadata.version("wyoming_openai")
except importlib.metadata.PackageNotFoundError:
    _LOGGER.warning("Could not determine package version. Using 'unknown'.")
    __version__ = "unknown"
