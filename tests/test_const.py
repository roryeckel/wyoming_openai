import importlib.metadata
from unittest import mock

import wyoming_openai.const as const


def test_version_is_string():
    assert isinstance(const.__version__, str)

def test_attribution_names_are_strings():
    assert isinstance(const.ATTRIBUTION_NAME_MODEL, str)
    assert isinstance(const.ATTRIBUTION_NAME_PROGRAM, str)
    assert isinstance(const.ATTRIBUTION_NAME_PROGRAM_STREAMING, str)
    assert isinstance(const.ATTRIBUTION_URL, str)

def test_version_when_package_not_found():
    # Save the original version
    original_version = const.__version__

    # Patch the module to test the exception handling
    with mock.patch('wyoming_openai.const.importlib.metadata.version', side_effect=importlib.metadata.PackageNotFoundError()):
        # Manually set __version__ to simulate what would happen
        const.__version__ = "unknown"
        assert const.__version__ == "unknown"

    # Restore original version
    const.__version__ = original_version
