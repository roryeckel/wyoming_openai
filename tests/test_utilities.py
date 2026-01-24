import argparse
from enum import Enum
from io import BytesIO

import pytest

from wyoming_openai.utilities import NamedBytesIO, create_enum_parser


def test_named_bytes_io_name_property():
    buf = NamedBytesIO(b"abc", name="test.wav")
    assert buf.name == "test.wav"
    assert buf.read() == b"abc"

def test_named_bytes_io_default_name():
    buf = NamedBytesIO()
    assert buf.name == "audio.wav"

def test_named_bytes_io_inherits_bytesio():
    buf = NamedBytesIO(b"xyz", name="foo.wav")
    assert isinstance(buf, BytesIO)
    assert buf.read() == b"xyz"


# Test enum for create_enum_parser tests
class MockBackend(Enum):
    OPENAI = 1
    LOCAL = 2
    CUSTOM = 3


def test_create_enum_parser_valid_input():
    """Test that create_enum_parser successfully parses valid enum values."""
    parser = create_enum_parser(MockBackend)

    assert parser("openai") == MockBackend.OPENAI
    assert parser("OPENAI") == MockBackend.OPENAI
    assert parser("local") == MockBackend.LOCAL
    assert parser("custom") == MockBackend.CUSTOM


def test_create_enum_parser_invalid_input():
    """Test that create_enum_parser raises ArgumentTypeError for invalid values."""
    parser = create_enum_parser(MockBackend)

    with pytest.raises(argparse.ArgumentTypeError) as exc_info:
        parser("invalid")

    error_msg = str(exc_info.value)
    assert "Invalid MockBackend" in error_msg
    assert "invalid" in error_msg
    assert "OPENAI, LOCAL, CUSTOM" in error_msg


def test_create_enum_parser_case_sensitive():
    """Test that create_enum_parser respects case_insensitive parameter."""
    parser = create_enum_parser(MockBackend, case_insensitive=False)

    # Should work with exact case
    assert parser("OPENAI") == MockBackend.OPENAI

    # Should fail with wrong case
    with pytest.raises(argparse.ArgumentTypeError):
        parser("openai")


def test_create_enum_parser_with_argparse():
    """Test that create_enum_parser works correctly with argparse."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=create_enum_parser(MockBackend))

    args = parser.parse_args(["--backend", "openai"])
    assert args.backend == MockBackend.OPENAI

    # Test that invalid values are caught by argparse
    with pytest.raises(SystemExit):
        parser.parse_args(["--backend", "invalid"])
