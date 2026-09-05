import argparse
from enum import Enum
from io import BytesIO

import pytest

from wyoming_openai.utilities import (
    NamedBytesIO,
    create_enum_parser,
    create_json_object_parser,
    get_extra_body_boolean_field,
    strip_ssml,
    validate_stt_extra_body,
    validate_tts_extra_body,
)


def test_strip_ssml_strips_break_tags():
    assert strip_ssml('<speak>Hello <break time="500ms"/>world</speak>') == "Hello world"
    # Malformed markup (unclosed tag) takes the regex fallback but still loses tags
    assert strip_ssml("Hello <break>world") == "Hello world"
    # Plain text round-trips unchanged
    assert strip_ssml("Hello world.") == "Hello world."


def test_strip_ssml_handles_gt_in_quoted_attributes():
    # A quoted XML attribute may contain ">"; it must not truncate the tag
    # before the XML parser sees it.
    assert strip_ssml('<speak>Hello<break foo="a>b"/>world</speak>') == "Hello world"
    # The malformed-markup fallback uses the same quote-aware tag scanner.
    assert strip_ssml('<speak>Hello<break foo="a>b"/>world &</speak>') == "Hello world &"


def test_strip_ssml_fallback_recovers_from_unterminated_attribute_quote():
    # The unmatched quote forces fallback; its tag and subsequent closing tags
    # must still be removed.
    assert strip_ssml('<speak>before<prosody rate="fast>after</prosody> &</speak>') == "beforeafter &"


def test_strip_ssml_preserves_word_boundaries():
    # Pause/block elements separate words even without surrounding whitespace
    assert strip_ssml("<speak>Hello<break/>world</speak>") == "Hello world"
    assert strip_ssml("<speak><p>one</p><p>two</p></speak>").strip() == "one two"
    assert strip_ssml("<speak><s>First.</s><s>Second.</s></speak>").strip() == "First. Second."
    # Inline elements must not introduce extra spaces
    assert strip_ssml("<speak>a<emphasis>b</emphasis>c</speak>") == "abc"
    assert strip_ssml('<speak><prosody rate="slow">slow</prosody> text</speak>') == "slow text"


def test_synthesize_ssml_decodes_entities():
    # Well-formed entities are decoded by the XML parser
    assert strip_ssml("Tom &amp; Jerry") == "Tom & Jerry"
    # A bare ampersand breaks XML parsing; the fallback path unescapes what it can
    assert strip_ssml("<speak>AT&T &amp; friends</speak>") == "AT&T & friends"


def test_strip_ssml_fallback_preserves_word_boundaries():
    # Malformed markup takes the regex fallback; tags must not join adjacent words
    assert strip_ssml('<speak>Foo<bad-attr=&>bar</speak>') == "Foo bar"


def test_strip_ssml_fallback_keeps_inline_tags_flowing():
    # A bare ampersand forces the fallback, but known inline elements must not
    # introduce pauses between the text they wrap
    assert strip_ssml("<speak>a<emphasis>b</emphasis>c &</speak>") == "abc &"
    assert strip_ssml('<speak>slow<prosody rate="slow">s</prosody>t</speak>') == "slowst"


def test_strip_ssml_fallback_preserves_literal_whitespace():
    # Whitespace around words is text content, not tag-generated
    assert strip_ssml("<speak> hello & goodbye </speak>") == " hello & goodbye "


def test_strip_ssml_fallback_adjacent_unknown_tags_join_words():
    # A contiguous run of unknown tags counts as a single separator unit
    assert strip_ssml("Foo<vendor></vendor>bar &") == "Foo bar &"


def test_strip_ssml_fallback_mixed_tag_run():
    # A contiguous run mixing known inline and unknown tags still separates the words
    assert strip_ssml("a<emphasis><vendor/></emphasis>b &") == "a b &"
    # A run of only known inline elements keeps its text flowing
    assert strip_ssml("a<emphasis><prosody></prosody></emphasis>b &") == "ab &"


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


def test_create_json_object_parser_valid_input():
    """Test that create_json_object_parser parses JSON objects."""
    parser = create_json_object_parser("TTS extra body")

    assert parser('{"stream": true, "nested": {"enabled": false}}') == {
        "stream": True,
        "nested": {"enabled": False},
    }


def test_create_json_object_parser_rejects_invalid_json():
    """Test that create_json_object_parser rejects invalid JSON."""
    parser = create_json_object_parser("STT extra body")

    with pytest.raises(argparse.ArgumentTypeError) as exc_info:
        parser('{"stream": true')

    assert "Invalid STT extra body" in str(exc_info.value)


def test_create_json_object_parser_rejects_non_object():
    """Test that create_json_object_parser rejects non-object JSON values."""
    parser = create_json_object_parser("TTS extra body")

    with pytest.raises(argparse.ArgumentTypeError) as exc_info:
        parser('["stream"]')

    assert "expected a JSON object" in str(exc_info.value)


def test_validate_stt_extra_body_allows_boolean_stream_override():
    """Test that STT extra_body accepts a boolean stream override."""
    validate_stt_extra_body({"response_format": "json", "stream": True})


def test_validate_stt_extra_body_rejects_non_boolean_stream_override():
    """Test that STT extra_body rejects non-boolean stream values."""
    with pytest.raises(ValueError, match="STT extra_body stream must be a boolean"):
        validate_stt_extra_body({"stream": "yes"})


def test_validate_tts_extra_body_rejects_transport_overrides():
    """Test that TTS extra_body rejects transport-shaping fields."""
    with pytest.raises(ValueError, match="does not support overriding 'stream', 'stream_format'"):
        validate_tts_extra_body({"stream": True, "stream_format": "sse"})


def test_get_extra_body_boolean_field_returns_default_or_override():
    """Test that boolean extra_body fields fall back correctly."""
    assert get_extra_body_boolean_field(None, field_name="stream", default=False, body_name="STT") is False
    assert (
        get_extra_body_boolean_field({"stream": True}, field_name="stream", default=False, body_name="STT")
        is True
    )
