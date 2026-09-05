import argparse
from enum import Enum
from io import BytesIO

import pytest

from wyoming_openai.utilities import (
    NamedBytesIO,
    SsmlTextTransformer,
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
    # An unmatched quote is an incomplete invalid construct and remains literal.
    assert strip_ssml('<speak>before<prosody rate="fast>after</prosody> &</speak>') == (
        'before<prosody rate="fast>after</prosody> &</speak>'
    )


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


def test_ssml_transformer_handles_entities_and_lexical_constructs():
    assert strip_ssml("Tom &amp; &#65; &#x42; &unknown;") == "Tom & A B &unknown;"
    assert strip_ssml("A<!-- comment > -->B<?pi value?>C") == "ABC"
    assert strip_ssml("A<![CDATA[<b>& c]]>D") == "A<b>& cD"


def test_ssml_transformer_preserves_incomplete_and_invalid_literals():
    assert strip_ssml("one<vendor") == "one<vendor"
    assert strip_ssml("one&am") == "one&am"
    assert strip_ssml("one<>two") == "one<>two"
    assert strip_ssml("one<vendor two") == "one<vendor two"


def test_ssml_transformer_handles_oversized_unfinished_token():
    text = "one<" + ("x" * 4096) + "two"
    transformer = SsmlTextTransformer()

    streamed = transformer.feed(text[:3]) + transformer.feed(text[3:]) + transformer.finish()

    assert streamed == text
    assert strip_ssml(text) == text


def test_ssml_transformer_delays_boundaries_until_text_or_finish():
    transformer = SsmlTextTransformer()

    assert transformer.feed("one<vendor/>") == "one"
    assert transformer.feed("two") == " two"
    assert transformer.finish() == ""

    transformer = SsmlTextTransformer()
    assert transformer.feed("one<break/>") == "one"
    assert transformer.finish() == " "


_SSML_PARTITION_CORPUS = (
    "<speak>Hello<break/>world</speak>",
    "<speak>Hello<break time='a>b'/>world</speak>",
    "one<vendor/>two",
    "<speak>a<emphasis>b</emphasis>c</speak>",
    "Tom &amp; Jerry &#x1f600;",
    "A<!-- comment > -->B<![CDATA[<raw>&]]>C<?pi?>D",
    "one<vendor",
    "one&am",
    '<speak>one<prosody rate="fast>two</speak>',
    "one  two",
)


@pytest.mark.parametrize("text", _SSML_PARTITION_CORPUS)
def test_ssml_transformer_is_invariant_to_chunk_partitions(text):
    expected = strip_ssml(text)
    split_patterns: list[tuple[int, ...]] = [(split,) for split in range(len(text) + 1)]
    if len(text) > 2:
        split_patterns.extend(((1, len(text) - 1), (len(text) // 3, 2 * len(text) // 3)))

    for split_pattern in split_patterns:
        transformer = SsmlTextTransformer()
        start = 0
        actual_parts: list[str] = []
        for split in (*split_pattern, len(text)):
            actual_parts.append(transformer.feed(text[start:split]))
            start = split
        actual = "".join(actual_parts) + transformer.finish()
        assert actual == expected, f"splits={split_pattern} text={text!r}"


def test_ssml_transformer_regression_split_speak_and_break():
    transformer = SsmlTextTransformer()

    actual = transformer.feed("<speak>Hello")
    actual += transformer.feed("<break/>world</speak>")
    actual += transformer.finish()

    assert actual == "Hello world"


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
