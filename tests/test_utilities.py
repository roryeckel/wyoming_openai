import argparse
import math
from enum import Enum
from io import BytesIO

import pytest
from pcm_reference import assert_samples_close, reference_resample, sine_samples

from wyoming_openai.utilities import (
    NamedBytesIO,
    RealtimeAudioConverter,
    RealtimePcmResampler,
    SsmlTextTransformer,
    clamp_realtime_tts_speed,
    create_enum_parser,
    create_json_object_parser,
    decode_pcm_samples,
    downmix_pcm_samples,
    get_extra_body_boolean_field,
    get_realtime_tts_audio_output,
    get_realtime_tts_speed,
    parse_finite_float,
    parse_non_negative_int,
    parse_voice_labels,
    strip_ssml,
    validate_audio_format,
    validate_realtime_stt_extra_body,
    validate_realtime_tts_extra_body,
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


def test_parse_voice_labels_returns_trimmed_labels_by_voice_name():
    """Test that voice labels are parsed from a JSON object and surrounding whitespace is dropped."""
    assert parse_voice_labels('{" af_heart ": " Heart ", "af_bella": "Bella (warm)"}') == {
        "af_heart": "Heart",
        "af_bella": "Bella (warm)",
    }


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ('{"af_heart": "Heart"', "Invalid TTS voice labels"),
        ('["af_heart"]', "expected a JSON object"),
        ('{"af_heart": 3}', "label for 'af_heart' must be a non-empty string"),
        ('{"af_heart": null}', "label for 'af_heart' must be a non-empty string"),
        ('{"af_heart": "  "}', "label for 'af_heart' must be a non-empty string"),
        ('{"  ": "Heart"}', "voice names must be non-empty strings"),
    ],
)
def test_parse_voice_labels_rejects_invalid_values(value, message):
    """Test that voice labels must be a JSON object of non-empty strings."""
    with pytest.raises(argparse.ArgumentTypeError) as exc_info:
        parse_voice_labels(value)

    assert message in str(exc_info.value)


@pytest.mark.parametrize("value", ["inf", "-inf", "nan", "fast"])
def test_parse_finite_float_rejects_values_a_request_cannot_carry(value):
    assert parse_finite_float("1.5") == 1.5
    with pytest.raises(argparse.ArgumentTypeError):
        parse_finite_float(value)


@pytest.mark.parametrize(("value", "expected"), [("0", 0), ("1", 1), ("2048", 2048)])
def test_parse_non_negative_int_accepts_whole_numbers(value, expected):
    assert parse_non_negative_int(value) == expected


@pytest.mark.parametrize("value", ["-1", "1.5", "two", ""])
def test_parse_non_negative_int_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError, match="at least 0"):
        parse_non_negative_int(value)


def test_validate_stt_extra_body_allows_boolean_stream_override():
    """Test that STT extra_body accepts a boolean stream override."""
    validate_stt_extra_body({"response_format": "json", "stream": True})


@pytest.mark.parametrize("languages", ["en", [], ["en", 1], [""], None])
def test_validate_stt_extra_body_rejects_languages_that_are_not_language_codes(languages):
    """A `languages` override replaces the request language, so a wrong one is reported at startup."""
    validate_stt_extra_body({"languages": ["en", "de"]})
    with pytest.raises(ValueError, match="STT extra_body languages must be a list of language codes"):
        validate_stt_extra_body({"languages": languages})


def test_validate_realtime_stt_extra_body_checks_languages_and_allows_dropping_them():
    """In the Realtime body a null drops the field, as for any other field there."""
    validate_realtime_stt_extra_body({"languages": None})
    with pytest.raises(ValueError, match="STT Realtime extra_body languages must be a list of language codes"):
        validate_realtime_stt_extra_body({"languages": "en"})


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


@pytest.mark.parametrize(
    "extra_body",
    [
        None,
        {"reasoning": {"effort": "low"}},
        {"audio": {"output": {"format": {"type": "audio/pcm"}, "speed": 1}}},
        {"audio": {"output": {"format": {"type": "audio/pcm", "rate": 24000}}}},
        # A null is the same as leaving the object out
        {"audio": None, "reasoning": {"effort": "low"}},
        {"audio": {"output": None}},
    ],
)
def test_validate_realtime_tts_extra_body_allows_compatible_overrides(extra_body):
    """Realtime TTS accepts overrides that keep 24 kHz PCM output."""
    validate_realtime_tts_extra_body(extra_body)


@pytest.mark.parametrize(
    ("extra_body", "message"),
    [
        ({"audio": {"output": {"format": {"type": "audio/pcmu"}}}}, r"audio\.output\.format"),
        ({"audio": {"output": {"format": {"type": "audio/pcm", "rate": 16000}}}}, r"audio\.output\.format"),
        ({"audio": {"output": {"format": "pcm16"}}}, r"audio\.output\.format"),
        ({"audio": "pcm"}, "audio must be an object"),
        ({"audio": False}, "audio must be an object"),
        ({"audio": {"output": 0}}, "audio must be an object"),
        ({"audio": {"output": {"speed": "fast"}}}, "speed must be a finite number"),
        ({"audio": {"output": {"speed": True}}}, "speed must be a finite number"),
        ({"audio": {"output": {"speed": float("nan")}}}, "speed must be a finite number"),
        # An int too large for a float is rejected like any other unusable number
        ({"audio": {"output": {"speed": 10**400}}}, "speed must be a finite number"),
        ({"audio": {"output": {"voice": "cedar"}}}, r"audio\.output\.voice"),
        # The pre-GA session shape would be rejected by the API on every request
        ({"voice": "cedar"}, "'voice'"),
        ({"speed": 1.2}, r"'speed'.*audio\.output\.speed"),
        ({"output_modalities": ["text"]}, "'output_modalities'"),
        ({"type": "transcription", "tools": []}, "'tools', 'type'"),
        ({"tool_choice": "required"}, "'tool_choice'"),
        # The Wyoming client selects the model, and the read-aloud instructions are not replaceable
        ({"model": "gpt-realtime"}, "'model'"),
        ({"instructions": "Answer the user"}, "'instructions'.*TTS_INSTRUCTIONS"),
    ],
)
def test_validate_realtime_tts_extra_body_rejects_incompatible_overrides(extra_body, message):
    """Realtime TTS rejects overrides that would change the audio Wyoming is told to expect."""
    with pytest.raises(ValueError, match=message):
        validate_realtime_tts_extra_body(extra_body)


def test_validate_realtime_stt_extra_body_rejects_model_override():
    """Realtime STT keeps the model the Wyoming client selected."""
    validate_realtime_stt_extra_body(None)
    validate_realtime_stt_extra_body({"keywords": ["Wyoming"], "languages": ["en"]})
    with pytest.raises(ValueError, match="'model'"):
        validate_realtime_stt_extra_body({"model": "gpt-live-transcribe"})


def test_realtime_tts_speed_prefers_audio_output_override_and_clamps():
    """The Realtime speed comes from audio.output.speed, then the TTS speed, and is kept in range."""
    assert get_realtime_tts_speed(None, None) is None
    assert get_realtime_tts_speed(1.2, None) == 1.2
    assert get_realtime_tts_speed(1.2, {"audio": {"output": {"speed": 0.5}}}) == 0.5
    assert get_realtime_tts_speed(1.2, {"audio": {"output": {"voice": "cedar"}}}) == 1.2
    # A speed that is not a finite number is never sent
    assert get_realtime_tts_speed(float("nan"), None) is None
    assert get_realtime_tts_speed(float("inf"), None) is None
    assert clamp_realtime_tts_speed(3.0) == 1.5
    assert clamp_realtime_tts_speed(0.1) == 0.25
    assert clamp_realtime_tts_speed(1.0) == 1.0


def test_get_realtime_tts_audio_output_returns_a_copy_or_empty():
    """The audio.output object is copied, and anything else reads as unset."""
    configured_output = {"voice": "cedar"}
    extra_body: dict[str, object] = {"audio": {"output": configured_output}}

    output = get_realtime_tts_audio_output(extra_body)
    output["voice"] = "marin"

    assert configured_output == {"voice": "cedar"}
    assert get_realtime_tts_audio_output(None) == {}
    assert get_realtime_tts_audio_output({"audio": "pcm"}) == {}
    assert get_realtime_tts_audio_output({"audio": {"output": None}}) == {}


def test_realtime_tts_speed_null_override_falls_back_to_tts_speed():
    """Test a null audio.output.speed is treated as no override instead of dropping TTS_SPEED."""
    assert get_realtime_tts_speed(1.3, {"audio": {"output": {"speed": None}}}) == 1.3
    assert get_realtime_tts_speed(None, {"audio": {"output": {"speed": None}}}) is None


def test_audio_format_validation_coerces_whole_numbers():
    """JSON numbers without a fraction are accepted as integers, not passed on as floats."""
    assert validate_audio_format(16000.0, 2.0, 1.0) == (16000, 2, 1)


def test_audio_format_validation_rejects_a_wav_byte_rate_overflow():
    with pytest.raises(ValueError, match="byte rate"):
        validate_audio_format(384000, 4, 16000)


def test_audio_format_validation_rejects_an_excessive_rate():
    with pytest.raises(ValueError, match="384000"):
        validate_audio_format(1_600_000, 2, 1)


def test_audio_format_validation_reports_a_fractional_rate():
    with pytest.raises(ValueError, match="16000.5"):
        validate_audio_format(16000.5, 2, 1)


def test_realtime_converter_flush_drops_a_partial_frame_and_starts_clean():
    converter = RealtimeAudioConverter(24000, 2, 1)
    assert converter.convert_chunk(b"\x00") == b""
    assert converter.flush() == b""

    assert converter.convert_chunk(b"\x00\x01") == b"\x00\x01"


@pytest.mark.parametrize("value", [10**300, 10**400])
def test_audio_format_validation_bounds_large_client_values_in_errors(value):
    with pytest.raises(ValueError) as error:
        validate_audio_format(value, 2, 1)

    assert len(str(error.value)) < 120


def test_realtime_pcm_resampler_rejects_non_positive_rates():
    with pytest.raises(ValueError, match="positive"):
        RealtimePcmResampler(0, 24000)


def test_realtime_pcm_resampler_flush_resets_the_stream_state():
    resampler = RealtimePcmResampler(16000, 24000)
    resampler.process([0, 1000, 2000])

    assert resampler.flush()
    assert resampler.flush() == []


@pytest.mark.parametrize("width", [0, 5])
def test_decode_pcm_samples_rejects_an_unsupported_width(width):
    with pytest.raises(ValueError, match="sample width"):
        decode_pcm_samples(b"\x00\x00", width)


def test_downmix_pcm_samples_rejects_zero_channels():
    with pytest.raises(ValueError, match="channel count"):
        downmix_pcm_samples([1, 2], 0)


def test_realtime_pcm_resampler_interpolates_known_samples():
    """A small input pins the interpolation position, fraction, and final-sample clamp."""
    resampler = RealtimePcmResampler(16000, 24000)

    converted = resampler.process([0, 3000, 6000]) + resampler.flush()

    assert converted == [0, 2000, 4000, 6000, 6000]


def test_realtime_pcm_resampler_interpolates_across_chunks():
    """The carried sample is the left neighbour of the chunk's first interpolation."""
    resampler = RealtimePcmResampler(16000, 24000)

    converted = resampler.process([0, 3000]) + resampler.process([6000]) + resampler.flush()

    assert converted == [0, 2000, 4000, 6000, 6000]


@pytest.mark.parametrize("rate", [8000, 16000, 44100, 48000])
def test_realtime_pcm_resampler_carries_state_across_single_sample_chunks(rate):
    """A stream of one-sample chunks is resampled, rather than passed through at the source rate."""
    samples = sine_samples(rate, 0.25)
    expected = reference_resample(samples, rate)

    resampler = RealtimePcmResampler(rate, 24000)
    streamed: list[int] = []
    for sample in samples:
        streamed += resampler.process([sample])
    streamed += resampler.flush()

    assert_samples_close(streamed, expected)
    assert len(streamed) == math.ceil(len(samples) * 24000 / rate)


@pytest.mark.parametrize("chunk_size", [1, 160, 161])
def test_realtime_pcm_resampler_output_length_does_not_drift(chunk_size):
    """One second of 16 kHz audio is one second of 24 kHz audio however it is chunked."""
    samples = sine_samples(16000, 1)
    resampler = RealtimePcmResampler(16000, 24000)
    converted: list[int] = []
    for start in range(0, len(samples), chunk_size):
        converted += resampler.process(samples[start : start + chunk_size])
    converted += resampler.flush()

    assert len(converted) == 24000
    assert_samples_close(converted, reference_resample(samples, 16000))


@pytest.mark.parametrize("rate,expected_count", [(96000, 3), (192000, 2)])
def test_realtime_pcm_resampler_length_follows_the_position_walk_at_high_rates(rate, expected_count):
    """A short high-rate stream ends at its last output position instead of overshooting a rounded count."""
    samples = list(range(100, 110))

    one_piece = RealtimePcmResampler(rate, 24000)
    expected = one_piece.process(samples) + one_piece.flush()

    resampler = RealtimePcmResampler(rate, 24000)
    streamed: list[int] = []
    for sample in samples:
        streamed += resampler.process([sample])
    streamed += resampler.flush()

    assert expected == reference_resample(samples, rate)
    assert len(expected) == expected_count
    assert streamed == expected


def test_realtime_pcm_decodes_signed_24_bit_samples():
    assert decode_pcm_samples(b"\x00\x00\x80\xff\xff\x7f", 3) == [-32768, 32767]


def test_realtime_converter_rejects_a_rate_below_the_minimum():
    with pytest.raises(ValueError, match="realtime transcription requires"):
        RealtimeAudioConverter(7999, 2, 1)


@pytest.mark.parametrize("width,channels", [(5, 1), (0, 1), (1, 0)])
def test_realtime_converter_rejects_an_unsupported_pcm_format(width, channels):
    with pytest.raises(ValueError):
        RealtimeAudioConverter(24000, width, channels)
