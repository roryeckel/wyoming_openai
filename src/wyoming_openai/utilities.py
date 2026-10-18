import argparse
import html.entities
import json
import logging
import math
import struct
from collections.abc import Callable
from enum import Enum
from io import BytesIO
from typing import TypeGuard

from .const import (
    REALTIME_AUDIO_CHANNELS,
    REALTIME_AUDIO_FORMAT,
    REALTIME_AUDIO_RATE,
    REALTIME_AUDIO_WIDTH,
    REALTIME_MIN_AUDIO_RATE,
    REALTIME_TTS_MAX_SPEED,
    REALTIME_TTS_MIN_SPEED,
)

_LOGGER = logging.getLogger(__name__)

MAX_AUDIO_RATE = 384_000  # Hz, above the highest consumer audio rate

# Pause/block elements separate words even without surrounding whitespace;
# inline elements (emphasis, prosody, say-as, ...) wrap text and must not.
SSML_PAUSE_TAG_NAMES = frozenset(("break", "p", "s"))
# Known inline elements are removed transparently without a word boundary.
SSML_INLINE_TAG_NAMES = frozenset(
    ("audio", "emphasis", "lang", "mark", "phoneme", "prosody", "say-as", "speak", "sub", "voice", "w")
)

# XML named and numeric character references only: HTML-only names (e.g.
# "&nbsp;") are markup text here, and a numeric reference ends at its ";" so
# "&#65mp;" decodes as "A" followed by the literal text "mp;".
_XML_ENTITY_NAMES = frozenset(("amp", "apos", "gt", "lt", "quot"))
# Bound the held-back partial token so a stream that never completes a tag or
# reference cannot grow the pending buffer (and its per-chunk rescan/copy)
# without limit. Far above any realistic SSML token; an oversized token is
# spilled and re-walked as literal text rather than reassembled.
SSML_MAX_TOKEN_LENGTH = 4096

_TAG_NAME_CHARS = frozenset("._:-")
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def truncate_for_log(text: str, max_length: int = 100) -> str:
    """Truncate text for logging, adding ellipsis only if truncated."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def _decode_entity_reference(body: str) -> str | None:
    """Decode an XML character/entity reference body (without '&' and ';').

    Returns the decoded text, or None when the body is not a well-formed XML
    reference. Out-of-range codepoints and lone surrogates stay literal.
    """
    if body.startswith("#"):
        digits = body[1:]
        if digits.startswith(("x", "X")):
            digits = digits[1:]
            base = 16
            valid = bool(digits) and all(c in _HEX_DIGITS for c in digits)
        else:
            base = 10
            valid = bool(digits) and digits.isdigit()
        if not valid:
            return None
        try:
            codepoint = int(digits, base)
            if not (
                codepoint in (0x09, 0x0A, 0x0D)
                or 0x20 <= codepoint <= 0xD7FF
                or 0xE000 <= codepoint <= 0xFFFD
                or 0x10000 <= codepoint <= 0x10FFFF
            ):
                return None
            return chr(codepoint)
        except (ValueError, OverflowError):
            return None
    if body in _XML_ENTITY_NAMES:
        return chr(html.entities.name2codepoint[body])
    return None


def _is_entity_prefix(body: str) -> bool:
    """Return whether body can still become a complete XML reference."""
    if body.startswith("#"):
        digits = body[1:]
        if not digits:
            return True
        if digits.startswith(("x", "X")):
            return len(digits) == 1 or all(char in _HEX_DIGITS for char in digits[1:])
        return all(char in "0123456789" for char in digits)
    return any(name.startswith(body) for name in _XML_ENTITY_NAMES)


def _parse_tag_name(markup: str) -> str | None:
    """Return the lower-case local tag name of a complete "<...>" construct.

    Namespace prefixes are dropped; names are matched case-insensitively.
    Returns None for comments, CDATA sections, processing instructions, and
    declarations, which are removed transparently.
    """
    pos = 1
    if pos < len(markup) and markup[pos] == "/":
        pos += 1
    while pos < len(markup) and markup[pos] in " \t\r\n":
        pos += 1
    name_start = pos
    while pos < len(markup) and (markup[pos].isalnum() or markup[pos] in _TAG_NAME_CHARS):
        pos += 1
    return markup[name_start:pos].rsplit(":", 1)[-1].lower() or None


class SsmlTextTransformer:
    """Incrementally project lenient SSML to backend-safe plain text."""

    def __init__(self) -> None:
        self._token = ""
        self._token_kind: str | None = None
        self._tag_quote: str | None = None
        self._space_pending = False

    def feed(self, text: str) -> str:
        """Consume one chunk and return the text that is safe to emit now."""
        output: list[str] = []
        for char in text:
            if self._token_kind is not None:
                self._consume_token_char(char, output)
            elif char == "<":
                self._start_token("tag")
            elif char == "&":
                self._start_token("entity")
            else:
                self._consume_plain_char(char, output)
        return "".join(output)

    def finish(self) -> str:
        """Flush an incomplete token and pending whitespace at end of stream."""
        output: list[str] = []
        if self._token_kind is not None:
            token = self._token
            self._clear_token()
            self._emit_literal_text(token, output)
        if self._space_pending:
            self._space_pending = False
            output.append(" ")
        return "".join(output)

    def _start_token(self, kind: str) -> None:
        self._token = "<" if kind == "tag" else "&"
        self._token_kind = kind
        self._tag_quote = None

    def _clear_token(self) -> None:
        self._token = ""
        self._token_kind = None
        self._tag_quote = None

    def _consume_token_char(self, char: str, output: list[str]) -> None:
        if self._token_kind == "entity":
            self._consume_entity_char(char, output)
            return

        if self._token.startswith("<!--"):
            self._token += char
            if self._token.endswith("-->"):
                self._clear_token()
            elif len(self._token) > SSML_MAX_TOKEN_LENGTH:
                self._flush_oversized_literal(output)
            return

        if self._token.startswith("<![CDATA["):
            self._token += char
            if self._token.endswith("]]>"):
                content = self._token[9:-3]
                self._clear_token()
                self._emit_literal_text(content, output)
            elif len(self._token) > SSML_MAX_TOKEN_LENGTH:
                self._flush_oversized_literal(output)
            return

        if self._token.startswith("<?"):
            self._token += char
            if self._token.endswith("?>"):
                self._clear_token()
            elif len(self._token) > SSML_MAX_TOKEN_LENGTH:
                self._flush_oversized_literal(output)
            return

        if self._tag_quote is not None:
            self._token += char
            if char == self._tag_quote:
                self._tag_quote = None
        elif char in "\"'":
            self._token += char
            self._tag_quote = char
        elif char == ">":
            self._token += char
            token = self._token
            self._clear_token()
            self._resolve_tag(token, output)
            return
        elif char == "<":
            # A nested opening bracket makes the preceding construct invalid.
            # Preserve it literally and let the new bracket start a token.
            self._emit_literal_text(self._token, output)
            self._start_token("tag")
            return
        else:
            self._token += char

        if len(self._token) > SSML_MAX_TOKEN_LENGTH:
            self._flush_oversized_literal(output)

    def _consume_entity_char(self, char: str, output: list[str]) -> None:
        if char == ";":
            token = self._token + char
            decoded = _decode_entity_reference(token[1:-1])
            self._clear_token()
            if decoded is None:
                self._emit_literal_text(token, output)
            else:
                self._emit_literal_text(decoded, output)
            return

        if char in "<&" or char.isspace():
            token = self._token
            self._clear_token()
            self._emit_literal_text(token, output)
            if char == "<":
                self._start_token("tag")
            elif char == "&":
                self._start_token("entity")
            else:
                self._consume_plain_char(char, output)
            return

        candidate = self._token + char
        if not _is_entity_prefix(candidate[1:]):
            token = self._token
            self._clear_token()
            self._emit_literal_text(token, output)
            self._consume_plain_char(char, output)
            return

        self._token = candidate
        if len(self._token) > SSML_MAX_TOKEN_LENGTH:
            self._flush_oversized_literal(output)

    def _flush_oversized_literal(self, output: list[str]) -> None:
        token = self._token
        self._clear_token()
        self._emit_literal_text(token, output)

    def _resolve_tag(self, token: str, output: list[str]) -> None:
        if token.startswith("<!"):
            # Complete declarations are markup and have no text projection.
            return
        name = _parse_tag_name(token)
        if name is None:
            self._emit_literal_text(token, output)
        elif name in SSML_PAUSE_TAG_NAMES:
            self._request_space()
        elif name in SSML_INLINE_TAG_NAMES:
            return
        else:
            self._request_space()

    def _consume_plain_char(self, char: str, output: list[str]) -> None:
        if char == " ":
            self._request_space()
        else:
            self._emit_plain_char(char, output)

    def _emit_plain_char(self, char: str, output: list[str]) -> None:
        if self._space_pending:
            output.append(" ")
            self._space_pending = False
        output.append(char)

    def _emit_literal_text(self, text: str, output: list[str]) -> None:
        for char in text:
            self._consume_plain_char(char, output)

    def _request_space(self) -> None:
        self._space_pending = True


def strip_ssml(text: str) -> str:
    """Project SSML to backend-safe plain text with shared stream semantics."""
    transformer = SsmlTextTransformer()
    return transformer.feed(text) + transformer.finish()


def create_enum_parser[E: Enum](enum_class: type[E], case_insensitive: bool = True) -> Callable[[str], E]:
    """
    Create a type-safe parser function for argparse that converts strings to enum members.

    This function generates a parser that:
    - Handles case-insensitive matching (optional)
    - Provides clear error messages listing all valid options
    - Raises argparse.ArgumentTypeError for invalid inputs

    Args:
        enum_class: The Enum class to parse into
        case_insensitive: Whether to allow case-insensitive matching (default: True)

    Returns:
        A callable that takes a string and returns the corresponding enum member

    Raises:
        argparse.ArgumentTypeError: When the input string doesn't match any enum member

    Example:
        >>> from enum import Enum
        >>> class Color(Enum):
        ...     RED = 1
        ...     BLUE = 2
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument('--color', type=create_enum_parser(Color))
        >>> args = parser.parse_args(['--color', 'red'])
        >>> args.color == Color.RED
        True
    """

    def parse_enum(value: str) -> E:
        lookup_value = value.upper() if case_insensitive else value
        try:
            return enum_class[lookup_value]
        except KeyError as exc:
            valid_options = ", ".join(member.name for member in enum_class)
            raise argparse.ArgumentTypeError(
                f"Invalid {enum_class.__name__}: '{value}'. Valid options are: {valid_options}"
            ) from exc

    return parse_enum


def create_json_object_parser(option_name: str) -> Callable[[str], dict[str, object]]:
    """
    Create an argparse parser that validates a JSON object string.

    Args:
        option_name: Human-readable option name to include in error messages.

    Returns:
        A callable that parses a JSON object string into a dictionary.

    Raises:
        argparse.ArgumentTypeError: When the value is not valid JSON or is not a JSON object.
    """

    def parse_json_object(value: str) -> dict[str, object]:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise argparse.ArgumentTypeError(f"Invalid {option_name}: {exc.msg}") from exc

        if not isinstance(parsed, dict):
            raise argparse.ArgumentTypeError(
                f"Invalid {option_name}: expected a JSON object, got {type(parsed).__name__}"
            )

        return parsed

    return parse_json_object


def parse_voice_labels(value: str) -> dict[str, str]:
    """
    Parse an argparse value holding a JSON object that maps TTS voice names to display names.

    Example: '{"af_heart": "Heart", "af_bella": "Bella"}'

    Returns:
        A dictionary of voice name to display name, with surrounding whitespace removed from both.

    Raises:
        argparse.ArgumentTypeError: When the value is not a JSON object whose keys and values are non-empty strings.
    """
    parsed = create_json_object_parser("TTS voice labels")(value)

    labels: dict[str, str] = {}
    for voice_name, label in parsed.items():
        if not voice_name.strip():
            raise argparse.ArgumentTypeError("Invalid TTS voice labels: voice names must be non-empty strings")
        if not isinstance(label, str) or not label.strip():
            raise argparse.ArgumentTypeError(
                f"Invalid TTS voice labels: the label for '{voice_name}' must be a non-empty string"
            )
        labels[voice_name.strip()] = label.strip()

    return labels


def parse_finite_float(value: str) -> float:
    """
    Parse an argparse value that must be a finite number, since a request cannot carry inf or nan as JSON.

    Raises:
        argparse.ArgumentTypeError: When the value is not a number or is not finite.
    """
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got '{value}'") from None

    if not math.isfinite(number):
        raise argparse.ArgumentTypeError(f"expected a finite number, got {number}")

    return number


def parse_positive_int(value: str) -> int:
    """
    Parse an argparse value that must be a whole number of at least 1.

    Raises:
        argparse.ArgumentTypeError: When the value is not an integer or is smaller than 1.
    """
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number of at least 1, got '{value}'") from None

    if number < 1:
        raise argparse.ArgumentTypeError(f"expected a whole number of at least 1, got {number}")

    return number


def validate_extra_body_response_format(
    extra_body: dict[str, object] | None, *, allowed_formats: set[str], body_name: str
) -> None:
    """
    Reject response formats that the handler cannot decode.

    Args:
        extra_body: Optional extra request fields to validate.
        allowed_formats: Response formats supported by the consumer.
        body_name: Human-readable request name for the error message.

    Raises:
        ValueError: When extra_body requests an unsupported response_format.
    """
    if not extra_body:
        return

    if "response_format" not in extra_body:
        return

    response_format = extra_body["response_format"]
    if isinstance(response_format, str) and response_format in allowed_formats:
        return

    expected_formats = ", ".join(repr(fmt) for fmt in sorted(allowed_formats))
    raise ValueError(
        f"{body_name} extra_body response_format must be one of {expected_formats}; "
        f"got {response_format!r}"
    )


def validate_extra_body_boolean_field(
    extra_body: dict[str, object] | None, *, field_name: str, body_name: str
) -> None:
    """Reject non-boolean extra_body fields that affect response parsing."""
    if not extra_body or field_name not in extra_body:
        return

    field_value = extra_body[field_name]
    if isinstance(field_value, bool):
        return

    raise ValueError(f"{body_name} extra_body {field_name} must be a boolean; got {field_value!r}")


def validate_extra_body_disallowed_fields(
    extra_body: dict[str, object] | None,
    *,
    field_names: set[str] | frozenset[str],
    body_name: str,
    reason: str = "Wyoming expects raw audio bytes",
) -> None:
    """Reject extra_body fields that would change the response transport."""
    if not extra_body:
        return

    disallowed_fields = sorted(field_name for field_name in field_names if field_name in extra_body)
    if not disallowed_fields:
        return

    formatted_fields = ", ".join(repr(field_name) for field_name in disallowed_fields)
    raise ValueError(f"{body_name} extra_body does not support overriding {formatted_fields}; {reason}")


def get_extra_body_boolean_field(
    extra_body: dict[str, object] | None, *, field_name: str, default: bool, body_name: str
) -> bool:
    """Return a boolean override from extra_body or fall back to a default value."""
    if not extra_body or field_name not in extra_body:
        return default

    field_value = extra_body[field_name]
    if isinstance(field_value, bool):
        return field_value

    raise ValueError(f"{body_name} extra_body {field_name} must be a boolean; got {field_value!r}")


def validate_extra_body_languages(extra_body: dict[str, object] | None, *, body_name: str, allow_null: bool) -> None:
    """
    Reject a `languages` override that is not a list of language codes. It replaces the request language, so a
    wrong one would otherwise only fail at the first transcription; `null` drops it where a null drops a field.
    """
    if not extra_body or "languages" not in extra_body:
        return
    languages = extra_body["languages"]
    if languages is None and allow_null:
        return
    if isinstance(languages, list) and languages and all(isinstance(code, str) and code for code in languages):
        return
    raise ValueError(f"{body_name} extra_body languages must be a list of language codes; got {languages!r}")


def validate_stt_extra_body(extra_body: dict[str, object] | None) -> None:
    """Validate STT extra_body fields that can affect client-side parsing."""
    validate_extra_body_response_format(extra_body, allowed_formats={"json"}, body_name="STT")
    validate_extra_body_boolean_field(extra_body, field_name="stream", body_name="STT")
    validate_extra_body_languages(extra_body, body_name="STT", allow_null=False)


def validate_tts_extra_body(extra_body: dict[str, object] | None) -> None:
    """Validate TTS extra_body fields that can affect response decoding."""
    validate_extra_body_response_format(extra_body, allowed_formats={"pcm", "wav"}, body_name="TTS")
    validate_extra_body_disallowed_fields(
        extra_body,
        field_names={"stream", "stream_format"},
        body_name="TTS",
    )


def validate_realtime_stt_extra_body(extra_body: dict[str, object] | None) -> None:
    """Validate Realtime STT extra_body fields; the model is the one the Wyoming client selected."""
    validate_extra_body_disallowed_fields(
        extra_body, field_names={"model"}, body_name="STT Realtime", reason="use STT_REALTIME_MODELS"
    )
    validate_extra_body_languages(extra_body, body_name="STT Realtime", allow_null=True)


# Session fields Realtime TTS sets itself: the ones it relies on to get speech, and only speech, back,
# the model the Wyoming client selected, and the read-aloud instructions
REALTIME_TTS_FIXED_SESSION_FIELDS = frozenset(
    ("type", "output_modalities", "tool_choice", "tools", "model", "instructions")
)


def validate_realtime_tts_extra_body(extra_body: dict[str, object] | None) -> None:
    """Validate Realtime TTS extra_body fields that affect the audio Wyoming is told to expect."""
    if not extra_body:
        return

    validate_extra_body_disallowed_fields(
        extra_body,
        field_names=REALTIME_TTS_FIXED_SESSION_FIELDS,
        body_name="TTS Realtime",
        reason=(
            "the session has to stay audio-only speech, the model comes from TTS_REALTIME_MODELS "
            "and the delivery style from TTS_INSTRUCTIONS"
        ),
    )

    # These pre-GA top-level voice and speed fields conflict with the supported audio.output settings.
    validate_extra_body_disallowed_fields(
        extra_body,
        field_names={"voice", "speed"},
        body_name="TTS Realtime",
        reason="the voice is chosen per request by the Wyoming client and the speed is set with audio.output.speed",
    )

    # A null audio or audio.output is the same as leaving it out, as get_realtime_tts_audio_output reads it
    audio = extra_body.get("audio")
    if audio is None:
        audio = {}
    audio_output = audio.get("output") if isinstance(audio, dict) else None
    if not isinstance(audio, dict) or not (audio_output is None or isinstance(audio_output, dict)):
        raise ValueError("TTS Realtime extra_body audio must be an object whose output, if set, is an object")

    output = get_realtime_tts_audio_output(extra_body)
    if "voice" in output:
        raise ValueError(
            "TTS Realtime extra_body audio.output.voice is not supported; "
            "the voice is chosen per request by the Wyoming client"
        )

    speed = output.get("speed")
    if speed is not None and not _is_finite_number(speed):
        raise ValueError(f"TTS Realtime extra_body audio.output.speed must be a finite number; got {speed!r}")

    if "format" not in output:
        return

    audio_format = output["format"]
    if isinstance(audio_format, dict) and {"rate": REALTIME_AUDIO_FORMAT["rate"], **audio_format} == (
        REALTIME_AUDIO_FORMAT
    ):
        return

    raise ValueError(
        f"TTS Realtime extra_body audio.output.format must be {REALTIME_AUDIO_FORMAT!r}; got {audio_format!r}"
    )


def get_realtime_tts_audio_output(extra_body: dict[str, object] | None) -> dict[str, object]:
    """Return a copy of the audio.output object of a Realtime TTS extra_body, or an empty dict when unset."""
    audio = (extra_body or {}).get("audio")
    output = audio.get("output") if isinstance(audio, dict) else None
    return dict(output) if isinstance(output, dict) else {}


def get_realtime_tts_speed(tts_speed: float | None, extra_body: dict[str, object] | None) -> float | None:
    """
    Return the speed requested for Realtime TTS, or None for the default.
    An audio.output.speed in the extra body overrides the TTS speed.
    """
    speed = get_realtime_tts_audio_output(extra_body).get("speed")
    if speed is None:
        # A null override is the same as no override
        speed = tts_speed
    if not _is_finite_number(speed):
        # A speed that is not a finite number cannot be clamped or serialized, so the default is used
        return None
    return speed


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    """Check for a finite int or float; booleans are not numbers here."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        # An int too large for a float is not a usable number either
        return False


def _coerce_audio_format_int(value: object, name: str) -> int:
    """Coerce one client-declared audio-start field to an integer, rejecting fractions and non-numbers."""
    if isinstance(value, int) and not isinstance(value, bool):
        # An int can be arbitrarily large; the range checks below reject it without a float conversion
        return value
    if isinstance(value, float):
        if not _is_finite_number(value):
            raise ValueError(f"audio-start {name} must be finite, got {truncate_for_log(repr(value), 40)}")
        if not value.is_integer():
            raise ValueError(f"audio-start {name} must be a whole number, got {truncate_for_log(repr(value), 40)}")
        return int(value)
    raise ValueError(f"audio-start {name} must be a finite number, got {type(value).__name__}")


def validate_audio_format(sample_rate: object, audio_width: object, audio_channels: object) -> tuple[int, int, int]:
    """
    Coerce and validate the audio format of an audio-start once, before either transport uses it.

    The limits are those of a WAV header: rate and byte rate fit in 32 bits, channel count and frame size in 16.
    """
    rate = _coerce_audio_format_int(sample_rate, "rate")
    width = _coerce_audio_format_int(audio_width, "width")
    channels = _coerce_audio_format_int(audio_channels, "channels")

    if not 1 <= rate <= MAX_AUDIO_RATE:
        raise ValueError(
            f"audio-start rate must be between 1 and {MAX_AUDIO_RATE} Hz, "
            f"got {truncate_for_log(repr(rate), 40)}"
        )
    if width not in (1, 2, 3, 4):
        raise ValueError(f"audio-start width must be 1, 2, 3 or 4 bytes, got {truncate_for_log(repr(width), 40)}")
    if not 1 <= channels <= 0xFFFF:
        raise ValueError(
            f"audio-start channels must be between 1 and 65535, got {truncate_for_log(repr(channels), 40)}"
        )
    if width * channels > 0xFFFF:
        raise ValueError("audio-start width and channels give a frame larger than 65535 bytes")
    if rate * width * channels > 0xFFFFFFFF:
        raise ValueError("audio-start rate, width and channels give a byte rate above 4294967295 B/s")
    return rate, width, channels


def decode_pcm_samples(audio: bytes, audio_width: int) -> list[int]:
    """Decode validated PCM bytes into signed 16-bit sample values."""
    if audio_width not in (1, 2, 3, 4):
        raise ValueError(f"Unsupported audio sample width for realtime transcription: {audio_width}")

    sample_count = len(audio) // audio_width
    if sample_count == 0:
        return []

    audio = audio[: sample_count * audio_width]
    if audio_width == 1:
        return [(sample - 128) << 8 for sample in audio]

    if audio_width == 2:
        return list(struct.unpack(f"<{sample_count}h", audio))

    # 24-bit and 32-bit samples, shifted to 16-bit
    shift = 8 * (audio_width - 2)
    return [
        int.from_bytes(audio[index : index + audio_width], "little", signed=True) >> shift
        for index in range(0, len(audio), audio_width)
    ]


def downmix_pcm_samples(samples: list[int], audio_channels: int) -> list[int]:
    """Downmix validated PCM samples to mono."""
    if audio_channels <= 0:
        raise ValueError(f"Unsupported audio channel count for realtime transcription: {audio_channels}")

    if audio_channels == REALTIME_AUDIO_CHANNELS:
        return samples

    frame_count = len(samples) // audio_channels
    mono_samples = []
    for frame_index in range(frame_count):
        start = frame_index * audio_channels
        mono_samples.append(round(sum(samples[start : start + audio_channels]) / audio_channels))
    return mono_samples


def take_whole_frames(*, partial: bytes, audio: bytes, frame_size: int) -> tuple[bytes, bytes]:
    """
    Take the whole frames of partial and audio, holding back the start of one that continues in the next bytes.

    A frame can be split wherever a stream is, for one by a header of odd length.
    """
    if frame_size <= 1:
        return partial + audio, b""
    audio = partial + audio
    whole = len(audio) - len(audio) % frame_size
    return audio[:whole], audio[whole:]


def encode_pcm16_samples(samples: list[int]) -> bytes:
    """Encode signed sample values as little-endian PCM16."""
    if not samples:
        return b""

    clamped_samples = (max(-32768, min(32767, int(sample))) for sample in samples)
    return struct.pack(f"<{len(samples)}h", *clamped_samples)


class RealtimePcmResampler:
    """
    Linear resampler whose phase and last sample span the chunks of a Realtime stream.

    Output sample k sits at input position k * source_rate / target_rate from the start of the stream. The position is
    kept in whole target-rate units from the carried sample, so a long stream does not drift, and the last sample of
    a chunk is carried so the next output interpolates towards the next chunk's first sample.
    """

    def __init__(self, source_rate: int, target_rate: int) -> None:
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("sample rates must be positive")
        self._source_rate = source_rate
        self._target_rate = target_rate
        self._position = 0  # The next output sample, in 1/target_rate input samples, after the carried sample
        self._last_sample: int | None = None  # The last input sample of the previous chunk

    def process(self, samples: list[int]) -> list[int]:
        """Resample one chunk, carrying the position and last sample to the next one."""
        if not samples:
            return []
        source_rate = self._source_rate
        target_rate = self._target_rate
        if source_rate == target_rate:
            return samples

        carried = self._last_sample
        offset = 0 if carried is None else 1
        length = len(samples) + offset
        end_position = (length - 1) * target_rate
        resampled: list[int] = []
        position = self._position
        while position < end_position:
            left_index, remainder = divmod(position, target_rate)
            # A carried sample takes virtual index 0, and the chunk's samples follow it
            left_sample = carried if carried is not None and left_index < offset else samples[left_index - offset]
            right_sample = samples[left_index + 1 - offset]
            if remainder:
                fraction = remainder / target_rate
                resampled.append(round(left_sample + (right_sample - left_sample) * fraction))
            else:
                resampled.append(left_sample)
            position += source_rate

        # The next position is relative to the last sample, which the next chunk carries
        self._position = position - end_position
        self._last_sample = samples[-1]
        return resampled

    def flush(self) -> list[int]:
        """
        Emit the outputs held for the end of the stream: each starts at or after its last input sample, so all are
        repeats of it. This follows the position walk rather than a rounded sample count, so nothing emitted by
        process() is ever beyond the stream's own length.
        """
        last_sample = self._last_sample
        if last_sample is None or self._source_rate == self._target_rate:
            return []

        # Ceiling division of the distance to one input sample past the carried sample
        remaining = max(0, -(-(self._target_rate - self._position) // self._source_rate))
        self._position = 0
        self._last_sample = None
        return [last_sample] * remaining


class RealtimeAudioConverter:
    """Converts one Realtime stream's Wyoming PCM to 24 kHz mono PCM16, carrying frame and resampling state."""

    def __init__(self, sample_rate: int, audio_width: int, audio_channels: int) -> None:
        sample_rate, audio_width, audio_channels = validate_audio_format(sample_rate, audio_width, audio_channels)
        if sample_rate < REALTIME_MIN_AUDIO_RATE:
            raise ValueError(
                f"realtime transcription requires at least {REALTIME_MIN_AUDIO_RATE} Hz, got {sample_rate}"
            )
        self._audio_width = audio_width
        self._audio_channels = audio_channels
        self._frame_size = audio_width * audio_channels
        self._passthrough = (
            sample_rate == REALTIME_AUDIO_RATE
            and audio_width == REALTIME_AUDIO_WIDTH
            and audio_channels == REALTIME_AUDIO_CHANNELS
        )
        self._pcm_tail = b""
        self._resampler = RealtimePcmResampler(sample_rate, REALTIME_AUDIO_RATE)

    def convert_chunk(self, audio: bytes) -> bytes:
        """Convert the complete frames of one chunk, carrying a partial frame to the next chunk."""
        audio, self._pcm_tail = take_whole_frames(
            partial=self._pcm_tail, audio=audio, frame_size=self._frame_size
        )
        if not audio:
            return b""

        if self._passthrough:
            return audio

        samples = decode_pcm_samples(audio, self._audio_width)
        samples = downmix_pcm_samples(samples, self._audio_channels)
        return encode_pcm16_samples(self._resampler.process(samples))

    def flush(self) -> bytes:
        """Encode the resampler's held samples and drop a final partial frame."""
        audio = encode_pcm16_samples(self._resampler.flush())
        if self._pcm_tail:
            _LOGGER.debug("Dropping %d bytes that end the Realtime audio in the middle of a frame", len(self._pcm_tail))
            self._pcm_tail = b""
        return audio


def clamp_realtime_tts_speed(speed: float) -> float:
    """Clamp a speed to the range the Realtime API accepts."""
    return min(max(speed, REALTIME_TTS_MIN_SPEED), REALTIME_TTS_MAX_SPEED)


class NamedBytesIO(BytesIO):
    """
    A subclass of BytesIO that adds a 'name' attribute to the file-like object.
    """

    def __init__(self, *args, name="audio.wav", **kwargs):
        """
        Initialize a new NamedBytesIO instance.

        Args:
            *args: Variable length argument list passed to BytesIO constructor.
            name (str): The name or filename associated with this byte stream.
                        Default is 'audio.wav'.
            **kwargs: Arbitrary keyword arguments passed to BytesIO constructor.
        """
        super().__init__(*args, **kwargs)
        self._name = name

    @property
    def name(self):
        """
        Returns the name of the byte stream.

        Returns:
            str: The name or filename associated with this byte stream.
        """
        return self._name
