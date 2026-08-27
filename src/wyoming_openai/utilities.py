import argparse
import html
import json
import re
from collections.abc import Callable
from enum import Enum
from io import BytesIO
from xml.etree import ElementTree

_SSML_TAG_RE = re.compile(r"<[^>]*>")
# Pause/block elements separate words even without surrounding whitespace;
# inline elements (emphasis, prosody, say-as, ...) wrap text and must not.
_SSML_PAUSE_TAG_RE = re.compile(r"</?(?:break|p|s)\b[^>]*>", re.IGNORECASE)
# Known inline elements removed without a separator in the regex fallback.
_SSML_INLINE_TAG_RE = re.compile(
    r"</?(?:audio|emphasis|lang|mark|phoneme|prosody|say-as|speak|sub|voice|w)\b[^>]*>",
    re.IGNORECASE,
)
_MULTI_SPACE_RE = re.compile(r" {2,}")


def _strip_ssml_tags(text: str) -> str:
    """Regex fallback for malformed markup.

    Contiguous tag runs are handled as one unit: a run of only known inline
    elements (emphasis, prosody, ...) is removed without a separator, keeping
    their text flow intact. A run containing any unknown tag that sits between
    two non-whitespace characters gets a single space so adjacent words are
    never joined; literal text whitespace is preserved as-is.
    """
    parts: list[str] = []
    pos = 0
    for match in _SSML_TAG_RE.finditer(text):
        parts.append(text[pos : match.start()])
        run_start = match.start()
        run_end = match.end()
        run = [match.group(0)]
        while (next_match := _SSML_TAG_RE.search(text, run_end)) and next_match.start() == run_end:
            run.append(next_match.group(0))
            run_end = next_match.end()
        pos = run_end
        if any(not _SSML_INLINE_TAG_RE.fullmatch(tag) for tag in run):
            prev_char = text[run_start - 1] if run_start > 0 else ""
            next_char = text[run_end] if run_end < len(text) else ""
            if prev_char and next_char and not prev_char.isspace() and not next_char.isspace():
                parts.append(" ")
    parts.append(text[pos:])
    return "".join(parts)


def strip_ssml(text: str, *, force_fallback: bool = False) -> str:
    """Strip SSML markup, returning plain text for backends that only accept it.

    Pause semantics (e.g. <break/>) are reduced to a word boundary; no in-scope
    backend accepts them. Malformed markup falls back to regex tag removal with
    entity decoding. With force_fallback=True the XML parser is skipped so a
    caller can protect raw "&" text from cross-context entity decoding.
    """
    text = _SSML_PAUSE_TAG_RE.sub(" ", text)
    if force_fallback:
        return _MULTI_SPACE_RE.sub(" ", html.unescape(_strip_ssml_tags(text)))
    try:
        root = ElementTree.fromstring(f"<root>{text}</root>")
        stripped = "".join(root.itertext())
    except ElementTree.ParseError:
        stripped = html.unescape(_strip_ssml_tags(text))
    return _MULTI_SPACE_RE.sub(" ", stripped)


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
    extra_body: dict[str, object] | None, *, field_names: set[str], body_name: str
) -> None:
    """Reject extra_body fields that would change the response transport."""
    if not extra_body:
        return

    disallowed_fields = sorted(field_name for field_name in field_names if field_name in extra_body)
    if not disallowed_fields:
        return

    formatted_fields = ", ".join(repr(field_name) for field_name in disallowed_fields)
    raise ValueError(
        f"{body_name} extra_body does not support overriding {formatted_fields}; "
        "Wyoming expects raw audio bytes"
    )


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


def validate_stt_extra_body(extra_body: dict[str, object] | None) -> None:
    """Validate STT extra_body fields that can affect client-side parsing."""
    validate_extra_body_response_format(extra_body, allowed_formats={"json"}, body_name="STT")
    validate_extra_body_boolean_field(extra_body, field_name="stream", body_name="STT")


def validate_tts_extra_body(extra_body: dict[str, object] | None) -> None:
    """Validate TTS extra_body fields that can affect response decoding."""
    validate_extra_body_response_format(extra_body, allowed_formats={"pcm", "wav"}, body_name="TTS")
    validate_extra_body_disallowed_fields(
        extra_body,
        field_names={"stream", "stream_format"},
        body_name="TTS",
    )


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
