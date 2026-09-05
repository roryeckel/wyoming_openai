import argparse
import html
import json
import re
from collections.abc import Callable
from enum import Enum
from io import BytesIO
from xml.etree import ElementTree

# Pause/block elements separate words even without surrounding whitespace;
# inline elements (emphasis, prosody, say-as, ...) wrap text and must not.
_SSML_PAUSE_TAG_NAMES = frozenset(("break", "p", "s"))
# Known inline elements removed without a separator in the malformed-markup fallback.
_SSML_INLINE_TAG_NAMES = frozenset(
    ("audio", "emphasis", "lang", "mark", "phoneme", "prosody", "say-as", "speak", "sub", "voice", "w")
)
_MULTI_SPACE_RE = re.compile(r" {2,}")


def _iter_ssml_tag_spans(text: str):
    """Yield complete tag spans, respecting quotes in attribute values."""
    start = 0
    while (tag_start := text.find("<", start)) >= 0:
        quote: str | None = None
        quoted_gt: int | None = None
        index = tag_start + 1
        while index < len(text):
            char = text[index]
            if quote:
                if char == quote:
                    quote = None
                elif char == ">" and quoted_gt is None:
                    # If the quote never closes, use this as a recovery point
                    # so malformed markup does not leak into plain text.
                    quoted_gt = index
            elif char in "\"'":
                quote = char
            elif char == ">":
                yield tag_start, index + 1
                start = index + 1
                break
            elif char == "<":
                # The previous opening bracket is literal/malformed; this one
                # may start a tag that the fallback can still remove.
                tag_start = index
            index += 1
        else:
            if quoted_gt is not None:
                yield tag_start, quoted_gt + 1
                start = quoted_gt + 1
                continue
            return


def _ssml_tag_name(tag: str) -> str | None:
    """Return a lower-case local tag name from a complete XML-like tag."""
    content = tag[1:-1].strip().lstrip("/").strip()
    if not content or content[0] in "!?":
        return None

    name_end = 0
    while name_end < len(content) and content[name_end] not in " \t\r\n/":
        name_end += 1

    return content[:name_end].rsplit(":", 1)[-1].lower()


def _strip_ssml_tree(root: ElementTree.Element) -> str:
    """Extract text from parsed SSML while retaining pause-element boundaries."""
    parts: list[str] = []

    def walk(element: ElementTree.Element) -> None:
        tag = element.tag.rsplit("}", 1)[-1].lower() if isinstance(element.tag, str) else ""
        is_pause = tag in _SSML_PAUSE_TAG_NAMES
        if is_pause:
            parts.append(" ")

        if element.text:
            parts.append(element.text)

        for child in element:
            walk(child)
            if child.tail:
                parts.append(child.tail)

        if is_pause:
            parts.append(" ")

    walk(root)
    return "".join(parts)


def _strip_ssml_tags(text: str) -> str:
    """Fallback for malformed markup.

    Contiguous tag runs are handled as one unit: a run of only known inline
    elements (emphasis, prosody, ...) is removed without a separator, keeping
    their text flow intact. A run containing any unknown tag that sits between
    two non-whitespace characters gets a single space so adjacent words are
    never joined; literal text whitespace is preserved as-is.
    """
    parts: list[str] = []
    pos = 0
    tag_spans = list(_iter_ssml_tag_spans(text))
    index = 0
    while index < len(tag_spans):
        run_start, run_end = tag_spans[index]
        parts.append(text[pos:run_start])
        run = [text[run_start:run_end]]
        index += 1
        while index < len(tag_spans) and tag_spans[index][0] == run_end:
            _, run_end = tag_spans[index]
            run.append(text[tag_spans[index][0]:run_end])
            index += 1
        pos = run_end
        if any(_ssml_tag_name(tag) not in _SSML_INLINE_TAG_NAMES for tag in run):
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
    if force_fallback:
        return _MULTI_SPACE_RE.sub(" ", html.unescape(_strip_ssml_tags(text)))
    try:
        root = ElementTree.fromstring(f"<root>{text}</root>")
        stripped = _strip_ssml_tree(root)
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
