import argparse
import html.entities
import json
from collections.abc import Callable
from enum import Enum
from io import BytesIO

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
