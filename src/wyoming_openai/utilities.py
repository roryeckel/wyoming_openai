import argparse
import json
from collections.abc import Callable
from enum import Enum
from io import BytesIO


def create_enum_parser[E: Enum](enum_class: type[E], case_insensitive: bool = True) -> Callable[[str], E]:
    """Create an argparse type function that converts strings to enum members."""

    def parse_enum(value: str) -> E:
        lookup = value.upper() if case_insensitive else value
        try:
            return enum_class[lookup]
        except KeyError as exc:
            valid = ", ".join(m.name for m in enum_class)
            raise argparse.ArgumentTypeError(f"Invalid {enum_class.__name__}: '{value}'. Valid: {valid}") from exc

    return parse_enum


def create_json_object_parser(option_name: str) -> Callable[[str], dict[str, object]]:
    """Create an argparse type function that validates a JSON object string."""

    def parse(value: str) -> dict[str, object]:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise argparse.ArgumentTypeError(f"Invalid {option_name}: {exc.msg}") from exc
        if not isinstance(parsed, dict):
            raise argparse.ArgumentTypeError(
                f"Invalid {option_name}: expected a JSON object, got {type(parsed).__name__}"
            )
        return parsed

    return parse


def _check_response_format(extra_body: dict[str, object] | None, *, allowed: set[str], name: str) -> None:
    """Reject unsupported response_format values in extra_body."""
    if not extra_body or "response_format" not in extra_body:
        return
    fmt = extra_body["response_format"]
    if isinstance(fmt, str) and fmt in allowed:
        return
    expected = ", ".join(repr(f) for f in sorted(allowed))
    raise ValueError(f"{name} extra_body response_format must be one of {expected}; got {fmt!r}")


def _check_bool_field(extra_body: dict[str, object] | None, *, field: str, name: str) -> None:
    """Reject non-boolean extra_body fields."""
    if not extra_body or field not in extra_body:
        return
    if not isinstance(extra_body[field], bool):
        raise ValueError(f"{name} extra_body {field} must be a boolean; got {extra_body[field]!r}")


def _check_disallowed_fields(extra_body: dict[str, object] | None, *, fields: set[str], name: str) -> None:
    """Reject extra_body fields that would change response transport."""
    if not extra_body:
        return
    bad = sorted(f for f in fields if f in extra_body)
    if bad:
        formatted = ", ".join(repr(f) for f in bad)
        raise ValueError(f"{name} extra_body does not support overriding {formatted}; Wyoming expects raw audio bytes")


def get_extra_body_boolean_field(
    extra_body: dict[str, object] | None, *, field_name: str, default: bool, body_name: str
) -> bool:
    """Return a boolean override from extra_body, or the default."""
    if not extra_body or field_name not in extra_body:
        return default
    val = extra_body[field_name]
    if isinstance(val, bool):
        return val
    raise ValueError(f"{body_name} extra_body {field_name} must be a boolean; got {val!r}")


def validate_stt_extra_body(extra_body: dict[str, object] | None) -> None:
    _check_response_format(extra_body, allowed={"json"}, name="STT")
    _check_bool_field(extra_body, field="stream", name="STT")


def validate_tts_extra_body(extra_body: dict[str, object] | None) -> None:
    _check_response_format(extra_body, allowed={"pcm", "wav"}, name="TTS")
    _check_disallowed_fields(extra_body, fields={"stream", "stream_format"}, name="TTS")


# Aliases for backward compatibility with old names used in tests
validate_extra_body_response_format = lambda eb, *, allowed_formats, body_name: _check_response_format(
    eb, allowed=allowed_formats, name=body_name
)
validate_extra_body_boolean_field = lambda eb, *, field_name, body_name: _check_bool_field(
    eb, field=field_name, name=body_name
)
validate_extra_body_disallowed_fields = lambda eb, *, field_names, body_name: _check_disallowed_fields(
    eb, fields=field_names, name=body_name
)


class NamedBytesIO(BytesIO):
    """BytesIO with a name attribute (needed by OpenAI SDK for file uploads)."""

    def __init__(self, *args, name="audio.wav", **kwargs):
        super().__init__(*args, **kwargs)
        self._name = name

    @property
    def name(self):
        return self._name
