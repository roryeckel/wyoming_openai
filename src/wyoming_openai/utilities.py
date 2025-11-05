import argparse
from collections.abc import Callable
from enum import Enum
from io import BytesIO
from typing import TypeVar

E = TypeVar('E', bound=Enum)


def create_enum_parser(enum_class: type[E], case_insensitive: bool = True) -> Callable[[str], E]:
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
            valid_options = ', '.join(member.name for member in enum_class)
            raise argparse.ArgumentTypeError(
                f"Invalid {enum_class.__name__}: '{value}'. "
                f"Valid options are: {valid_options}"
            ) from exc

    return parse_enum


class NamedBytesIO(BytesIO):
    """
    A subclass of BytesIO that adds a 'name' attribute to the file-like object.
    """
    def __init__(self, *args, name='audio.wav', **kwargs):
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
