"""User states recognised by the EEG classification pipeline."""

from __future__ import annotations

from enum import Enum, unique


@unique
class UserState(str, Enum):
    """Mental state fed into the tamagotchi engine by the classifier.

    The enum derives from ``str`` so that values round-trip through JSON
    and the CLI without custom (de)serialisation.
    """

    RELAX = "RELAX"
    FOCUS = "FOCUS"
    OVERLOAD = "OVERLOAD"

    @classmethod
    def from_value(cls, value: str) -> UserState:
        """Parse a state from a string, raising ``ValueError`` if unknown."""
        try:
            return cls(value.upper())
        except KeyError:
            valid = ", ".join(s.value for s in cls)
            raise ValueError(f"unknown user state: {value!r} (expected one of {valid})") from None
