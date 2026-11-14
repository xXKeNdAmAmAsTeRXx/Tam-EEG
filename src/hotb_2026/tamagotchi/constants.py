"""Shared constants for the tamagotchi package."""

from __future__ import annotations

import os
from pathlib import Path

from hotb_2026.tamagotchi.states import UserState

#: Happiness delta applied *per reported state* (not per tick).  This is the
#: single source of truth for how the classifier output influences the pet.
HAPPINESS_EFFECT: dict[UserState, float] = {
    UserState.RELAX: -1.0,
    UserState.FOCUS: 10.0,
    UserState.OVERLOAD: -9.0,
}

#: Stat bounds.
STAT_MIN = 0.0
STAT_MAX = 100.0

#: A stat at or below this level marks the pet as critical.
CRITICAL_THRESHOLD = 10.0

#: Default daemon tick, in seconds.
DEFAULT_TICK_SECONDS = 60.0


def state_dir() -> Path:
    """Directory holding ``pet.json``, ``status.txt`` and the pid-file.

    Overridable with the ``HOTB_TAMAGOTCHI_HOME`` environment variable so
    tests can run against a temporary directory.
    """
    override = os.environ.get("HOTB_TAMAGOTCHI_HOME")
    root = Path(override) if override else Path.home() / ".hotb_tamagotchi"
    return root


PET_FILE_NAME = "pet.json"
STATUS_FILE_NAME = "status.txt"
PID_FILE_NAME = "daemon.pid"

#: Markers used by :mod:`hotb_2026.tamagotchi.install` for idempotent hooks.
HOOK_BEGIN = "# >>> hotb tamagotchi >>>"
HOOK_END = "# <<< hotb tamagotchi <<<"
