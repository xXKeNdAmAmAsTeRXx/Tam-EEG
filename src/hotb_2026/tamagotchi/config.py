"""Simple, user-tweakable defaults for the neuro-tamagotchi.

Edit the values below (or override them with CLI flags / environment
variables) instead of hunting through the runtime code.  Nothing here
imports the rest of the package, so it is safe to import from anywhere.
"""

from __future__ import annotations

from pathlib import Path

#: Default pet name and kind used by ``init``.
DEFAULT_PET_NAME = "Brainy"
DEFAULT_PET_KIND = "cat"

#: Care-command defaults (units are stat points, 0-100 scale).
DEFAULT_FEED_AMOUNT = 20.0
DEFAULT_PLAY_AMOUNT = 10.0
DEFAULT_CLEAN_AMOUNT = 25.0

#: Background daemon tick interval, in seconds.
DEFAULT_TICK_SECONDS = 60.0

#: Environment variable that overrides the state directory.
STATE_DIR_ENV = "HOTB_TAMAGOTCHI_HOME"

#: Default location of ``pet.json``, ``status.txt`` and ``daemon.pid``.
DEFAULT_STATE_DIR = Path.home() / ".hotb_tamagotchi"