"""Background daemon: periodic decay → mood → evolution → save + render."""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

from hotb_2026.tamagotchi.core.constants import DEFAULT_TICK_SECONDS, PID_FILE_NAME, state_dir
from hotb_2026.tamagotchi.render.display import render
from hotb_2026.tamagotchi.core.engine import tick
from hotb_2026.tamagotchi.core.pet import Pet

_running = True


def _handle_signal(signum: int, frame: object) -> None:
    global _running
    _running = False


def _pid_path(directory: Path) -> Path:
    return directory / PID_FILE_NAME


def run(tick_seconds: float = DEFAULT_TICK_SECONDS) -> None:
    """Run the daemon loop until SIGTERM/SIGINT.

    Requires an existing ``pet.json``; call ``python -m hotb_2026.tamagotchi
    init`` first.  Uses no third-party process tooling: the pid-file plus
    :func:`is_running` are enough to control the daemon.
    """
    global _running
    directory = state_dir()
    pet = Pet.load_or_none(directory)
    if pet is None:
        sys.exit("no pet found — run `python -m hotb_2026.tamagotchi init` first")

    directory.mkdir(parents=True, exist_ok=True)
    _pid_path(directory).write_text(str(os.getpid()) + "\n", encoding="utf-8")

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    _running = True
    try:
        while _running:
            tick(pet)
            render(pet, directory)
            pet.save(directory)
            # Interruptible sleep: check the stop flag once per second.
            deadline = time.monotonic() + tick_seconds
            while _running and time.monotonic() < deadline:
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    finally:
        _pid_path(directory).unlink(missing_ok=True)


def is_running(directory: Path | None = None) -> bool:
    """True when a daemon pid-file points at a live process (no psutil)."""
    path = _pid_path(Path(directory) if directory else state_dir())
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
