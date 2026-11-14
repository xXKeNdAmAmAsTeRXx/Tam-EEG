"""Frame rendering: pet art + stat bars into ``status.txt``."""

from __future__ import annotations

from pathlib import Path

from hotb_2026.tamagotchi.art_registry import get_pet_params
from hotb_2026.tamagotchi.constants import STATUS_FILE_NAME, state_dir
from hotb_2026.tamagotchi.engine import resolve_mood
from hotb_2026.tamagotchi.pet import Pet

_ANSI = {
    "HAPPY": "\033[32m",  # green
    "NEUTRAL": "\033[36m",  # cyan
    "SAD": "\033[33m",  # yellow
    "CRITICAL": "\033[31m",  # red
}


def _bar(value: float, width: int = 20) -> str:
    filled = int(round(value / 100.0 * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def build_frame(pet: Pet, use_ansi: bool = False) -> str:
    """Render the full status frame as text.

    ``use_ansi=False`` produces a plain-text frame safe for files and
    piped output (the default); ``use_ansi=True`` colours the header by
    mood for interactive terminals.
    """
    mood = resolve_mood(pet)
    params = get_pet_params(pet.pet_kind)
    art = params["art"][pet.stage.value][mood.value]

    lines = [
        f"{pet.name} the {pet.pet_kind} — {pet.stage.value} [{mood.value}]",
        "",
        art,
        "",
        f"  happiness {_bar(pet.stats.happiness)} {pet.stats.happiness:5.1f}",
        f"  hunger   {_bar(pet.stats.hunger)} {pet.stats.hunger:5.1f}",
        f"  energy   {_bar(pet.stats.energy)} {pet.stats.energy:5.1f}",
        f"  hygiene  {_bar(pet.stats.hygiene)} {pet.stats.hygiene:5.1f}",
    ]
    frame = "\n".join(lines)
    if use_ansi:
        frame = _ANSI[mood.value] + frame + "\033[0m"
    return frame


def render(pet: Pet, directory: Path | None = None) -> Path:
    """Write ``status.txt`` (plain text, no ANSI codes) and return its path."""
    target_dir = Path(directory) if directory else state_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / STATUS_FILE_NAME
    path.write_text(build_frame(pet) + "\n", encoding="utf-8")
    return path
