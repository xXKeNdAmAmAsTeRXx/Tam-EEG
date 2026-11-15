"""Pet plugin registry.

Adding a new creature means adding one module to this package (or
calling :func:`register_pet` at runtime) — the engine, daemon and CLI
need no changes.  Every ``*.py`` module next to this ``__init__`` is
imported automatically, so a plugin is simply a file that calls
``register_pet`` once.  A plugin provides:

``art``
    ``{stage: {mood: ASCII art string}}`` — art per life stage × mood.
``decay_rates``
    Stat decay **per hour**: ``{"hunger": ..., "energy": ..., "hygiene": ...}``.
``evolution_thresholds``
    ``{target_stage: {"seconds": ..., "avg": ...}}`` — evolve when the pet
    spent ``seconds`` in the current stage AND its stat average is at least
    ``avg``.
"""

from __future__ import annotations

import pkgutil
from typing import Any, TypedDict

from hotb_2026.tamagotchi.core.pet import Mood, Stage


class PetParams(TypedDict):
    art: dict[str, dict[str, str]]
    decay_rates: dict[str, float]
    evolution_thresholds: dict[str, dict[str, float]]


PETS: dict[str, PetParams] = {}


def register_pet(name: str, art: dict, params: dict[str, Any]) -> str:
    """Register a new pet plugin at runtime.

    Returns the normalised plugin name.  Raises ``ValueError`` when the
    art/params do not cover every stage × mood combination.
    """
    name = name.strip().lower()
    if not name:
        raise ValueError("pet name must not be empty")

    stages = [s.value for s in Stage]
    moods = [m.value for m in Mood]
    for stage in stages:
        if stage not in art:
            raise ValueError(f"art for pet {name!r} is missing stage {stage}")
        for mood in moods:
            if mood not in art[stage] or not art[stage][mood].strip():
                raise ValueError(f"art for pet {name!r} is missing {stage}/{mood}")

    for stat in ("hunger", "energy", "hygiene"):
        if stat not in params.get("decay_rates", {}):
            raise ValueError(f"pet {name!r} is missing decay rate for {stat}")
    if not params.get("evolution_thresholds"):
        raise ValueError(f"pet {name!r} is missing evolution thresholds")

    PETS[name] = {
        "art": {s: {m: art[s][m] for m in moods} for s in stages},
        "decay_rates": {k: float(v) for k, v in params["decay_rates"].items()},
        "evolution_thresholds": {
            target: {"seconds": float(c["seconds"]), "avg": float(c["avg"])}
            for target, c in params["evolution_thresholds"].items()
        },
    }
    return name


def get_pet_params(pet_kind: str) -> PetParams:
    """Look up plugin params, raising ``KeyError`` for unknown kinds."""
    try:
        return PETS[pet_kind.strip().lower()]
    except KeyError:
        known = ", ".join(sorted(PETS))
        raise KeyError(f"unknown pet kind {pet_kind!r} (known: {known})") from None


def list_pets() -> list[str]:
    """Names of all registered pet plugins."""
    return sorted(PETS)

# -- auto-import every plugin module in this package ----------------------
# Import order does not matter: each module only registers itself in PETS.
for _module_info in pkgutil.iter_modules(__path__):
    __import__(f"{__name__}.{_module_info.name}")
