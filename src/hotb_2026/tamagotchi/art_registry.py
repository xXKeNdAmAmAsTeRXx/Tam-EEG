"""Pet plugin registry.

Adding a new creature means adding one ``PETS`` entry (or calling
:func:`register_pet` at runtime) — the engine, daemon and CLI need no
changes.  A plugin provides:

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

from typing import Any, TypedDict

from hotb_2026.tamagotchi.pet import Mood, Stage


class PetParams(TypedDict):
    art: dict[str, dict[str, str]]
    decay_rates: dict[str, float]
    evolution_thresholds: dict[str, dict[str, float]]


PETS: dict[str, PetParams] = {
    "cat": {
        "art": {
            "EGG": {
                "HAPPY": "   _____\n  ( ovo )\n   -----",
                "NEUTRAL": "   _____\n  ( ovo )\n   -----",
                "SAD": "   _____\n  ( ovo )\n   -----",
                "CRITICAL": "   _____\n  ( xvx )\n   -----",
            },
            "BABY": {
                "HAPPY": "  /\\_/\\\n ( ^w^ )\n",
                "NEUTRAL": "  /\\_/\\\n ( -.- )\n",
                "SAD": "  /\\_/\\\n ( ;_; )\n",
                "CRITICAL": "  /\\_/\\\n ( x x )\n",
            },
            "ADULT": {
                "HAPPY": "  /\\_/\\_______\n ( ^w^ )      )~\n",
                "NEUTRAL": "  /\\_/\\_______\n ( -.- )      )~\n",
                "SAD": "  /\\_/\\_______\n ( ;_; )      )~\n",
                "CRITICAL": "  /\\_/\\_______\n ( x x )      )~\n",
            },
        },
        "decay_rates": {"hunger": 8.0, "energy": 6.0, "hygiene": 5.0},
        "evolution_thresholds": {
            "BABY": {"seconds": 300.0, "avg": 40.0},
            "ADULT": {"seconds": 3600.0, "avg": 60.0},
        },
    },
    "dog": {
        "art": {
            "EGG": {
                "HAPPY": "   .-----.\n  (  ovo  )\n   '-----'",
                "NEUTRAL": "   .-----.\n  (  ovo  )\n   '-----'",
                "SAD": "   .-----.\n  (  ovo  )\n   '-----'",
                "CRITICAL": "   .-----.\n  (  xvx  )\n   '-----'",
            },
            "BABY": {
                "HAPPY": "  / \\__\n (    @\\\n",
                "NEUTRAL": "  / \\__\n (    -\\\n",
                "SAD": "  / \\__\n (    ;\\\n",
                "CRITICAL": "  / \\__\n (    x\\\n",
            },
            "ADULT": {
                "HAPPY": "  / \\_____/ \\\n ( ^w^      )\n",
                "NEUTRAL": "  / \\_____/ \\\n ( -.-      )\n",
                "SAD": "  / \\_____/ \\\n ( ;_;      )\n",
                "CRITICAL": "  / \\_____/ \\\n ( x x      )\n",
            },
        },
        "decay_rates": {"hunger": 10.0, "energy": 8.0, "hygiene": 6.0},
        "evolution_thresholds": {
            "BABY": {"seconds": 300.0, "avg": 40.0},
            "ADULT": {"seconds": 3600.0, "avg": 60.0},
        },
    },
    "slime": {
        "art": {
            "EGG": {
                "HAPPY": "   .-.\n  ( o )\n   '-'",
                "NEUTRAL": "   .-.\n  ( o )\n   '-'",
                "SAD": "   .-.\n  ( o )\n   '-'",
                "CRITICAL": "   .-.\n  ( x )\n   '-'",
            },
            "BABY": {
                "HAPPY": "  .-~-.\n ( ^w^ )\n  '~ ~'",
                "NEUTRAL": "  .-~-.\n ( -.- )\n  '~ ~'",
                "SAD": "  .-~-.\n ( ;.; )\n  '~ ~'",
                "CRITICAL": "  .-~-.\n ( x.x )\n  '~ ~'",
            },
            "ADULT": {
                "HAPPY": "  .-~~~~-.\n (  ^w^   )\n  '~~~~~~~'",
                "NEUTRAL": "  .-~~~~-.\n (  -.-   )\n  '~~~~~~~'",
                "SAD": "  .-~~~~-.\n (  ;.;   )\n  '~~~~~~~'",
                "CRITICAL": "  .-~~~~-.\n (  x.x   )\n  '~~~~~~~'",
            },
        },
        "decay_rates": {"hunger": 4.0, "energy": 4.0, "hygiene": 3.0},
        "evolution_thresholds": {
            "BABY": {"seconds": 300.0, "avg": 35.0},
            "ADULT": {"seconds": 3600.0, "avg": 50.0},
        },
    },
}


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
