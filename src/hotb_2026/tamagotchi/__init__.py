"""Neuro-tamagotchi: a pet whose happiness follows your EEG UserState.

Architecture is modelled on *tamaterm* (background daemon + state file +
shell hook + CLI), but the *only* influence on happiness is the
:class:`UserState` reported by the EEG classifier — there are no commit
counters, git hooks or CPU measurements.
"""

from __future__ import annotations

from hotb_2026.tamagotchi.art_registry import PETS, list_pets, register_pet
from hotb_2026.tamagotchi.constants import HAPPINESS_EFFECT
from hotb_2026.tamagotchi.display import build_frame, render
from hotb_2026.tamagotchi.engine import (
    apply_state,
    check_evolution,
    decay,
    new_pet,
    resolve_mood,
    tick,
)
from hotb_2026.tamagotchi.pet import Mood, Pet, Stage, Stats
from hotb_2026.tamagotchi.states import UserState

__all__ = [
    "HAPPINESS_EFFECT",
    "PETS",
    "Mood",
    "Pet",
    "Stage",
    "Stats",
    "UserState",
    "apply_state",
    "build_frame",
    "check_evolution",
    "decay",
    "list_pets",
    "new_pet",
    "register_pet",
    "render",
    "resolve_mood",
    "tick",
]
