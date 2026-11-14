"""Game engine: state application, time-based decay, mood and evolution."""

from __future__ import annotations

import time

from hotb_2026.tamagotchi.pets import get_pet_params
from hotb_2026.tamagotchi.core.constants import HAPPINESS_EFFECT, STAT_MAX
from hotb_2026.tamagotchi.core.pet import Mood, Pet, Stage, Stats
from hotb_2026.tamagotchi.core.states import UserState


def apply_state(pet: Pet, state: UserState) -> Pet:
    """Apply a classifier-reported state to the pet.

    Happiness changes by exactly ``HAPPINESS_EFFECT[state]`` *per reported
    state* (not per tick); all other stats are untouched.  The returned pet
    is the same object, mutated in place for convenience.
    """
    pet.stats.happiness = max(0.0, min(STAT_MAX, pet.stats.happiness + HAPPINESS_EFFECT[state]))
    return pet


def decay(pet: Pet, now: float | None = None) -> Pet:
    """Advance hunger/energy/hygiene by the elapsed wall-clock time.

    Rates are defined *per hour* in the pet plugin params, so the decay is
    robust to daemon downtime: one call after a long pause applies the full
    accumulated drop.  Happiness is *not* time-decayed; only ``apply_state``
    changes it.
    """
    now = time.time() if now is None else now
    elapsed_hours = max(0.0, (now - pet.last_update) / 3600.0)
    rates = get_pet_params(pet.pet_kind)["decay_rates"]
    for stat in ("hunger", "energy", "hygiene"):
        rate = float(rates[stat])
        value = getattr(pet.stats, stat) - rate * elapsed_hours
        setattr(pet.stats, stat, max(0.0, value))
    pet.last_update = now
    return pet


def resolve_mood(pet: Pet) -> Mood:
    """Derive the mood from the current stats."""
    stats = pet.stats
    if stats.is_critical():
        return Mood.CRITICAL
    if stats.average >= 75.0:
        return Mood.HAPPY
    if stats.average >= 50.0:
        return Mood.NEUTRAL
    return Mood.SAD


def check_evolution(pet: Pet, now: float | None = None) -> Pet:
    """Evolve the pet when time-in-stage AND stat-average conditions hold.

    Thresholds come from the pet plugin params:
    ``evolution_thresholds[Stage] = {"seconds": ..., "avg": ...}``.
    """
    now = time.time() if now is None else now
    params = get_pet_params(pet.pet_kind)
    thresholds = params["evolution_thresholds"]
    while True:
        nxt = pet.stage.next
        if nxt is None:
            return pet
        requirement = thresholds.get(nxt.value)
        if requirement is None:
            return pet
        seconds_in_stage = now - pet.stage_started_at
        if (
            seconds_in_stage >= requirement["seconds"]
            and pet.stats.average >= requirement["avg"]
        ):
            pet.stage = nxt
            pet.stage_started_at = now
        else:
            return pet


def tick(pet: Pet, now: float | None = None) -> Pet:
    """One full engine step: decay → evolution."""
    decay(pet, now)
    check_evolution(pet, now)
    return pet


def new_pet(name: str, pet_kind: str) -> Pet:
    """Create a fresh egg of the given kind."""
    get_pet_params(pet_kind)  # validate the kind exists
    now = time.time()
    return Pet(
        name=name,
        pet_kind=pet_kind,
        stage=Stage.EGG,
        stats=Stats(),
        last_update=now,
        stage_started_at=now,
    )
