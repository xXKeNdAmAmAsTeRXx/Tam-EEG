from __future__ import annotations

import pytest

from hotb_2026.tamagotchi import (
    HAPPINESS_EFFECT,
    UserState,
    apply_state,
    check_evolution,
    decay,
    new_pet,
    resolve_mood,
)
from hotb_2026.tamagotchi.core.pet import Mood, Pet, Stage, Stats


def make_pet(last_update: float = 0.0, stage_started_at: float = 0.0) -> Pet:
    return Pet(
        name="Testy",
        pet_kind="cat",
        stats=Stats(happiness=50.0, hunger=80.0, energy=80.0, hygiene=80.0),
        last_update=last_update,
        stage_started_at=stage_started_at,
    )


class TestApplyState:
    @pytest.mark.parametrize(
        ("state", "expected_delta"),
        [
            (UserState.RELAX, -1.0),
            (UserState.FOCUS, 10.0),
            (UserState.OVERLOAD, -9.0),
        ],
    )
    def test_effects_match_constants(self, state, expected_delta) -> None:
        assert HAPPINESS_EFFECT[state] == expected_delta
        pet = make_pet()
        apply_state(pet, state)
        assert pet.stats.happiness == pytest.approx(50.0 + expected_delta)

    def test_happiness_clamped_to_bounds(self) -> None:
        pet = make_pet()
        pet.stats.happiness = 5.0
        apply_state(pet, UserState.OVERLOAD)
        assert pet.stats.happiness == 0.0

        pet.stats.happiness = 95.0
        apply_state(pet, UserState.FOCUS)
        assert pet.stats.happiness == 100.0

    def test_only_happiness_changes(self) -> None:
        pet = make_pet()
        apply_state(pet, UserState.FOCUS)
        assert (pet.stats.hunger, pet.stats.energy, pet.stats.hygiene) == (
            80.0,
            80.0,
            80.0,
        )


class TestDecay:
    def test_no_decay_without_elapsed_time(self) -> None:
        pet = make_pet(last_update=100.0)
        decay(pet, now=100.0)
        assert pet.stats.hunger == 80.0

    def test_decay_is_proportional_to_elapsed_hours(self) -> None:
        # cat decay_rates: hunger 8/hour, energy 6/hour, hygiene 5/hour
        pet = make_pet(last_update=0.0)
        decay(pet, now=3600.0)  # exactly one hour later
        assert pet.stats.hunger == pytest.approx(72.0)
        assert pet.stats.energy == pytest.approx(74.0)
        assert pet.stats.hygiene == pytest.approx(75.0)
        assert pet.last_update == 3600.0

    def test_downtime_accumulates_full_drop(self) -> None:
        pet = make_pet(last_update=0.0)
        # One call after 10 hours of "daemon downtime" must apply 10 hours.
        decay(pet, now=10 * 3600.0)
        assert pet.stats.hunger == pytest.approx(0.0)  # 80 - 8*10
        assert pet.stats.hygiene == pytest.approx(30.0)  # 80 - 5*10

    def test_decay_clamps_at_zero(self) -> None:
        pet = make_pet(last_update=0.0)
        decay(pet, now=100 * 3600.0)
        assert pet.stats.hunger == 0.0

    def test_happiness_not_time_decayed(self) -> None:
        pet = make_pet(last_update=0.0)
        decay(pet, now=100 * 3600.0)
        assert pet.stats.happiness == 50.0


class TestEvolution:
    def test_needs_time_and_average(self) -> None:
        # High average but no time elapsed -> no evolution.
        pet = make_pet(stage_started_at=0.0)
        pet.stats = Stats(happiness=90.0, hunger=90.0, energy=90.0, hygiene=90.0)
        check_evolution(pet, now=100.0)
        assert pet.stage is Stage.EGG

        # Enough time but low average -> no evolution.
        pet = make_pet(stage_started_at=0.0)
        pet.stats = Stats(happiness=20.0, hunger=20.0, energy=20.0, hygiene=20.0)
        check_evolution(pet, now=10_000.0)
        assert pet.stage is Stage.EGG

        # Both conditions -> evolves, timestamp resets.
        pet = make_pet(stage_started_at=0.0)
        pet.stats = Stats(happiness=90.0, hunger=90.0, energy=90.0, hygiene=90.0)
        check_evolution(pet, now=10_000.0)
        assert pet.stage is Stage.BABY
        assert pet.stage_started_at == 10_000.0

    def test_evolution_stops_at_adult(self) -> None:
        pet = make_pet(stage_started_at=0.0)
        pet.stats = Stats(happiness=90.0, hunger=90.0, energy=90.0, hygiene=90.0)
        check_evolution(pet, now=10_000.0)
        check_evolution(pet, now=20_000.0)
        assert pet.stage is Stage.ADULT
        check_evolution(pet, now=1_000_000.0)
        assert pet.stage is Stage.ADULT


class TestMood:
    @pytest.mark.parametrize(
        ("stats", "mood"),
        [
            (Stats(90.0, 90.0, 90.0, 90.0), Mood.HAPPY),
            (Stats(60.0, 60.0, 60.0, 60.0), Mood.NEUTRAL),
            (Stats(40.0, 40.0, 40.0, 40.0), Mood.SAD),
            (Stats(50.0, 5.0, 50.0, 50.0), Mood.CRITICAL),
        ],
    )
    def test_resolve_mood(self, stats, mood) -> None:
        pet = make_pet()
        pet.stats = stats
        assert resolve_mood(pet) is mood


class TestNewPet:
    def test_new_pet_starts_as_egg(self) -> None:
        pet = new_pet("Brainy", "cat")
        assert pet.stage is Stage.EGG
        assert pet.name == "Brainy"

    def test_new_pet_rejects_unknown_kind(self) -> None:
        with pytest.raises(KeyError):
            new_pet("Brainy", "unicorn")
