from __future__ import annotations

import json

import pytest

from hotb_2026.tamagotchi.runtime.cli import main as cli_main
from hotb_2026.tamagotchi.core.constants import state_dir
from hotb_2026.tamagotchi.render.display import build_frame, render
from hotb_2026.tamagotchi.core.pet import Pet, Stage, Stats


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOTB_TAMAGOTCHI_HOME", str(tmp_path))
    return tmp_path


def test_roundtrip_save_load(home) -> None:
    from hotb_2026.tamagotchi.core.engine import new_pet

    pet = new_pet("Brainy", "slime")
    pet.stats.happiness = 73.5
    path = pet.save()
    loaded = Pet.load()
    assert loaded.to_dict() == json.loads(path.read_text(encoding="utf-8"))
    assert loaded.stats.happiness == 73.5
    assert loaded.stage is Stage.EGG


def test_register_pet_adds_new_creature(home) -> None:
    from hotb_2026.tamagotchi.pets import PETS, get_pet_params, register_pet

    name = register_pet(
        "unicorn",
        art={
            stage: {mood: "  (u)" for mood in ("HAPPY", "NEUTRAL", "SAD", "CRITICAL")}
            for stage in ("EGG", "BABY", "ADULT")
        },
        params={
            "decay_rates": {"hunger": 1.0, "energy": 1.0, "hygiene": 1.0},
            "evolution_thresholds": {"BABY": {"seconds": 60.0, "avg": 30.0}},
        },
    )
    assert name == "unicorn"
    assert "unicorn" in PETS
    assert get_pet_params("unicorn")["decay_rates"]["hunger"] == 1.0

    from hotb_2026.tamagotchi.core.engine import new_pet

    pet = new_pet("Sparkle", "unicorn")
    assert build_frame(pet)  # renders without errors


def test_register_pet_validates_art(home) -> None:
    from hotb_2026.tamagotchi.pets import register_pet

    incomplete_art = {stage: {"HAPPY": "x"} for stage in ("EGG", "BABY", "ADULT")}
    with pytest.raises(ValueError):
        register_pet("bad_pet", incomplete_art, params={
            "decay_rates": {"hunger": 1.0, "energy": 1.0, "hygiene": 1.0},
            "evolution_thresholds": {"BABY": {"seconds": 60.0, "avg": 30.0}},
        })


def test_render_plain_has_no_ansi_leak(home) -> None:
    from hotb_2026.tamagotchi.core.engine import new_pet

    pet = new_pet("Brainy", "cat")
    plain = build_frame(pet, use_ansi=False)
    assert "\033[" not in plain
    assert "happiness" in plain
    assert "Brainy" in plain

    path = render(pet)
    assert path == state_dir() / "status.txt"
    content = path.read_text(encoding="utf-8")
    assert "\033[" not in content

    # ANSI variant exists for terminals.
    assert "\033[" in build_frame(pet, use_ansi=True)


def run_cli(*argv: str) -> None:
    cli_main(list(argv))


def test_cli_smoke_flow(home, capsys) -> None:
    run_cli("init", "--name", "Brainy", "--pet", "slime")
    assert "created Brainy the slime" in capsys.readouterr().out

    pet = Pet.load()
    assert pet.name == "Brainy"
    assert pet.stats.happiness == 50.0

    run_cli("state", "FOCUS")
    assert Pet.load().stats.happiness == 60.0

    run_cli("state", "OVERLOAD")
    assert Pet.load().stats.happiness == 51.0

    run_cli("state", "RELAX")
    assert Pet.load().stats.happiness == 50.0

    run_cli("feed", "--amount", "30")
    # A hair of time decay passes between the CLI call and the load.
    assert Pet.load().stats.hunger > 99.9

    run_cli("play")
    assert Pet.load().stats.happiness == 60.0

    run_cli("clean")
    assert Pet.load().stats.hygiene > 99.9

    out = capsys.readouterr().out
    run_cli("status")
    assert "Brainy the slime" in capsys.readouterr().out
    assert out


def test_cli_state_rejects_unknown_state(home, capsys) -> None:
    run_cli("init")
    capsys.readouterr()
    with pytest.raises(ValueError):
        run_cli("state", "PANIC")


def test_cli_requires_existing_pet(home) -> None:
    with pytest.raises(SystemExit):
        run_cli("status")
