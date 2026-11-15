"""Command-line interface for the tamagotchi."""

from __future__ import annotations

import argparse
import sys

from hotb_2026.tamagotchi.config import (
    DEFAULT_CLEAN_AMOUNT,
    DEFAULT_FEED_AMOUNT,
    DEFAULT_PET_KIND,
    DEFAULT_PET_NAME,
    DEFAULT_PLAY_AMOUNT,
    DEFAULT_TICK_SECONDS,
)
from hotb_2026.tamagotchi.render import install
from hotb_2026.tamagotchi.pets import list_pets
from hotb_2026.tamagotchi.core.constants import state_dir
from hotb_2026.tamagotchi.runtime.daemon import is_running, run
from hotb_2026.tamagotchi.render.display import build_frame, render
from hotb_2026.tamagotchi.core.engine import apply_state, tick
from hotb_2026.tamagotchi.core.pet import clamp
from hotb_2026.tamagotchi.core.states import UserState


def _load_pet():
    from hotb_2026.tamagotchi.core.pet import Pet

    pet = Pet.load_or_none()
    if pet is None:
        sys.exit("no pet found — run `python -m hotb_2026.tamagotchi init` first")
    return pet


def cmd_init(args: argparse.Namespace) -> None:
    from hotb_2026.tamagotchi.core.engine import new_pet

    directory = state_dir()
    if not args.force and (directory / "pet.json").exists():
        sys.exit("pet already exists (use --force to overwrite)")
    pet = new_pet(args.name, args.pet)
    pet.save(directory)
    render(pet, directory)
    print(f"created {pet.name} the {pet.pet_kind} ({pet.stage.value})")
    print(f"state dir: {directory}")


def cmd_feed(args: argparse.Namespace) -> None:
    pet = _load_pet()
    pet.stats.hunger = clamp(pet.stats.hunger + args.amount)
    tick(pet)
    pet.save()
    render(pet)
    print(f"{pet.name} was fed (+{args.amount} hunger)")


def cmd_play(args: argparse.Namespace) -> None:
    pet = _load_pet()
    pet.stats.happiness = clamp(pet.stats.happiness + args.amount)
    pet.stats.energy = clamp(pet.stats.energy - args.amount / 2.0)
    tick(pet)
    pet.save()
    render(pet)
    print(f"{pet.name} played with you (+{args.amount} happiness)")


def cmd_clean(args: argparse.Namespace) -> None:
    pet = _load_pet()
    pet.stats.hygiene = clamp(pet.stats.hygiene + args.amount)
    tick(pet)
    pet.save()
    render(pet)
    print(f"{pet.name} is clean again (+{args.amount} hygiene)")


def cmd_status(args: argparse.Namespace) -> None:
    pet = _load_pet()
    print(build_frame(pet))


def cmd_state(args: argparse.Namespace) -> None:
    """Feed a classifier state into the engine (primarily for tests)."""
    pet = _load_pet()
    state = UserState.from_value(args.state)
    apply_state(pet, state)
    tick(pet)
    pet.save()
    render(pet)
    print(f"applied {state.value}: happiness is now {pet.stats.happiness:.1f}")


def cmd_daemon(args: argparse.Namespace) -> None:
    if is_running():
        sys.exit("daemon already running")
    print(f"running tamagotchi daemon (tick: {args.tick}s), Ctrl+C to stop")
    run(tick_seconds=args.tick)


def cmd_install_hook(args: argparse.Namespace) -> None:
    for shell_file in install.install_hook():
        print(f"hook installed in {shell_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m hotb_2026.tamagotchi",
        description="Neuro-tamagotchi: happiness driven by your EEG UserState.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="create a new pet")
    p.add_argument("--name", default=DEFAULT_PET_NAME)
    p.add_argument(
        "--pet",
        default=DEFAULT_PET_KIND,
        metavar="KIND",
        help=f"pet plugin to use (one of: {', '.join(list_pets())})",
    )
    p.add_argument("--force", action="store_true", help="overwrite an existing pet")
    p.set_defaults(func=cmd_init)

    for name, amount, help_text in (
        ("feed", DEFAULT_FEED_AMOUNT, "feed the pet (hunger up)"),
        ("clean", DEFAULT_CLEAN_AMOUNT, "clean the pet (hygiene up)"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--amount", type=float, default=amount)
        p.set_defaults(func=getattr(sys.modules[__name__], f"cmd_{name}"))

    p = sub.add_parser("play", help="play with the pet (happiness up)")
    p.add_argument("--amount", type=float, default=DEFAULT_PLAY_AMOUNT)
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("status", help="print the current status frame")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("state", help="apply a UserState (RELAX/FOCUS/OVERLOAD)")
    p.add_argument("state", help="one of RELAX, FOCUS, OVERLOAD")
    p.set_defaults(func=cmd_state)

    p = sub.add_parser("daemon", help="run the background loop in the foreground")
    p.add_argument("--tick", type=float, default=DEFAULT_TICK_SECONDS, help="tick seconds")
    p.set_defaults(func=cmd_daemon)

    p = sub.add_parser("install-hook", help="install the shell prompt hook")
    p.set_defaults(func=cmd_install_hook)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
