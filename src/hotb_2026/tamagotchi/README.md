# Neuro-tamagotchi 🥚

A tamagotchi whose happiness is driven by **your brain** — the EEG
classifier reports a `UserState` (`RELAX` / `FOCUS` / `OVERLOAD`) and the
pet reacts. Focused work makes it happy; overload makes it miserable.

The architecture is modelled on *tamaterm* (background daemon + state
file + shell hook + CLI), **minus** everything non-neuro: there are no
commit counters, git hooks, CPU measurements or `psutil` — the *only*
input that moves happiness is the reported `UserState`.

All commands below use [uv](https://docs.astral.sh/uv/), the project's
package manager (`uv sync` first, if you haven't).

## Quick start

```bash
uv run python -m hotb_2026.tamagotchi init --name Brainy --pet cat
uv run python -m hotb_2026.tamagotchi daemon          # background loop (Ctrl+C to stop)
uv run python -m hotb_2026.tamagotchi status
uv run python -m hotb_2026.tamagotchi install-hook    # show the pet before each prompt

# care for your pet
uv run python -m hotb_2026.tamagotchi feed
uv run python -m hotb_2026.tamagotchi play
uv run python -m hotb_2026.tamagotchi clean
```

State lives in `~/.hotb_tamagotchi/` (`pet.json`, `status.txt`,
`daemon.pid`); override the directory with `HOTB_TAMAGOTCHI_HOME`.

## Package layout

```text
src/hotb_2026/tamagotchi/
├── __init__.py        # public API re-exports
├── __main__.py        # `python -m hotb_2026.tamagotchi` entry point
├── README.md
├── config.py          # simple, user-tweakable defaults
├── core/              # state + game logic — zero rendering, zero OS integration
│   ├── states.py      #   UserState enum (RELAX, FOCUS, OVERLOAD)
│   ├── constants.py   #   HAPPINESS_EFFECT; stat bounds; paths
│   ├── pet.py         #   Pet + Stats (clamped 0–100), Stage (EGG→BABY→ADULT), JSON persistence
│   └── engine.py      #   apply_state, decay, resolve_mood, check_evolution, tick
├── pets/              # animal plugins — ONE file per creature (art + params)
│   ├── __init__.py    #   registry: auto-imports every module, register_pet()
│   ├── cat.py
│   ├── dog.py
│   └── slime.py
├── render/            # presentation
│   ├── display.py     #   build_frame → status.txt (plain fallback, optional ANSI colours)
│   └── install.py     #   idempotent `# >>> hotb tamagotchi >>>` hook in .bashrc/.zshrc
└── runtime/           # run processes
    ├── daemon.py      #   background tick loop (default 60 s, configurable), pid-file
    └── cli.py         #   init/feed/play/clean/status/state/daemon/install-hook
```

## CLI reference

| Command | Options | Effect |
|---|---|---|
| `init` | `--name` (default `Brainy`), `--pet` (default `cat`), `--force` | create a new pet and render `status.txt` |
| `feed` | `--amount` (default `20`) | `hunger += amount` |
| `play` | `--amount` (default `10`) | `happiness += amount`, `energy -= amount / 2` |
| `clean` | `--amount` (default `25`) | `hygiene += amount` |
| `status` | — | print the current frame |
| `state` | `RELAX` / `FOCUS` / `OVERLOAD` | apply `HAPPINESS_EFFECT`, then tick |
| `daemon` | `--tick` (default `60`) | run the background loop in the foreground |
| `install-hook` | — | add the prompt hook to `.bashrc` / `.zshrc` |

Every care command runs `tick()` (decay + evolution) before saving and
re-rendering, so stats are always up to date. All stats are clamped to
`0–100`.

## Rules of the game

* `HAPPINESS_EFFECT = {RELAX: -1, FOCUS: +10, OVERLOAD: -9}` — applied
  **per reported state**, not per tick. This dict (in
  `core/constants.py`) is the only place that defines the mapping.
* `decay(now)` uses wall-clock timestamps, so a daemon outage is not
  "free time": after a pause one call applies the full accumulated drop
  of hunger / energy / hygiene (rates are per-hour, defined per pet).
  Happiness is *never* time-decayed.
* Evolution requires **time in stage AND stat average ≥ threshold**
  (thresholds are per-pet plugin params).
* Moods: `CRITICAL` (hunger / energy / hygiene ≤ 10), then `HAPPY` /
  `NEUTRAL` / `SAD` by stat average.

## Example frame

```text
Brainy the cat — EGG [NEUTRAL]

   _____
  ( ovo )
   -----

  happiness [##########----------]  50.0
  hunger   [################----]  80.0
  energy   [################----]  80.0
  hygiene  [################----]  80.0
```

## State & persistence

State lives in `~/.hotb_tamagotchi/`:

- `pet.json` — persisted pet (name, kind, stage, stats, timestamps)
- `status.txt` — last rendered frame (used by the shell hook)
- `daemon.pid` — pid-file used to detect a running daemon (no `psutil`)

Override the directory with `HOTB_TAMAGOTCHI_HOME` (used by tests and
notebooks). Defaults such as tick interval and care amounts live in
`config.py`.

## Python API — integrating with the EEG loop

The classifier lives in `hotb_2026.classification`; the tamagotchi does
not import it. Whenever your pipeline produces a state, push it in:

```python
from hotb_2026.tamagotchi import Pet, UserState, apply_state, resolve_mood

pet = Pet.load()

# E.g. inside your streamer callback, after window classification:
state = classifier.predict(window)        # -> "FOCUS" / "RELAX" / "OVERLOAD"
apply_state(pet, UserState.from_value(state))
mood = resolve_mood(pet)
print(f"{pet.name} is {mood.value} (happiness {pet.stats.happiness:.0f})")

pet.save()                                 # daemon/CLI pick it up from here
```

Core helpers: `apply_state`, `decay`, `resolve_mood`, `check_evolution`,
`tick`, `new_pet`. Models: `Pet`, `Stats`, `Stage`, `Mood`, `UserState`,
`clamp`. Rendering: `build_frame`, `render`. Plugins: `register_pet`,
`get_pet_params`, `list_pets`, `PETS`.

For a runnable demo see
`notebooks/tamagotchi/tamagotchi_demo.ipynb` (mock classifier, states
applied in sequence, frames printed).

## Adding a new pet (plugin system)

A pet is just a module in `pets/` that calls `register_pet` (or a
`register_pet` call from your own code) — the engine, daemon and CLI
need no changes. One plugin = ASCII art per **stage × mood** plus
parameters. Drop `pets/unicorn.py` into the package:

```python
"""The ``unicorn`` pet plugin."""
from hotb_2026.tamagotchi.pets import register_pet

register_pet(
    "unicorn",
    art={
        "EGG":   {"HAPPY": " (u)", "NEUTRAL": " (u)", "SAD": " (u)", "CRITICAL": " (x)"},
        "BABY":  {"HAPPY": "/u\\", "NEUTRAL": "/u\\", "SAD": "/u\\", "CRITICAL": "/x\\"},
        "ADULT": {"HAPPY": "<u>",  "NEUTRAL": "<u>",  "SAD": "<u>",  "CRITICAL": "<x>"},
    },
    params={
        "decay_rates": {"hunger": 6.0, "energy": 5.0, "hygiene": 4.0},   # per hour
        "evolution_thresholds": {
            "BABY":  {"seconds": 300.0,  "avg": 40.0},
            "ADULT": {"seconds": 3600.0, "avg": 60.0},
        },
    },
)
```

`pets/__init__.py` auto-imports every module next to it, so the new
creature is usable immediately:

```bash
uv run python -m hotb_2026.tamagotchi init --pet unicorn
```

Built-in plugins: `cat`, `dog`, `slime`.

## Shell hook

`install-hook` appends a marked block to `~/.bashrc` / `~/.zshrc`:

```bash
# >>> hotb tamagotchi >>>
if [ -f "$HOME/.hotb_tamagotchi/status.txt" ]; then
    cat "$HOME/.hotb_tamagotchi/status.txt"
fi
# <<< hotb tamagotchi <<<
```

The markers make it idempotent — rerunning updates the block instead of
duplicating it.

## Tests

```bash
uv run pytest tests/test_tamagotchi_engine.py tests/test_tamagotchi_cli.py
```

Tests use `HOTB_TAMAGOTCHI_HOME` to run against a temporary directory.

## Troubleshooting

- `no pet found` — run `init` first.
- `daemon already running` — a pid-file exists for a live process; stop it
  or remove a stale pid-file.
- `unknown user state` / `unknown pet kind` — check spelling; states are
  `RELAX`, `FOCUS`, `OVERLOAD`; pets are `cat`, `dog`, `slime`.
- Frame shows ANSI codes in files — render uses plain text by default;
  ANSI is only enabled with `use_ansi=True`.
- Need a fresh pet — `init --force` overwrites the existing one.