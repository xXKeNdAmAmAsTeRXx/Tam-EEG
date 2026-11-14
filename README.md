# HOTB_2026

Heroes of The Brain Neurohackathon 2026!

## Architecture

The system is split into three main packages, each with a single responsibility.
More packages (UI layers, additional connections, etc.) may be added later.

    src/hotb_2026/
    ├── brainaccess_connection/   # connection to BrainAccess EEG hardware
    ├── preprocessing/            # per-window cleaning and standardization
    ├── reve_processor/           # REVE processing of the incoming signal
    ├── state_aware_agent/        # state-aware decision-making and agent logic
    └── tamagotchi/              # neuro-tamagotchi: pet driven by UserState

- **`brainaccess_connection`** — handles connecting to and streaming data from the
  BrainAccess EEG device. See
  [`src/hotb_2026/brainaccess_connection/README.md`](src/hotb_2026/brainaccess_connection/README.md)
  for hardware notes, CLI usage, and the LSL stream layout.
- **`preprocessing`** — cleans each completed streamer window: notch 50 Hz →
  band-pass 0.5–45 Hz → resample 250 → 200 Hz → z-score → clip ±15. See
  [`src/hotb_2026/preprocessing/README.md`](src/hotb_2026/preprocessing/README.md)
  and the demo notebook
  [`notebooks/preprocessing/01_preprocessing.ipynb`](notebooks/preprocessing/01_preprocessing.ipynb).
- **`reve_processor`** — processes the cleaned signal (REVE) and produces
  features for downstream packages.
- **`state_aware_agent`** — consumes the processed state and drives the
  state-aware agent behavior.
- **`tamagotchi`** — a pet whose happiness follows the reported
  `UserState` (daemon + state file + shell hook + CLI). See
  [`src/hotb_2026/tamagotchi/README.md`](src/hotb_2026/tamagotchi/README.md).

These packages are not the full set — new packages (e.g. UI, other connections)
can be added alongside them in `src/hotb_2026/`.

## Quick start: BrainAccess EEG

```bash
uv sync
uv run hotb-brainaccess --scan          # list nearby headsets (~5 s)
uv run hotb-brainaccess                 # stream the configured headset
uv run hotb-brainaccess --simulate      # 250 Hz fake signal, no hardware
```

The headset, electrode layout, rate and bias come from
[`config.py`](src/hotb_2026/brainaccess_connection/config.py) (default device:
`BA MINI 078`, 250 Hz). Edit that file to point at a different headset or cap,
or pass `--sfreq 500` for the higher hardware rate. The MINI supports 250 or
500 Hz only.

The stream is advertised over LSL as `BrainAccess EEG`, type `EEG`, 8 channels
in µV at the configured rate. Full CLI and troubleshooting docs live in the
[package README](src/hotb_2026/brainaccess_connection/README.md).

## Quick start: Neuro-tamagotchi

```bash
uv run python -m hotb_2026.tamagotchi init --name Brainy --pet cat
uv run python -m hotb_2026.tamagotchi daemon          # background loop (Ctrl+C to stop)
uv run python -m hotb_2026.tamagotchi status
```

See [`src/hotb_2026/tamagotchi/README.md`](src/hotb_2026/tamagotchi/README.md)
for the full CLI, care commands, and plugin guide.

## Git Convention

### Branches

    <name>/<TYPE>/<what_is_changed>

- `<name>` — your git username.
- `<TYPE>` — one of `FEAT`, `FIX`, `CHORE`.
- `<what_is_changed>` — short kebab-case description.

Examples:

    bajon1/FEAT/eeg-stream-connection
    bajon1/FIX/reve-parsing
    bajon1/CHORE/update-readme

### Commits

    <verb>(<scope>): <description>

- `<verb>` — `add`, `fix`, or `chore` (lowercase).
- `<scope>` — the affected package/module.
- `<description>` — lowercase, imperative.

Examples:

    add(brainaccess_connection): connect to eeg device
    fix(reve_processor): handle empty signal
    chore(readme): add architecture section
