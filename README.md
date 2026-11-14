# HOTB_2026

Heroes of The Brain Neurohackathon 2026!

## Architecture

The system is split into three main packages, each with a single responsibility.
More packages (UI layers, additional connections, etc.) may be added later.

    src/hotb_2026/
    ├── brainaccess_connection/   # connection to BrainAccess EEG hardware
    ├── reve_processor/           # REVE processing of the incoming signal
    └── state_aware_agent/        # state-aware decision-making and agent logic

- **`brainaccess_connection`** — handles connecting to and streaming data from the
  BrainAccess EEG device.
- **`reve_processor`** — processes the incoming signal (REVE) and produces clean,
  usable data for downstream packages.
- **`state_aware_agent`** — consumes the processed state and drives the
  state-aware agent behavior.

The three packages are not the full set — new packages (e.g. UI, other
connections) can be added alongside them in `src/hotb_2026/`.

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
