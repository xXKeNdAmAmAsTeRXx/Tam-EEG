# HOTB_2026

Heroes of The Brain Neurohackathon 2026!

## Architecture

The system is split into several packages, each with a single responsibility.
More packages (UI layers, additional connections, etc.) may be added later.

    src/hotb_2026/
    ├── brainaccess_connection/   # connection to BrainAccess EEG hardware
    ├── preprocessing/            # per-window cleaning and standardization
    ├── classification/           # REVE embeddings + SVM classifier
    ├── mappers/                  # adapters between package models
    ├── record/                   # PsychoPy n-back recording sessions
    └── state_aware_agent/        # planned: state-aware decision-making

- **`brainaccess_connection`** — handles connecting to and streaming data from the
  BrainAccess EEG device. See
  [`src/hotb_2026/brainaccess_connection/README.md`](src/hotb_2026/brainaccess_connection/README.md)
  for hardware notes, CLI usage, and the LSL stream layout.
- **`preprocessing`** — cleans each completed streamer window: notch 50 Hz →
  band-pass 0.5–45 Hz → resample 250 → 200 Hz → z-score → clip ±15. See
  [`src/hotb_2026/preprocessing/README.md`](src/hotb_2026/preprocessing/README.md)
  and the demo notebook
  [`notebooks/preprocessing/01_preprocessing.ipynb`](notebooks/preprocessing/01_preprocessing.ipynb).
- **`classification`** — embeds each preprocessed window with a frozen REVE
  transformer (`brain-bzh/reve-base`) and classifies the pooled embedding with a
  scikit-learn SVM. See
  [`src/hotb_2026/classification/model/ReveSVMClassifier.py`](src/hotb_2026/classification/model/ReveSVMClassifier.py)
  and the REVE notebook
  [`notebooks/reve_model/01_reve_embddings.ipynb`](notebooks/reve_model/01_reve_embddings.ipynb).
- **`mappers`** — small adapters between package models, e.g.
  `montage_to_list_electrode` turns a `Montage` into the electrode-name list
  REVE's position encoding expects.
- **`record`** — runs the PsychoPy n-back procedure (rest / 1-back / 3-back)
  and records EEG sessions. See
  [`src/hotb_2026/record/README.md`](src/hotb_2026/record/README.md).
- **`state_aware_agent`** — planned, not implemented yet. Will consume the
  classified state and drive the agent behavior.

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

## Quick start: recorded n-back sessions

```bash
uv run hotb-record --simulate --subject demo --sessions 1
```

Each session is written to `data/session_<n>/` with `meta.json`, `signal.npy`,
`time.npy`, and `state_changes.csv`. Instruction/feedback periods are cut out of
the recorded signal. See
[`src/hotb_2026/record/README.md`](src/hotb_2026/record/README.md).

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
    fix(classification): handle empty signal
    chore(readme): add architecture section
