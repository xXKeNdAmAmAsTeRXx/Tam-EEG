# Tam-EEG

**A framework for online EEG analysis and a cognitive-load-aware tamagotchi.**

Heroes of The Brain Neurohackathon 2026 project — **work in progress**.

Tam-EEG streams 8-channel EEG from a BrainAccess headset, classifies the
user's cognitive state (RELAX / FOCUS / OVERLOAD) in real time, and feeds
it to a virtual pet that lives in the terminal prompt: focused work makes
the pet thrive, overload makes it wilt.

```
BrainAccess EEG  →  preprocessing  →  REVE embeddings  →  SVM classifier
                                                                       │
                                              UserState (RELAX/FOCUS/OVERLOAD)
                                                                       ▼
                          terminal pet: happiness, mood, life stage
                          (FOCUS +10 · RELAX −1 · OVERLOAD −9 per report)
```

## What's in the box

The system is split into packages, each with a single responsibility and its
own README with usage details:

    src/hotb_2026/
    ├── brainaccess_connection/   # LSL stream from BrainAccess EEG hardware
    ├── record/                   # PsychoPy n-back recording sessions
    ├── preprocessing/            # per-window cleaning and standardization
    ├── features/                 # frozen REVE encoder + embedding cache
    ├── classification/           # training: dataset, Optuna search, SVM
    ├── reporting/                # metrics, figures, run reports
    ├── model_store/              # versioned model bundles, Hugging Face publish
    ├── inference/                # load bundle + live window classification
    ├── mappers/                  # adapters between package models
    └── tamagotchi/               # the cognitive-load-aware virtual pet

Each package README is the entry point for its part of the story:

- **`brainaccess_connection`** — connects to and streams from the BrainAccess
  EEG device (scan, stream, or `--simulate` a fake signal). Hardware notes,
  CLI usage and the LSL stream layout:
  [`src/hotb_2026/brainaccess_connection/README.md`](src/hotb_2026/brainaccess_connection/README.md)
- **`record`** — PsychoPy n-back procedure (rest / 1-back / 3-back) that
  records labeled EEG sessions:
  [`src/hotb_2026/record/README.md`](src/hotb_2026/record/README.md)
- **`preprocessing`** — cleans each streamer window: notch 50 Hz →
  band-pass 0.5–45 Hz → resample 250→200 Hz → z-score → clip ±15:
  [`src/hotb_2026/preprocessing/README.md`](src/hotb_2026/preprocessing/README.md)
- **`features`** — the frozen REVE-Base encoder
  (`brain-bzh/reve-base` + `brain-bzh/reve-positions`) and a
  content-addressed embedding cache, shared by training and inference.
- **`classification`** — builds labeled windows, tunes an RBF SVM over
  frozen REVE embeddings with Optuna and grouped cross-validation:
  [`src/hotb_2026/classification/README.md`](src/hotb_2026/classification/README.md)
- **`reporting`** — metric bundles, Matplotlib figures and a Markdown
  `report.md` per training run.
- **`model_store`** — self-describing bundles (`model.joblib`,
  `metadata.json`, `model_card.md`), versioning and Hugging Face
  publishing: [`src/hotb_2026/model_store/README.md`](src/hotb_2026/model_store/README.md)
- **`inference`** — loads a bundle and classifies one live window at a
  time; returns a `Prediction` (label, confidence, probabilities,
  state-change time) and plugs into `EEGStreamer`:
  [`src/hotb_2026/inference/README.md`](src/hotb_2026/inference/README.md)
- **`mappers`** — small adapters, e.g. `montage_to_list_electrode` for
  REVE's position encoding.
- **`tamagotchi`** — **the reason it's called Tam-EEG**: a terminal pet
  whose happiness is driven by the classified `UserState` — and by nothing
  else (no commit counters, no CPU metrics, no `psutil`). Happiness moves
  per classified report: FOCUS +10, RELAX −1, OVERLOAD −9. The pet ages
  through life stages when you keep it (and yourself) in good shape:
  [`src/hotb_2026/tamagotchi/README.md`](src/hotb_2026/tamagotchi/README.md)

New packages (UI layers, other connections, …) can be added alongside them.

## End-to-end walkthrough

[`notebooks/00_end_to_end_walkthrough.ipynb`](notebooks/00_end_to_end_walkthrough.ipynb)
follows one synthetic session through every package: stream contract,
preprocessing, windowing, dataset construction, REVE embeddings, Optuna
training, reporting and model packaging — and hands the final prediction to
the tamagotchi. Package-specific notebooks live under
[`notebooks/`](notebooks/), including the
[`notebooks/tamagotchi/tamagotchi_demo.ipynb`](notebooks/tamagotchi/tamagotchi_demo.ipynb)
demo of the state-driven pet.

## Quick start: EEG stream

```bash
uv sync
uv run hotb-brainaccess --scan          # list nearby headsets (~5 s)
uv run hotb-brainaccess                 # stream the configured headset
uv run hotb-brainaccess --simulate      # fake signal, no hardware
```

The stream is advertised over LSL as `BrainAccess EEG`, type `EEG`,
8 channels in µV. Device, layout and rate come from
[`config.py`](src/hotb_2026/brainaccess_connection/config.py)
(default device: `BA MINI 078`, 250 Hz).

## Quick start: train a classifier

```bash
# Synthetic recordings: windowing, REVE embeddings, Optuna, reports.
uv run python -m hotb_2026.classification.training --synthetic --trials 20

# Recorded n-back sessions, on-disk format (see the classification README).
uv run python -m hotb_2026.classification.training \
    --recordings data/sessions --out runs/nback --trials 60
```

Each run writes config, metrics, Optuna trials, predictions, figures and a
Markdown `report.md` to `runs/classification-<timestamp>`.

## Quick start: live inference

```bash
uv run python -m hotb_2026.inference --bundle bundles/nback --windows 3
```

Or in Python, wired into the streamer:

```python
from hotb_2026.inference import Predictor

predictor = Predictor.from_bundle("bundles/nback")
streamer.add_processor(predictor.processor(on_prediction=consume_prediction))
```

`Prediction` carries the label, confidence, full probabilities and the
state-change time. Details in the
[inference README](src/hotb_2026/inference/README.md).

## Quick start: the pet

```bash
uv run python -m hotb_2026.tamagotchi init --name Brainy --pet monkey
uv run python -m hotb_2026.tamagotchi daemon       # background loop
uv run python -m hotb_2026.tamagotchi install-hook   # pet above each prompt
uv run python -m hotb_2026.tamagotchi feed
```

From the EEG side, feed the classifier output straight into the pet:

```python
from hotb_2026.tamagotchi import apply_state

apply_state(pet, prediction.user_state)   # RELAX −1 · FOCUS +10 · OVERLOAD −9
```

State lives in `~/.hotb_tamagotchi/` (`pet.json`, `status.txt`,
`daemon.pid`); override with `HOTB_TAMAGOTCHI_HOME`.

Meet the team pet — **the monkey** (`--pet monkey`), drawn from the
[`pets/monkey.py`](src/hotb_2026/tamagotchi/pets/monkey.py) plugin:

```text
   EGG          BABY                     ADULT
  ( ovo )      @( ^o^ )@          @( ^o^ )@  _
   -----         _~_                /(   )\__/
              focused work       a happy grown monkey
```

Care for it and it hatches from the egg and grows through stages;
overload it and happiness drops 9 points per report until it wilts.

New pets are drop-in plugins — one file per pet under
[`src/hotb_2026/tamagotchi/pets/`](src/hotb_2026/tamagotchi/pets/):
ASCII art per stage and mood plus the decay and evolution parameters.
See the [tamagotchi README](src/hotb_2026/tamagotchi/README.md).

## Recorded data

Record n-back sessions for training:

```bash
uv run hotb-record --simulate --subject demo --sessions 1
```

Sessions land in `data/<subject_id>/session_<n>/` (`meta.json`,
`signal.npy`, `time.npy`, `state_changes.csv`). Existing session folders
abort the run rather than being overwritten. Details in the
[record README](src/hotb_2026/record/README.md).

## License

MIT — see [`LICENSE`](LICENSE).