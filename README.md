# HOTB_2026 Project

Heroes of The Brain Neurohackathon 2026! Project is not finished yet. 
Provides a BrainAccess device connection and 8-channel EEG ML training framework. 

## Architecture

The system is split into several packages, each with a single responsibility.
More packages (UI layers, additional connections, etc.) may be added later.

    src/hotb_2026/
    ├── brainaccess_connection/   # connection to BrainAccess EEG hardware
    ├── preprocessing/            # per-window cleaning and standardization
    ├── features/                 # frozen REVE encoder + embedding cache
    ├── classification/           # training: dataset, Optuna search, SVM
    ├── reporting/                # metrics, figures, run reports
    ├── model_store/              # save/deploy the trained model
    ├── inference/                # load model + live prediction
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
- **`features`** — the frozen REVE-Base encoder (`brain-bzh/reve-base` plus
  `brain-bzh/reve-positions`) and a content-addressed embedding cache. Shared by
  training and live inference. Encoder revisions can be pinned for
  reproducible bundles.
- **`classification`** — training only: builds labeled windows from recordings,
  tunes an RBF SVM over frozen REVE embeddings with Optuna and grouped
  cross-validation, and evaluates on held-out groups. See
  [`src/hotb_2026/classification/README.md`](src/hotb_2026/classification/README.md)
  and the notebook
  [`notebooks/classification/01_training.ipynb`](notebooks/classification/01_training.ipynb).
- **`reporting`** — metric bundles, Matplotlib figures, `metrics.json` and the
  Markdown `report.md` for a training run.
- **`model_store`** — writes a self-describing bundle (`model.joblib`,
  `metadata.json`, `model_card.md`), versions it, and publishes it to a Hugging
  Face model repo. See
  [`src/hotb_2026/model_store/README.md`](src/hotb_2026/model_store/README.md)
  and the notebook
  [`notebooks/classification/02_save_and_publish.ipynb`](notebooks/classification/02_save_and_publish.ipynb).
- **`inference`** — loads a bundle (local or pinned Hugging Face revision),
  rebuilds the recorded preprocessing and frozen REVE encoder, and classifies
  one live window at a time. Returns a `Prediction` dataclass (label,
  confidence, probabilities, current time, predicted state-change time, model
  version) and plugs into `EEGStreamer`. See
  [`src/hotb_2026/inference/README.md`](src/hotb_2026/inference/README.md)
  and the notebook
  [`notebooks/inference/01_live_inference.ipynb`](notebooks/inference/01_live_inference.ipynb).
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

## End-to-end walkthrough

[`notebooks/00_end_to_end_walkthrough.ipynb`](notebooks/00_end_to_end_walkthrough.ipynb)
follows one synthetic session through every implemented package: stream
contract, preprocessing, windowing, dataset construction, REVE embeddings,
Optuna training, reporting and model packaging. It links to the package-specific
notebooks for deeper dives, including
[`notebooks/inference/01_live_inference.ipynb`](notebooks/inference/01_live_inference.ipynb)
for loading the bundle and classifying live windows.

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

## Quick start: classification training

```bash
# Synthetic recordings: exercises windowing, REVE embeddings, Optuna and reports.
uv run python -m hotb_2026.classification.training --synthetic --trials 20

# Recorded sessions in the on-disk format (see the classification README).
uv run python -m hotb_2026.classification.training \
    --recordings data/sessions --out runs/nback --trials 60
```

Each run writes `config.json`, metrics, Optuna trials, predictions, figures and
a Markdown `report.md` to `runs/classification-<timestamp>`. See the
[classification README](src/hotb_2026/classification/README.md) for the data
format, labeling rules, grouped validation and options.

## Quick start: save and publish a model

Saving is a Python call because the fitted pipeline lives in the object returned
by `train`:

```python
from hotb_2026.model_store import StoreConfig, save_model

bundle = save_model(
    result,
    "bundles/nback",
    config=StoreConfig(version="0.1.0"),
    run_dir="runs/nback",
)
```

Inspect and publish it with the CLI:

```bash
uv run python -m hotb_2026.model_store show bundles/nback

HF_TOKEN=hf_... uv run python -m hotb_2026.model_store publish bundles/nback \
    --repo-id ai-med/hotb-2026-nback-classifier
```

The bundle contains `model.joblib`, a self-describing `metadata.json` and a
Hugging Face `model_card.md`. REVE weights are referenced by id, not uploaded.
See the [model_store README](src/hotb_2026/model_store/README.md) for the
metadata schema, versioning and the owner→user publishing fallback.

## Quick start: live inference

```python
from hotb_2026.inference import Predictor

predictor = Predictor.from_bundle("bundles/nback")
prediction = predictor.predict_raw(raw_window, current_time=42.0)

print(prediction.label_name, prediction.confidence)
print(prediction.state_change_time)
```

A pinned Hub bundle works the same way:

```python
predictor = Predictor.from_pretrained(
    "ai-med/hotb-2026-nback-classifier",
    revision="0.1.0",
)
```

Wire it into the streamer with one adapter:

```python
streamer = EEGStreamer(driver, MONTAGE, window_s=predictor.window_s + 1.0)
streamer.add_processor(predictor.processor(on_prediction=consume_prediction))
predictor.warmup()
```

Or run the simulated CLI:

```bash
uv run python -m hotb_2026.inference --bundle bundles/nback --windows 3
```

The inference package validates the metadata against the serialized model at
load time and returns a `Prediction` dataclass with probabilities, timing and
the model version. See the
[inference README](src/hotb_2026/inference/README.md) for the input contract,
state-change semantics and electrode alignment.

## Quick start: recorded n-back sessions

```bash
uv run hotb-record --simulate --subject demo --sessions 1
```

Each session is written to `data/<subject_id>/session_<n>/` with `meta.json`,
`signal.npy`, `time.npy`, and `state_changes.csv`. Pass `--session <n>` to pick
the session index; an existing session folder aborts the run instead of being
overwritten. Instruction/feedback periods are cut out of the recorded signal. See
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
