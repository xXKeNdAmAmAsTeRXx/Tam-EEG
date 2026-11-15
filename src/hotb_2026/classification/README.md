# Classification training

This package turns continuous EEG recordings into a 3-class cognitive-overload
classifier (0 = low, 1 = medium, 2 = high load). Training only: model saving,
reports and live inference are separate top-level packages.

The model is a frozen REVE-Base transformer followed by an RBF SVM. REVE turns
each preprocessed window into a 512-d embedding; the SVM is trained on those
embeddings.

## Data format

A recording directory contains:

    session/
    ├── meta.json            # session_id, subject_id, electrodes, sfreq
    ├── signal.npy           # (n_samples, n_channels), µV, electrode order
    ├── time.npy             # (n_samples,), current stream time in seconds
    └── state_changes.csv    # time,label — when the label becomes active

`time.npy` is optional: if it is missing, times are reconstructed from
`meta.json` (`start_time_s`, default 0) and `sfreq`. `state_changes.csv` must be
sorted by time; the first change marks when the stream becomes labeled.

`load_recordings("data/sessions")` reads every subdirectory that contains a
`meta.json`.

## Windowing and labels

- A window is cut on the raw time axis first, then passed through
  `Preprocessor` on its own (notch → band-pass → resample → per-window
  z-score → clip). This matches live inference, which also sees one window at
  a time.
- The label is the state active at the window **start**.
- With `drop_boundary=True` (default), a window is dropped when the label at
  its end differs from the label at its start. Boundary windows otherwise mix
  two states and teach the model ambiguous examples.
- `group_by="auto"` uses whole sessions when there are at least five; otherwise
  it uses state blocks (`session_id:b<index>`). Cross-validation never splits
  adjacent windows from the same group.

Training and inference share the same input contract: ordered electrode names,
per-sample current time, and timestamps where the label changes.

`build_dataset` snapshots the exact `PreprocessingConfig` in
`dataset.preprocessing`, and `EegDataset.save` persists it inside the `.npz`.
The snapshot also travels through `result["embedding_meta"]`, so a model bundle
can record the preprocessing that produced its embeddings.

## Train

```bash
# Offline smoke test: synthetic sessions, no hardware or recordings needed.
uv run python -m hotb_2026.classification.training --synthetic --trials 20

# Recorded sessions.
uv run python -m hotb_2026.classification.training \
    --recordings data/sessions --out runs/nback --trials 60

# Reuse a pre-built windowed dataset.
uv run python -m hotb_2026.classification.training \
    --data runs/dataset.npz --out runs/nback
```

Useful options: `--folds`, `--test-size`, `--metric`, `--group-by`, `--window`,
`--stride`, `--device`, `--no-cache`, `--save-dataset`, `--no-plots`.
The default run directory is `runs/classification-<timestamp>`.

### Recorded sessions

`state_changes.csv` holds `time,label` rows, one per state change, sorted by
time. The label is the state that becomes active at `time`.

Two label formats are accepted:

* integer overload ids — `0` (low), `1` (medium), `2` (high);
* task state names written by the N-back procedure — `rest`, `1back`, `2back`,
  `3back`. Task names are folded onto the overload tiers: `rest` maps to the
  lowest tier and `<n>back` maps to `min(2, n)`, so `1back` → medium,
  `2back`/`3back` → high and `rest` → low.

Before the first state change the stream is unlabeled and its windows are
dropped by the default windowing.

Windows that would reach past the end of a recording are dropped too, so every
window in a dataset has the fixed length planned from `window_s`.

### From run artifacts to a bundle

Training writes the run directory (metrics, trials, figures, report). To turn a
fitted result into a deployable bundle:

```python
from hotb_2026.classification.dataset import build_dataset, load_recordings
from hotb_2026.classification.training import TrainingConfig, train
from hotb_2026.model_store import StoreConfig, save_model

dataset = build_dataset(load_recordings("data/sessions"), window_s=10.0)
result = train(
    dataset,
    config=TrainingConfig(n_splits=5, n_trials=60, seed=0),
    cache_dir="runs/nback/embeddings",
)
bundle = save_model(result, "bundles/nback", config=StoreConfig(version="0.1.0"))
```

### Live inference on the recorded bundle

```bash
uv run python -m hotb_2026.inference --bundle bundles/nback --windows 3
```

The same bundle can be loaded from Python and scored on a recorded window:

```python
import numpy as np

from hotb_2026.classification.dataset import build_dataset, load_recordings
from hotb_2026.inference import Predictor

dataset = build_dataset(load_recordings("data/sessions"), window_s=10.0)
predictor = Predictor.from_bundle("bundles/nback", electrodes=dataset.electrodes)
recording = load_recordings("data/sessions")[0]
raw_window = recording.signal[:, : predictor.raw_samples]
prediction = predictor.predict_raw(raw_window, current_time=predictor.window_s)
print(prediction.label_name, prediction.confidence)
```

The run directory contains:

| file | contents |
|---|---|
| `config.json` | training configuration |
| `dataset_summary.json` | windows, classes, groups |
| `embeddings.meta.json` | encoder, shape, cache location |
| `trials.csv` | every Optuna trial |
| `best_params.json` | selected hyperparameters |
| `metrics.json` | CV, out-of-fold and held-out metrics |
| `predictions.npz` | held-out and out-of-fold predictions |
| `report.md` | human-readable report |
| `*.png` | confusion matrix, trials, embedding PCA, confidence |

Embeddings are cached under `<run>/embeddings` and reused on later runs over
the same windows. Since REVE is frozen, this is safe and makes the Optuna
search cheap: a trial is only a few SVM fits.

## Validation

- **Outer holdout** — whole groups are held out; this is the honest metric.
- **Inner grouped CV** — `StratifiedGroupKFold` used for model selection; its
  score is optimistic because the hyperparameters were chosen on it.
- **Out-of-fold predictions** — diagnostics on the training groups.

With fewer groups than requested folds, the number of folds is reduced
automatically (down to two). Very small datasets may not support a grouped
holdout at all; add recordings or use state-block grouping.

## Python API

```python
from hotb_2026.classification.dataset import build_dataset, load_recordings
from hotb_2026.classification.training import TrainingConfig, train
from hotb_2026.reporting import write_run

dataset = build_dataset(load_recordings("data/sessions"), window_s=10.0)
result = train(dataset, config=TrainingConfig(n_trials=60))
write_run("runs/nback", result)
```

`train` returns a plain dict. `result["model"]` is the pipeline trained on all
windows; `result["holdout_model"]` produced the held-out score. Saving these
models belongs to the `model_store` step, not to `classification`.

## From training to live inference

The bundle written by `model_store` records the ordered electrodes, window
length, sample rates, encoder ids/revision and preprocessing snapshot produced
here. [`hotb_2026.inference`](../inference/README.md) rebuilds exactly that
pipeline, so training and live predictions see the same windows. It returns a
`Prediction` dataclass and provides a `WindowedPreprocessor` adapter for
`EEGStreamer`.
