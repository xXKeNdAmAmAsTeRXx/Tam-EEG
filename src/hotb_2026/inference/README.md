# inference

Load a bundle written by [`model_store`](../model_store/README.md), rebuild the
frozen REVE encoder and the recorded preprocessing, and classify one live EEG
window at a time.

The output is a [`Prediction`](schema.py) dataclass. Training and offline
classification stay dict/array-based; only live inference returns a dataclass,
so downstream code (agent, UI, logging) has a stable object to consume.

## Quick start

```python
from hotb_2026.inference import Predictor

predictor = Predictor.from_bundle("bundles/nback")

prediction = predictor.predict_raw(raw_window, current_time=42.0)
print(prediction.label_name, prediction.confidence)
print(prediction.state_change_time, prediction.window_start, prediction.window_end)
```

`raw_window` is `(n_channels, n_samples)` in the channel order of the live
signal. The predictor selects and reorders channels to the training order,
keeps the trailing `window_s * raw_sfreq` samples, runs the recorded
preprocessing once, encodes the window with REVE and calls the fitted
classifier.

A pinned Hugging Face bundle works the same way:

```python
predictor = Predictor.from_pretrained(
    "ai-med/hotb-2026-nback-classifier",
    revision="0.1.0",  # tag, branch or commit
)
```

Only `model.joblib`, `metadata.json` and `model_card.md` are downloaded. REVE
weights are pulled separately by `transformers` and cached locally.

## `Prediction`

| field | meaning |
|---|---|
| `label` | predicted class id, `0..n_classes - 1` |
| `label_name` | class name from the bundle |
| `confidence` | probability of the predicted class |
| `probabilities` | class name -> probability, in bundle class order |
| `current_time` | stream time at the end of the window, seconds |
| `state_change_time` | when the **predicted** label was first reported |
| `window_start` | `current_time - window_s` |
| `window_end` | `current_time` |
| `model_version` | `model_version` from `metadata.json` |

Live inference has no ground-truth label changes, so `state_change_time` tracks
the current *predicted* state: it is set on the first prediction, stays fixed
while the label is stable, and moves to the current time when the label
changes. If stream time moves backwards (a restarted stream), the predictor
resets that tracking automatically. `predictor.reset()` clears it explicitly.

The callback time from `EEGStreamer` is the end of the chunk that crossed the
window boundary, so it can lag the exact boundary by up to one chunk.

## Live streaming

`Predictor.processor()` returns the `WindowedPreprocessor` that plugs into
`EEGStreamer`:

```python
from hotb_2026.brainaccess_connection import MONTAGE, EEGStreamer, SimulatedDriver

streamer = EEGStreamer(
    SimulatedDriver(MONTAGE),
    MONTAGE,
    window_s=predictor.window_s + 1.0,  # extra room for exact slicing
)
streamer.add_processor(
    predictor.processor(on_prediction=lambda p: print(p.label_name, p.confidence))
)

predictor.warmup()  # load REVE before the first window arrives
streamer.start()
try:
    ...
finally:
    streamer.stop()
```

Give the streamer a window **longer** than the model window. The adapter then
slices exact, non-overlapping segments out of the ring buffer. With equal
lengths it falls back to the latest full window and the prediction is shifted
by up to one chunk.

The processor runs on the streamer thread and is called once per completed
window. Keep `on_prediction` fast, or hand work off to a queue.

## Input contract

The bundle records everything inference needs:

- ordered electrode names (training order),
- `window_s`, `n_samples`, `sfreq` (after preprocessing), `raw_sfreq`,
- `time_units="s"`,
- the encoder ids and revision,
- the exact `PreprocessingConfig`.

`Predictor.from_bundle(..., electrodes=...)` (or
`predictor.set_source_electrodes(...)`) accepts a montage or a name list when
the live channel order differs. Extra channels are ignored; a missing training
electrode raises `ValueError`.

`predict(processed_window, current_time)` is the low-level path: it expects an
already-preprocessed window in training electrode order and exactly
`n_samples`. `predict_raw` is the safe path for the live stream.

## Validation

Loading fails fast instead of producing quiet, wrong predictions:

- unsupported `artifact_schema_version`, missing `model_version`,
- `model.joblib` missing or not a fitted classifier with `predict` /
  `predict_proba`,
- model classes that do not match `labels.classes` (must be `0..n-1`),
- feature count that does not match `encoder.embedding_dim`,
- missing or invalid electrodes, window length, sample counts or rates,
- a preprocessing snapshot whose input rate, output rate or output sample
  count disagrees with the metadata.

The last check runs the recorded preprocessing on a zero window during
loading, which catches a stale snapshot whose filters or resampling would
change the window shape.

`joblib.load` can execute arbitrary code. Only load bundles you trust.

## CLI

```bash
# Local bundle, fake headset, three predictions.
uv run python -m hotb_2026.inference --bundle bundles/nback --windows 3

# Pinned Hub bundle.
uv run python -m hotb_2026.inference \
    --repo-id ai-med/hotb-2026-nback-classifier --revision 0.1.0
```

Each completed window is printed as one JSON line. `--device`, `--chunk-ms`,
`--seed`, `--timeout` and `-v` control the encoder device and simulation.

## Demo

[`notebooks/inference/01_live_inference.ipynb`](../../../notebooks/inference/01_live_inference.ipynb)
trains a tiny bundle, loads it with `Predictor`, classifies offline windows and
runs a simulated stream.
