# model_store

Turn a fitted training result into a deployable, self-describing bundle and
publish it to the [Hugging Face Hub](https://huggingface.co/models).

Training is deliberately separate: `hotb_2026.classification` fits the model
and returns it as plain Python. This package decides how the model is stored,
versioned and shipped. Live loading and prediction live in `hotb_2026.inference`.

## Bundle layout

```
<bundle>/
├── model.joblib     # joblib (compress=3) of the fitted Pipeline
├── metadata.json    # self-describing contract (schema v1)
└── model_card.md    # Hugging Face model card
```

`metrics.json`, `report.md` and `predictions.npz` are copied in when a training
`run_dir` is supplied, as are any `extra_files`.

The model itself is `Pipeline(StandardScaler, CalibratedClassifierCV(SVC))`.
REVE weights are **not** stored: `metadata.json` records the encoder ids so
inference can download the same frozen model. This keeps bundles small and
avoids duplicating a multi-hundred-megabyte checkpoint.

## Saving

Saving needs the in-memory dict returned by
`hotb_2026.classification.training.train`, so it is a Python call:

```python
from hotb_2026.classification.training import TrainingConfig, train
from hotb_2026.features import ReveEncoder
from hotb_2026.model_store import StoreConfig, save_model

result = train(dataset, config=TrainingConfig(), encoder=ReveEncoder(dataset.electrodes))
bundle = save_model(
    result,
    "bundles/nback",
    config=StoreConfig(version="0.1.0"),
    run_dir="runs/nback",  # optional: copies metrics.json and report.md
)

print(bundle.version, bundle.metadata_path)
```

`save_model` validates before writing anything:

- `result` is the dict returned by `train`;
- `result["model"]` is fitted and implements `predict` / `predict_proba`;
- `classes_` are exactly `0..n_classes-1` and match `label_names`;
- the model's feature count matches `embedding_meta["embedding_dim"]`.

An invalid result leaves the output directory untouched.

## `metadata.json` (schema v1)

| section | contents |
|---|---|
| identity | `artifact_schema_version`, `model_version`, `created_at`, `git_commit`, package versions |
| `labels` | class names, class ids, class count |
| `input` | ordered electrodes, window length, samples per window, output/raw sample rates, time units, label rule |
| `encoder` | REVE model id, position-model id, embedding dimension, patch size, revision |
| `preprocessing` | exact `PreprocessingConfig` snapshot used to build the windows |
| `training` | config, best parameters, grouped CV, held-out metrics, dataset summary |
| `model` | estimator type, pipeline step names and classes |

`metadata.json` is the source of truth for inference. It never contains absolute
paths, tokens or datasets.

## Publishing

```bash
uv run python -m hotb_2026.model_store show bundles/nback

HF_TOKEN=hf_... uv run python -m hotb_2026.model_store publish bundles/nback \
    --repo-id ai-med/hotb-2026-nback-classifier
```

Or from Python:

```python
from hotb_2026.model_store import StoreConfig, publish

summary = publish(
    "bundles/nback",
    StoreConfig(repo_id="ai-med/hotb-2026-nback-classifier"),
)
print(summary["repo_url"], summary["tag"])
```

Behaviour:

- The repository is **public** by default (`private=False`).
- `create_repo(..., exist_ok=True)` makes repeated uploads safe.
- The version in `metadata.json` is used as the git tag and in the commit
  message. It is authoritative; `StoreConfig.version` only matters when saving.
- If the configured owner is not writable, `publish` retries under the
  authenticated user's namespace and logs a warning. Pass
  `fallback_to_user=False` (or `--no-fallback`) to fail instead.
- `HF_TOKEN` is used when set; otherwise `huggingface_hub` reads the token
  cached by `hf auth login`.

The repository id can also come from `HOTB_MODEL_REPO_ID`:

```bash
export HOTB_MODEL_REPO_ID=ai-med/hotb-2026-nback-classifier
uv run python -m hotb_2026.model_store publish bundles/nback
```

## Using a bundle

```python
import joblib

pipeline = joblib.load("model.joblib")
probabilities = pipeline.predict_proba(embedding)  # (n, n_classes)
label = pipeline.predict(embedding)
```

`embedding` must come from the encoder and preprocessing recorded in
`metadata.json`. Rebuilding that pipeline is the job of `hotb_2026.inference`.

## Demo

See [`notebooks/classification/02_save_and_publish.ipynb`](../../../notebooks/classification/02_save_and_publish.ipynb).
The publish cell is skipped when `HF_TOKEN` or `HOTB_MODEL_REPO_ID` is unset, so
the notebook runs offline.
