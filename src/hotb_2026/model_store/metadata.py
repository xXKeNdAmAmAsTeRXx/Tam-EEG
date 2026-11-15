"""Build the self-describing metadata that travels with a model bundle.

A bundle must be reproducible without the training run directory. The metadata
therefore records the input contract (electrodes, window length, rates), the
frozen encoder, the exact preprocessing snapshot and the selected
hyperparameters alongside the model file itself.
"""
from __future__ import annotations

import importlib.metadata
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from .config import validate_version

ARTIFACT_SCHEMA_VERSION = 1

_PACKAGES = (
    "hotb-2026",
    "scikit-learn",
    "joblib",
    "numpy",
    "scipy",
    "huggingface-hub",
    "torch",
    "transformers",
)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def package_versions() -> dict[str, str]:
    """Versions of the packages that determine the serialized model."""
    versions: dict[str, str] = {}
    for name in _PACKAGES:
        version = _package_version(name)
        if version is not None:
            versions[name] = version
    return versions


def git_commit(directory: str | Path | None = None) -> str | None:
    """Current git commit, or ``None`` when git is unavailable."""
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(directory) if directory is not None else None,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return output.strip() or None


def jsonable(value: Any) -> Any:
    """Recursively convert numpy, path and dataclass values for JSON output."""
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def validate_result(result: Any) -> Any:
    """Check that ``result`` holds a fitted, self-consistent pipeline.

    Returns the fitted estimator. Raises ``TypeError``/``ValueError`` with a
    message that points at the offending field.
    """
    if not isinstance(result, dict):
        raise TypeError(
            "result must be the dict returned by "
            f"hotb_2026.classification.training.train, got "
            f"{type(result).__name__}"
        )

    model = result.get("model")
    if model is None:
        raise ValueError("result has no 'model'; pass the dict returned by train()")
    try:
        check_is_fitted(model)
    except NotFittedError as exc:
        raise ValueError("result['model'] is not fitted") from exc

    for method in ("predict", "predict_proba"):
        if not callable(getattr(model, method, None)):
            raise ValueError(f"result['model'] does not implement {method}()")

    label_names = result.get("label_names")
    if not label_names:
        raise ValueError("result['label_names'] is required")

    classes = np.asarray(getattr(model, "classes_", []), dtype=int)
    if classes.size != len(label_names):
        raise ValueError(
            f"model was fitted on {classes.size} classes but label_names has "
            f"{len(label_names)} entries"
        )
    expected = np.arange(len(label_names), dtype=int)
    if not np.array_equal(np.sort(classes), expected):
        raise ValueError(
            "model classes must be the integers "
            f"0..{len(label_names) - 1}, got {classes.tolist()}"
        )

    embedding_meta = result.get("embedding_meta") or {}
    embedding_dim = embedding_meta.get("embedding_dim")
    n_features = getattr(model, "n_features_in_", None)
    if embedding_dim is not None and n_features is not None:
        if int(embedding_dim) != int(n_features):
            raise ValueError(
                f"embedding_meta.embedding_dim={embedding_dim} does not match "
                f"the fitted model ({int(n_features)} features)"
            )

    return model


def build_metadata(
    result: dict[str, Any],
    *,
    version: str,
    directory: str | Path | None = None,
) -> dict[str, Any]:
    """Assemble the JSON metadata for one bundle.

    Parameters
    ----------
    result:
        Dict returned by ``hotb_2026.classification.training.train``.
    version:
        Model version, normally a semver string such as ``"0.1.0"``.
    directory:
        Directory used to look up the current git commit.
    """
    model = validate_result(result)
    version = validate_version(str(version))

    embedding_meta = result.get("embedding_meta") or {}
    dataset = result.get("dataset_summary") or {}
    label_names = [str(name) for name in result.get("label_names", [])]
    raw_labels = result.get("labels")
    labels = (
        list(raw_labels)
        if raw_labels is not None
        else list(range(len(label_names)))
    )
    classes = np.asarray(getattr(model, "classes_", labels), dtype=int)

    preprocessing = (
        dataset.get("preprocessing")
        or embedding_meta.get("preprocessing")
        or {}
    )
    holdout = (result.get("holdout") or {}).get("metrics") or {}
    split = result.get("split") or {}
    cv = result.get("cv") or {}
    steps = [
        {"name": name, "type": type(step).__name__}
        for name, step in getattr(model, "steps", [])
    ]

    metadata: dict[str, Any] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_version": str(version),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(directory),
        "versions": package_versions(),
        "labels": {
            "names": label_names,
            "classes": classes.tolist(),
            "n_classes": len(label_names),
        },
        "input": {
            "electrodes": list(
                embedding_meta.get("electrodes")
                or dataset.get("electrodes")
                or []
            ),
            "window_s": _first(
                dataset.get("window_s"), embedding_meta.get("window_s")
            ),
            "n_samples": dataset.get("n_samples"),
            "sfreq": _first(dataset.get("sfreq"), embedding_meta.get("sfreq")),
            "raw_sfreq": _first(
                dataset.get("raw_sfreq"), embedding_meta.get("raw_sfreq")
            ),
            "time_units": "s",
            "label_rule": "state active at window start",
        },
        "encoder": {
            "model_name": embedding_meta.get("model_name"),
            "pos_model_name": embedding_meta.get("pos_model_name"),
            "embedding_dim": _first(
                embedding_meta.get("embedding_dim"),
                getattr(model, "n_features_in_", None),
            ),
            "patch_size": embedding_meta.get("patch_size"),
            "revision": embedding_meta.get("revision"),
        },
        "preprocessing": dict(preprocessing),
        "training": {
            "config": result.get("config") or {},
            "best_params": result.get("best_params") or {},
            "best_value": result.get("best_value"),
            "cv": {
                key: cv.get(key)
                for key in ("metric", "n_splits", "mean", "std", "scores")
            },
            "holdout": holdout,
            "split": {
                key: split.get(key)
                for key in (
                    "n_train",
                    "n_test",
                    "n_train_groups",
                    "test_groups",
                )
            },
            "dataset": {
                key: dataset.get(key)
                for key in (
                    "n_windows",
                    "n_channels",
                    "n_samples",
                    "class_counts",
                    "n_groups",
                )
            },
        },
        "model": {
            "estimator": type(model).__name__,
            "steps": steps,
            "classes": classes.tolist(),
            "n_features": int(getattr(model, "n_features_in_", 0)),
        },
    }
    return jsonable(metadata)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        text = f"{value:.{digits}f}"
        return text.rstrip("0").rstrip(".") if "." in text else text
    return str(value)


def render_model_card(metadata: dict[str, Any]) -> str:
    """Render the Hugging Face model card for a bundle."""
    labels = metadata.get("labels", {})
    label_names = [str(name) for name in labels.get("names", [])]
    encoder = metadata.get("encoder", {})
    contract = metadata.get("input", {})
    preprocessing = metadata.get("preprocessing", {})
    training = metadata.get("training", {})
    holdout = training.get("holdout") or {}
    version = metadata.get("model_version", "unknown")

    lines: list[str] = [
        "---",
        "library_name: scikit-learn",
        "tags:",
        "  - eeg",
        "  - neuroscience",
        "  - cognitive-load",
        "  - reve",
        "  - scikit-learn",
        "---",
        "",
        "# Cognitive-overload classifier (REVE + SVM)",
        "",
        f"{len(label_names)}-class cognitive-overload model for N-back EEG. "
        "A frozen REVE-Base transformer embeds each window; a calibrated RBF "
        "SVM classifies the embedding.",
        "",
        "| | |",
        "|---|---|",
        f"| Version | `{version}` |",
        f"| Classes | {', '.join(label_names)} |",
        f"| Encoder | `{encoder.get('model_name')}` |",
        f"| Position encoder | `{encoder.get('pos_model_name')}` |",
        f"| Embedding dim | {_fmt(encoder.get('embedding_dim'))} |",
        f"| Patch size | {_fmt(encoder.get('patch_size'))} |",
        f"| Window | {_fmt(contract.get('window_s'))} s at "
        f"{_fmt(contract.get('sfreq'))} Hz |",
        f"| Electrodes | {', '.join(contract.get('electrodes', [])) or '—'} |",
        "",
    ]

    if holdout:
        lines += [
            "## Held-out performance",
            "",
            "| metric | value |",
            "|---|---|",
            f"| Accuracy | {_fmt(holdout.get('accuracy'))} |",
            f"| Balanced accuracy | {_fmt(holdout.get('balanced_accuracy'))} |",
            f"| Macro F1 | {_fmt(holdout.get('f1_macro'))} |",
            f"| ROC AUC (OVR) | {_fmt(holdout.get('roc_auc_ovr'))} |",
            "",
        ]

    lines += [
        "## Input contract",
        "",
        f"- Electrodes, in channel order: "
        f"{', '.join(contract.get('electrodes', [])) or '—'}",
        f"- Window length: {_fmt(contract.get('window_s'))} s "
        f"({_fmt(contract.get('n_samples'))} samples at "
        f"{_fmt(contract.get('sfreq'))} Hz)",
        f"- Raw sample rate: {_fmt(contract.get('raw_sfreq'))} Hz",
        f"- Time units: {contract.get('time_units', 's')}",
        f"- Label rule: {contract.get('label_rule', '—')}",
        "",
    ]

    if preprocessing:
        lines += [
            "## Preprocessing",
            "",
            "| parameter | value |",
            "|---|---|",
        ]
        for key, value in preprocessing.items():
            lines.append(f"| `{key}` | {_fmt(value)} |")
        lines.append("")

    lines += [
        "## Usage",
        "",
        "```python",
        "import joblib",
        "",
        "pipeline = joblib.load(\"model.joblib\")",
        "probabilities = pipeline.predict_proba(embedding)",
        "label = pipeline.predict(embedding)[0]",
        "```",
        "",
        "`embedding` must come from the same frozen encoder and window "
        "preprocessing recorded in `metadata.json`.",
        "",
        "## Files",
        "",
        "- `model.joblib` — fitted `Pipeline(StandardScaler, "
        "CalibratedClassifierCV(SVC))`",
        "- `metadata.json` — input contract, encoder, preprocessing and metrics",
        "- `model_card.md` — this file",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "build_metadata",
    "git_commit",
    "jsonable",
    "package_versions",
    "render_model_card",
    "validate_result",
]
