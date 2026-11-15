"""Write a training run to a directory: JSON, CSV, plots and a Markdown report."""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import plots

log = logging.getLogger(__name__)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(_jsonable(payload), indent=2))
    return path


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _confusion_table(matrix: Any, label_names: list[str]) -> list[str]:
    matrix = np.asarray(matrix)
    header = "| true \\ pred | " + " | ".join(label_names) + " |"
    divider = "|---" * (len(label_names) + 1) + "|"
    lines = [header, divider]
    for index, name in enumerate(label_names):
        row = [str(int(value)) for value in matrix[index]]
        lines.append(f"| **{name}** | " + " | ".join(row) + " |")
    return lines


def render_report(result: dict[str, Any]) -> str:
    """Render the human-readable Markdown report."""
    config = result.get("config", {})
    summary = result.get("dataset_summary") or {}
    embedding_meta = result.get("embedding_meta") or {}
    split = result.get("split", {})
    cv = result.get("cv", {})
    holdout = result.get("holdout", {})
    oof = result.get("oof", {})
    metrics = holdout.get("metrics", {})
    label_names = [str(name) for name in result.get("label_names", [])]
    best = result.get("best_params", {})

    lines: list[str] = [
        "# REVE + SVM training report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Run",
        "",
        f"- Metric: `{config.get('metric', cv.get('metric', 'balanced_accuracy'))}`",
        f"- Window: {_fmt(summary.get('window_s', config.get('window_s')))} s "
        f"at {_fmt(summary.get('sfreq'))} Hz",
        f"- Trials: {len(result.get('trials', []))} (best CV score "
        f"{_fmt(result.get('best_value'))})",
        f"- Groups: {split.get('n_train_groups', '—')} train / "
        f"{len(split.get('test_groups', []))} held out",
        "",
    ]

    if summary:
        counts = summary.get("class_counts", {})
        lines += [
            "## Dataset",
            "",
            f"- Windows: {summary.get('n_windows', '—')}",
            f"- Channels: {summary.get('n_channels', '—')}",
            f"- Samples per window: {summary.get('n_samples', '—')}",
            f"- Raw rate: {_fmt(summary.get('raw_sfreq'))} Hz",
            "- Class counts: "
            + ", ".join(f"{name}: {count}" for name, count in counts.items()),
            "",
        ]

    if embedding_meta:
        lines += [
            "## Embeddings",
            "",
            f"- Encoder: `{embedding_meta.get('model_name', 'unknown')}`",
            f"- Shape: {embedding_meta.get('shape', '—')}",
        ]
        if embedding_meta.get("cache_dir"):
            lines.append(f"- Cache: `{embedding_meta['cache_dir']}`")
        lines.append("")

    lines += [
        "## Hyperparameters",
        "",
        "| parameter | value |",
        "|---|---|",
        f"| `C` | {_fmt(best.get('C'), 5)} |",
        f"| `gamma` | {_fmt(best.get('gamma'), 5)} |",
        f"| `class_weight` | {_fmt(best.get('class_weight'))} |",
        "",
        "Cross-validation was used to select these values, so its score is "
        "optimistic. The held-out score below is the honest estimate.",
        "",
        "## Held-out test groups",
        "",
        f"- Windows: {holdout.get('metrics', {}).get('n_samples', '—')}",
        f"- Groups: {', '.join(map(str, split.get('test_groups', []))) or '—'}",
        f"- Accuracy: {_fmt(metrics.get('accuracy'))}",
        f"- Balanced accuracy: {_fmt(metrics.get('balanced_accuracy'))}",
        f"- Macro F1: {_fmt(metrics.get('f1_macro'))}",
    ]
    if metrics.get("roc_auc_ovr") is not None:
        lines.append(f"- ROC AUC (OVR): {_fmt(metrics.get('roc_auc_ovr'))}")
    if metrics.get("mean_confidence") is not None:
        lines.append(
            f"- Mean confidence: {_fmt(metrics.get('mean_confidence'))} "
            f"(correct {_fmt(metrics.get('mean_confidence_correct'))}, "
            f"wrong {_fmt(metrics.get('mean_confidence_wrong'))})"
        )
    lines += [""]

    if metrics.get("confusion_matrix"):
        lines += ["### Confusion matrix", ""]
        lines += _confusion_table(
            metrics["confusion_matrix"], label_names
        )
        lines += [""]

    per_class = metrics.get("per_class", {})
    if per_class:
        lines += [
            "### Per class",
            "",
            "| class | precision | recall | f1 | support |",
            "|---|---|---|---|---|",
        ]
        for name in label_names:
            row = per_class.get(name, {})
            lines.append(
                f"| {name} | {_fmt(row.get('precision'))} | "
                f"{_fmt(row.get('recall'))} | {_fmt(row.get('f1'))} | "
                f"{row.get('support', '—')} |"
            )
        lines += [""]

    cv_scores = cv.get("scores", [])
    lines += [
        "## Cross-validation (training groups)",
        "",
        f"- Folds: {cv.get('n_splits', '—')}",
        f"- Scores: {', '.join(_fmt(score) for score in cv_scores) or '—'}",
        f"- Mean: {_fmt(cv.get('mean'))} ± {_fmt(cv.get('std'))}",
        "",
        "## Out-of-fold predictions",
        "",
        f"- Accuracy: {_fmt(oof.get('metrics', {}).get('accuracy'))}",
        f"- Balanced accuracy: "
        f"{_fmt(oof.get('metrics', {}).get('balanced_accuracy'))}",
        f"- Macro F1: {_fmt(oof.get('metrics', {}).get('f1_macro'))}",
        "",
        "## Files",
        "",
        "- `config.json` — run configuration",
        "- `dataset_summary.json` — windows, classes and groups",
        "- `embeddings.meta.json` — encoder and embedding shape",
        "- `trials.csv` — every Optuna trial",
        "- `best_params.json` — selected hyperparameters",
        "- `metrics.json` — CV, out-of-fold and held-out metrics",
        "- `predictions.npz` — held-out and out-of-fold predictions",
        "- `confusion_matrix.png`, `trials.png`, `embedding_pca.png` — figures",
        "",
    ]
    return "\n".join(lines)


def _write_trials(run_dir: Path, records: list[dict[str, Any]]) -> Path:
    path = run_dir / "trials.csv"
    fields = [
        "number",
        "state",
        "value",
        "C",
        "gamma",
        "class_weight",
        "fold_std",
        "fold_scores",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            params = record.get("params", {})
            writer.writerow(
                {
                    "number": record.get("number"),
                    "state": record.get("state"),
                    "value": record.get("value"),
                    "C": params.get("C"),
                    "gamma": params.get("gamma"),
                    "class_weight": params.get("class_weight"),
                    "fold_std": record.get("fold_std"),
                    "fold_scores": ";".join(
                        f"{score:.6f}"
                        for score in record.get("fold_scores", [])
                    ),
                }
            )
    return path


def write_run(
    run_dir: str | Path,
    result: dict[str, Any],
    *,
    make_plots: bool = True,
) -> dict[str, str]:
    """Persist a :func:`~hotb_2026.classification.training.train` result."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    written["config"] = str(
        _write_json(run_dir / "config.json", result.get("config", {}))
    )
    written["dataset_summary"] = str(
        _write_json(
            run_dir / "dataset_summary.json",
            result.get("dataset_summary") or {},
        )
    )
    written["embeddings_meta"] = str(
        _write_json(
            run_dir / "embeddings.meta.json",
            result.get("embedding_meta") or {},
        )
    )
    written["best_params"] = str(
        _write_json(
            run_dir / "best_params.json",
            {
                "params": result.get("best_params", {}),
                "best_value": result.get("best_value"),
                "metric": result.get("cv", {}).get("metric"),
                "cv": result.get("cv", {}),
            },
        )
    )
    written["metrics"] = str(
        _write_json(
            run_dir / "metrics.json",
            {
                "split": result.get("split", {}),
                "cv": result.get("cv", {}),
                "oof": (result.get("oof") or {}).get("metrics", {}),
                "holdout": (result.get("holdout") or {}).get("metrics", {}),
            },
        )
    )
    written["trials"] = str(
        _write_trials(run_dir, result.get("trials", []))
    )

    holdout = result.get("holdout") or {}
    oof = result.get("oof") or {}
    payload: dict[str, Any] = {
        "holdout_indices": np.asarray(holdout.get("indices", []), dtype=int),
        "holdout_y_true": np.asarray(holdout.get("y_true", []), dtype=int),
        "holdout_y_pred": np.asarray(holdout.get("y_pred", []), dtype=int),
        "oof_indices": np.asarray(oof.get("indices", []), dtype=int),
        "oof_y_true": np.asarray(oof.get("y_true", []), dtype=int),
        "oof_y_pred": np.asarray(oof.get("y_pred", []), dtype=int),
    }
    if holdout.get("y_proba") is not None:
        payload["holdout_y_proba"] = np.asarray(holdout["y_proba"])
    np.savez_compressed(run_dir / "predictions.npz", **payload)
    written["predictions"] = str(run_dir / "predictions.npz")

    if make_plots:
        try:
            metrics = holdout.get("metrics", {})
            if metrics.get("confusion_matrix"):
                written["confusion_matrix"] = str(
                    plots.plot_confusion_matrix(
                        metrics["confusion_matrix"],
                        result.get("label_names", []),
                        run_dir / "confusion_matrix.png",
                        title="Held-out confusion matrix",
                    )
                )
            trial_plot = plots.plot_trials(
                result.get("trials", []),
                run_dir / "trials.png",
                metric=result.get("cv", {}).get(
                    "metric", "balanced_accuracy"
                ),
            )
            if trial_plot is not None:
                written["trials_plot"] = str(trial_plot)
            embeddings = result.get("embeddings")
            oof_indices = np.asarray(oof.get("indices", []), dtype=int)
            oof_y = np.asarray(oof.get("y_true", []), dtype=int)
            if (
                embeddings is not None
                and oof_indices.size == oof_y.size
                and oof_indices.size > 0
            ):
                embedding_plot = plots.plot_embeddings(
                    np.asarray(embeddings)[oof_indices],
                    oof_y,
                    result.get("label_names", []),
                    run_dir / "embedding_pca.png",
                )
                if embedding_plot is not None:
                    written["embedding_pca"] = str(embedding_plot)
            if holdout.get("y_proba") is not None:
                confidence_plot = plots.plot_confidence(
                    holdout.get("y_true", []),
                    holdout.get("y_pred", []),
                    holdout["y_proba"],
                    run_dir / "confidence.png",
                )
                if confidence_plot is not None:
                    written["confidence"] = str(confidence_plot)
        except ImportError:  # pragma: no cover - matplotlib is a dev dependency
            log.warning("matplotlib is not installed; skipping plots")

    report_path = run_dir / "report.md"
    report_path.write_text(render_report(result))
    written["report"] = str(report_path)
    return written


__all__ = ["render_report", "write_run"]
