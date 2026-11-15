"""Classification metrics shared by the trainer and the reporting package."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)

SUPPORTED_METRICS = ("balanced_accuracy", "accuracy", "f1_macro")


def score_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "balanced_accuracy",
    *,
    labels: Any = None,
) -> float:
    """Single-number score used for model selection.

    ``labels`` should list every class of the problem. Classes missing from a
    fold then count as zero recall instead of silently disappearing from the
    average.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if metric == "balanced_accuracy":
        return float(
            recall_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "f1_macro":
        return float(
            precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )[2]
        )
    raise ValueError(
        f"unsupported metric {metric!r}; expected one of {SUPPORTED_METRICS}"
    )


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Any,
    label_names: Any,
    y_proba: np.ndarray | None = None,
    metric: str = "balanced_accuracy",
) -> dict[str, Any]:
    """Full metric bundle for one set of predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    labels = np.asarray(labels, dtype=int)
    label_names = [str(name) for name in label_names]

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got "
            f"{y_true.shape} and {y_pred.shape}"
        )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    result: dict[str, Any] = {
        "n_samples": int(y_true.size),
        "labels": labels.tolist(),
        "label_names": label_names,
        "score": score_metric(y_true, y_pred, metric, labels=labels),
        "metric": metric,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            recall_score(
                y_true, y_pred, labels=labels, average="macro", zero_division=0
            )
        ),
        "f1_macro": float(
            precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )[2]
        ),
        "confusion_matrix": matrix.tolist(),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(label_names)
        },
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba, dtype=float)
        if y_proba.ndim != 2 or y_proba.shape[0] != y_true.size:
            raise ValueError(
                f"y_proba must be (n_samples, n_classes), got {y_proba.shape}"
            )
        confidence = y_proba.max(axis=1)
        correct = y_pred == y_true
        result["mean_confidence"] = float(confidence.mean())
        result["mean_confidence_correct"] = (
            float(confidence[correct].mean()) if np.any(correct) else None
        )
        result["mean_confidence_wrong"] = (
            float(confidence[~correct].mean()) if np.any(~correct) else None
        )
        if y_proba.shape[1] == labels.size and np.unique(y_true).size > 1:
            present = np.unique(y_true)
            positions = [
                int(np.flatnonzero(labels == value)[0]) for value in present
            ]
            try:
                result["roc_auc_ovr"] = float(
                    roc_auc_score(
                        y_true,
                        y_proba[:, positions],
                        multi_class="ovr",
                        labels=present,
                        average="macro",
                    )
                )
            except ValueError:
                # For example, one class is absent from this split.
                result["roc_auc_ovr"] = None

    return result


def align_probabilities(
    estimator: Any, X: np.ndarray, labels: np.ndarray
) -> np.ndarray | None:
    """Expand ``predict_proba`` to a full class matrix in ``labels`` order.

    A fold can miss a class; its probability columns then cover only the
    classes seen during ``fit``. Missing columns are filled with zeros.
    """
    if not hasattr(estimator, "predict_proba"):
        return None
    proba = np.asarray(estimator.predict_proba(X), dtype=float)
    classes = np.asarray(getattr(estimator, "classes_", []), dtype=int)
    labels = np.asarray(labels, dtype=int)
    if classes.size == 0:
        return proba
    if proba.shape[1] != classes.size:
        return None

    full = np.zeros((proba.shape[0], labels.size), dtype=float)
    position = {int(label): index for index, label in enumerate(labels)}
    for column, label in enumerate(classes):
        if int(label) in position:
            full[:, position[int(label)]] = proba[:, column]
    return full


__all__ = [
    "SUPPORTED_METRICS",
    "align_probabilities",
    "classification_metrics",
    "score_metric",
]
