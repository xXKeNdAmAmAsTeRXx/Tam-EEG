"""Matplotlib figures for a training run.

Every function takes an output path and returns it (or ``None`` when the data
cannot produce a meaningful figure). Matplotlib is imported lazily so the
training pipeline works in environments where only the core dependencies are
installed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_confusion_matrix(
    matrix: Any,
    label_names: list[str] | tuple[str, ...],
    path: str | Path,
    *,
    title: str = "Confusion matrix",
) -> Path:
    plt = _pyplot()
    matrix = np.asarray(matrix, dtype=float)
    labels = [str(name) for name in label_names]

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            ax.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_trials(
    records: list[dict[str, Any]],
    path: str | Path,
    *,
    metric: str = "balanced_accuracy",
) -> Path | None:
    completed = [row for row in records if row.get("value") is not None]
    if not completed:
        return None

    plt = _pyplot()
    numbers = np.array([row["number"] for row in completed])
    values = np.array([row["value"] for row in completed], dtype=float)
    running_best = np.maximum.accumulate(values)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.scatter(numbers, values, s=18, alpha=0.65, label="trial")
    ax.step(numbers, running_best, where="post", color="C1", label="best so far")
    ax.set(
        xlabel="Trial",
        ylabel=metric.replace("_", " "),
        title="Optuna search",
    )
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_embeddings(
    embeddings: np.ndarray,
    y: np.ndarray,
    label_names: list[str] | tuple[str, ...],
    path: str | Path,
) -> Path | None:
    embeddings = np.asarray(embeddings, dtype=float)
    y = np.asarray(y).astype(int)
    if embeddings.ndim != 2 or embeddings.shape[0] < 3 or y.size != embeddings.shape[0]:
        return None

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    projected = PCA(n_components=2, random_state=0).fit_transform(
        StandardScaler().fit_transform(embeddings)
    )

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    labels = np.asarray(label_names)
    for value in np.unique(y):
        mask = y == value
        name = labels[value] if value < labels.size else str(value)
        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            s=22,
            alpha=0.75,
            label=f"{name} (n={int(mask.sum())})",
        )
    ax.set(
        xlabel="PC 1",
        ylabel="PC 2",
        title="REVE embeddings (PCA)",
    )
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confidence(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    path: str | Path,
) -> Path | None:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    if y_proba.ndim != 2 or y_proba.shape[0] != y_true.size:
        return None

    confidence = y_proba.max(axis=1)
    correct = y_true == y_pred

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    bins = np.linspace(0.0, 1.0, 21)
    if np.any(correct):
        ax.hist(
            confidence[correct], bins=bins, alpha=0.65, label="correct"
        )
    if np.any(~correct):
        ax.hist(
            confidence[~correct], bins=bins, alpha=0.65, label="wrong"
        )
    ax.set(
        xlabel="Maximum predicted probability",
        ylabel="Windows",
        title="Holdout confidence",
    )
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


__all__ = [
    "plot_confidence",
    "plot_confusion_matrix",
    "plot_embeddings",
    "plot_trials",
]
