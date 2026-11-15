"""Metrics, figures and Markdown reports for training runs."""
from .metrics import (
    SUPPORTED_METRICS,
    align_probabilities,
    classification_metrics,
    score_metric,
)
from .plots import (
    plot_confidence,
    plot_confusion_matrix,
    plot_embeddings,
    plot_trials,
)
from .report import render_report, write_run

__all__ = [
    "SUPPORTED_METRICS",
    "align_probabilities",
    "classification_metrics",
    "plot_confidence",
    "plot_confusion_matrix",
    "plot_embeddings",
    "plot_trials",
    "render_report",
    "score_metric",
    "write_run",
]
