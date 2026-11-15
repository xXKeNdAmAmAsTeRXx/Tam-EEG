"""Optuna search over the SVM head on precomputed REVE embeddings.

Only the SVM hyperparameters are tuned. The REVE encoder is frozen and the
embeddings are computed once, so a trial costs one small scikit-learn fit per
fold instead of a transformer forward pass.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from hotb_2026.reporting.metrics import score_metric

from .config import TrainingConfig
from .splits import inner_cv

log = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

C_RANGE = (1e-2, 1e3)
GAMMA_RANGE = (1e-4, 1e0)
CLASS_WEIGHTS = ("none", "balanced")


def normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Convert Optuna's JSON-friendly parameter values to sklearn values."""
    class_weight = params.get("class_weight", "none")
    if isinstance(class_weight, str):
        class_weight = None if class_weight == "none" else class_weight
    return {
        "C": float(params["C"]),
        "gamma": float(params["gamma"]),
        "class_weight": class_weight,
    }


def make_estimator(
    params: dict[str, Any],
    *,
    seed: int = 0,
    probability: bool = False,
    calibration_cv: int = 3,
) -> Pipeline:
    """Standardize embeddings and fit an RBF SVM.

    With ``probability=True`` the SVM is wrapped in
    :class:`~sklearn.calibration.CalibratedClassifierCV` (sigmoid/Platt,
    ``ensemble=False``), which is the scikit-learn replacement for the
    deprecated ``SVC(probability=True)``.
    """
    params = normalize_params(params)
    svc = SVC(
        kernel="rbf",
        C=params["C"],
        gamma=params["gamma"],
        class_weight=params["class_weight"],
        random_state=seed,
    )
    final: Any = svc
    if probability:
        if calibration_cv < 2:
            raise ValueError("calibration_cv must be at least 2")
        final = CalibratedClassifierCV(
            svc, method="sigmoid", cv=calibration_cv, ensemble=False
        )
    return Pipeline([("scaler", StandardScaler()), ("svc", final)])


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "C": trial.suggest_float("C", *C_RANGE, log=True),
        "gamma": trial.suggest_float("gamma", *GAMMA_RANGE, log=True),
        "class_weight": trial.suggest_categorical(
            "class_weight", list(CLASS_WEIGHTS)
        ),
    }


def cross_validate(
    embeddings: np.ndarray,
    y: np.ndarray,
    *,
    params: dict[str, Any],
    cv: StratifiedGroupKFold,
    groups: np.ndarray,
    labels: np.ndarray,
    metric: str = "balanced_accuracy",
    seed: int = 0,
    trial: optuna.Trial | None = None,
) -> list[float]:
    """Fit ``params`` on every fold and return the per-fold scores."""
    scores: list[float] = []
    for step, (train_index, test_index) in enumerate(
        cv.split(embeddings, y, groups=groups)
    ):
        estimator = make_estimator(params, seed=seed, probability=False)
        estimator.fit(embeddings[train_index], y[train_index])
        predictions = estimator.predict(embeddings[test_index])
        score = score_metric(
            y[test_index], predictions, metric, labels=labels
        )
        scores.append(float(score))

        if trial is not None:
            trial.report(float(np.mean(scores)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return scores


def run_study(
    embeddings: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    config: TrainingConfig,
    *,
    labels: np.ndarray | None = None,
    cv: StratifiedGroupKFold | None = None,
    n_splits: int | None = None,
) -> optuna.Study:
    """Run the Optuna search and return the study.

    The study's best trial carries ``fold_scores`` in ``user_attrs`` so the
    report can show the variance behind the selected score.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    if labels is None:
        labels = np.arange(int(y.max()) + 1)
    labels = np.asarray(labels, dtype=int)

    if cv is None:
        cv, n_splits = inner_cv(
            groups, y, n_splits=config.n_splits, seed=config.seed
        )
    if n_splits is None:
        n_splits = cv.get_n_splits()
    # Trials can repeat parameters; reuse the score instead of refitting.
    cache: dict[tuple[float, float, str], tuple[float, list[float]]] = {}

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        key = (
            round(params["C"], 12),
            round(params["gamma"], 12),
            params["class_weight"],
        )
        if key in cache:
            value, cached_scores = cache[key]
            trial.set_user_attr("fold_scores", cached_scores)
            trial.set_user_attr("fold_std", float(np.std(cached_scores)))
            return value

        scores = cross_validate(
            embeddings,
            y,
            params=params,
            cv=cv,
            groups=groups,
            labels=labels,
            metric=config.metric,
            seed=config.seed,
            trial=trial,
        )
        value = float(np.mean(scores))
        cache[key] = (value, scores)
        trial.set_user_attr("fold_scores", scores)
        trial.set_user_attr("fold_std", float(np.std(scores)))
        return value

    sampler = optuna.samplers.TPESampler(seed=config.seed, multivariate=True)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=min(10, max(2, config.n_trials // 5)),
        n_warmup_steps=1,
    )
    study = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=pruner
    )
    study.set_user_attr("metric", config.metric)
    study.set_user_attr("n_splits", n_splits)

    # A sensible starting point: C=1, gamma near sklearn's "scale" heuristic,
    # balanced classes. TPE then explores from a working configuration instead
    # of spending the first trials in a useless corner of the space.
    scale_gamma = 1.0 / max(1, embeddings.shape[1])
    study.enqueue_trial(
        {
            "C": 1.0,
            "gamma": float(
                min(GAMMA_RANGE[1], max(GAMMA_RANGE[0], scale_gamma))
            ),
            "class_weight": "balanced",
        },
        skip_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=config.n_trials,
        timeout=config.timeout_s,
        show_progress_bar=False,
    )

    if not study.trials:
        raise RuntimeError("Optuna produced no trials")
    completed = [trial for trial in study.trials if trial.value is not None]
    if not completed:
        raise RuntimeError("all Optuna trials were pruned or failed")
    return study


__all__ = [
    "C_RANGE",
    "CLASS_WEIGHTS",
    "GAMMA_RANGE",
    "cross_validate",
    "make_estimator",
    "normalize_params",
    "run_study",
    "suggest_params",
]
