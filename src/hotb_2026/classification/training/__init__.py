"""Training-only package: dataset -> frozen REVE embeddings -> SVM."""
from .config import TrainingConfig
from .search import (
    C_RANGE,
    CLASS_WEIGHTS,
    GAMMA_RANGE,
    cross_validate,
    make_estimator,
    normalize_params,
    run_study,
    suggest_params,
)
from .splits import GroupHoldout, group_holdout, inner_cv
from .trainer import (
    embed_dataset,
    train,
    train_from_embeddings,
    trial_records,
)

__all__ = [
    "C_RANGE",
    "CLASS_WEIGHTS",
    "GAMMA_RANGE",
    "GroupHoldout",
    "TrainingConfig",
    "cross_validate",
    "embed_dataset",
    "group_holdout",
    "inner_cv",
    "make_estimator",
    "normalize_params",
    "run_study",
    "suggest_params",
    "train",
    "train_from_embeddings",
    "trial_records",
]
