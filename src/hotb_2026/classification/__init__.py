"""Cognitive-overload classification: dataset, training and the REVE+SVM model."""
from .dataset import (
    EegDataset,
    Recording,
    StateChange,
    build_dataset,
    load_recordings,
    make_synthetic_recording,
)
from .model import ReveSVMClassifier
from .training import TrainingConfig, train, train_from_embeddings

__all__ = [
    "EegDataset",
    "Recording",
    "ReveSVMClassifier",
    "StateChange",
    "TrainingConfig",
    "build_dataset",
    "load_recordings",
    "make_synthetic_recording",
    "train",
    "train_from_embeddings",
]
