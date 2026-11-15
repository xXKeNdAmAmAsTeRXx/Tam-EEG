from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from hotb_2026.classification.dataset import EegDataset
from hotb_2026.classification.training import (
    TrainingConfig,
    group_holdout,
    inner_cv,
    normalize_params,
    train,
    train_from_embeddings,
)
from hotb_2026.reporting import write_run


class FakeEncoder:
    """Tiny deterministic stand-in for the frozen REVE encoder."""

    def __init__(self, electrodes) -> None:
        self.electrodes = tuple(electrodes)
        self.model_name = "fake-reve"
        self.pos_model_name = "fake-positions"
        self.patch_size = 64
        self.revision = "test-revision"
        self.embedding_dim = len(self.electrodes)

    def encode(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return X.mean(axis=2).astype(np.float32)


def make_dataset(
    n_classes: int = 3,
    groups_per_class: int = 4,
    windows_per_group: int = 3,
    n_channels: int = 4,
    n_samples: int = 64,
    seed: int = 0,
) -> EegDataset:
    """Class-dependent channel means so the fake encoder is separable."""
    rng = np.random.default_rng(seed)
    X: list[np.ndarray] = []
    y: list[int] = []
    groups: list[str] = []
    for label in range(n_classes):
        for group in range(groups_per_class):
            for _ in range(windows_per_group):
                signal = rng.normal(0.0, 0.5, size=(n_channels, n_samples))
                signal += 4.0 * label
                X.append(signal.astype(np.float32))
                y.append(label)
                groups.append(f"g{label}-{group}")

    n = len(X)
    return EegDataset(
        X=np.stack(X),
        y=np.asarray(y, dtype=np.int8),
        groups=np.asarray(groups, dtype=str),
        window_start_s=np.arange(n, dtype=float),
        window_end_s=np.arange(n, dtype=float) + 1.0,
        state_change_s=np.zeros(n, dtype=float),
        electrodes=tuple(f"E{index}" for index in range(n_channels)),
        sfreq=200.0,
        raw_sfreq=250.0,
        window_s=1.0,
        label_names=("low", "medium", "high"),
    )


@pytest.fixture(scope="module")
def dataset() -> EegDataset:
    return make_dataset()


@pytest.fixture(scope="module")
def encoder(dataset: EegDataset) -> FakeEncoder:
    return FakeEncoder(dataset.electrodes)


@pytest.fixture(scope="module")
def config() -> TrainingConfig:
    return TrainingConfig(n_splits=3, n_trials=6, test_size=0.25, seed=0)


@pytest.fixture(scope="module")
def result(dataset, encoder, config):
    return train(dataset, config=config, encoder=encoder)


def test_group_holdout_keeps_groups_intact(dataset: EegDataset) -> None:
    split = group_holdout(
        dataset.groups, dataset.y, test_size=0.25, seed=0
    )
    train_groups = set(dataset.groups[split.train])
    test_groups = set(dataset.groups[split.test])

    assert train_groups.isdisjoint(test_groups)
    assert split.n_train + split.n_test == dataset.n_windows
    assert set(split.test_groups) == test_groups

    again = group_holdout(dataset.groups, dataset.y, test_size=0.25, seed=0)
    np.testing.assert_array_equal(split.train, again.train)
    np.testing.assert_array_equal(split.test, again.test)


def test_group_holdout_handles_few_groups() -> None:
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    y = np.array([0, 0, 1, 1, 2, 2])
    split = group_holdout(groups, y, test_size=0.2, seed=0)
    assert split.n_test > 0
    assert split.n_train > 0
    assert set(groups[split.train]).isdisjoint(set(groups[split.test]))


def test_inner_cv_reduces_folds_for_few_groups() -> None:
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    y = np.array([0, 0, 1, 1, 2, 2])
    cv, folds = inner_cv(groups, y, n_splits=5, seed=0)
    assert folds == 2
    splits = list(cv.split(np.zeros_like(y), y, groups=groups))
    assert len(splits) == 2
    assert all(len(train) and len(test) for train, test in splits)


def test_normalize_params_maps_none_class_weight() -> None:
    params = normalize_params(
        {"C": "1.5", "gamma": 0.01, "class_weight": "none"}
    )
    assert params == {"C": 1.5, "gamma": 0.01, "class_weight": None}
    balanced = normalize_params(
        {"C": 1.0, "gamma": 0.1, "class_weight": "balanced"}
    )
    assert balanced["class_weight"] == "balanced"


def test_train_end_to_end(dataset, result) -> None:
    assert len(result["trials"]) == 6
    assert result["cv"]["n_splits"] == 3
    assert result["cv"]["mean"] >= 0.9
    assert result["holdout"]["metrics"]["accuracy"] >= 0.9
    assert result["oof"]["metrics"]["accuracy"] >= 0.9

    n_test = len(result["holdout"]["y_true"])
    proba = result["holdout"]["y_proba"]
    assert proba.shape == (n_test, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    assert np.all(result["oof"]["y_pred"] >= 0)
    assert isinstance(result["model"], Pipeline)
    assert isinstance(result["holdout_model"], Pipeline)

    final_proba = result["model"].predict_proba(result["embeddings"])
    assert final_proba.shape == (dataset.n_windows, 3)
    assert set(np.asarray(result["model"].predict(result["embeddings"]))) <= {
        0,
        1,
        2,
    }

    meta = result["embedding_meta"]
    assert meta["model_name"] == "fake-reve"
    assert meta["pos_model_name"] == "fake-positions"
    assert meta["patch_size"] == 64
    assert meta["revision"] == "test-revision"
    assert meta["preprocessing"] == dataset.preprocessing
    assert result["dataset_summary"]["preprocessing"] == dataset.preprocessing


def test_train_from_embeddings_matches_training_inputs(dataset, encoder) -> None:
    embeddings = encoder.encode(dataset.X)
    result = train_from_embeddings(
        embeddings,
        dataset.y,
        dataset.groups,
        config=TrainingConfig(n_splits=3, n_trials=3, seed=1),
        label_names=dataset.label_names,
    )
    assert result["embedding_meta"]["shape"] == list(embeddings.shape)
    assert result["best_params"]["C"] > 0
    assert len(result["cv"]["scores"]) == 3


def test_embedding_cache_is_reused(dataset, encoder, tmp_path) -> None:
    cache = tmp_path / "embeddings"
    config = TrainingConfig(n_splits=3, n_trials=2, seed=0)

    first = train(
        dataset, config=config, encoder=encoder, cache_dir=str(cache)
    )
    cache_file = cache / "embeddings.npy"
    assert cache_file.exists()
    mtime = cache_file.stat().st_mtime_ns

    second = train(
        dataset, config=config, encoder=encoder, cache_dir=str(cache)
    )
    assert cache_file.stat().st_mtime_ns == mtime
    np.testing.assert_allclose(first["embeddings"], second["embeddings"])
    assert json.loads((cache / "embeddings.json").read_text())[
        "model_name"
    ] == "fake-reve"


def test_embedding_cache_invalidates_on_preprocessing(tmp_path) -> None:
    cache = tmp_path / "embeddings"
    config = TrainingConfig(n_splits=3, n_trials=2, seed=0)

    first_dataset = make_dataset()
    first_dataset.preprocessing = {"notch_freq": 50.0}
    train(
        first_dataset,
        config=config,
        encoder=FakeEncoder(first_dataset.electrodes),
        cache_dir=str(cache),
    )
    first_key = json.loads((cache / "embeddings.json").read_text())["fingerprint"]

    second_dataset = make_dataset()
    second_dataset.preprocessing = {"notch_freq": 60.0}
    train(
        second_dataset,
        config=config,
        encoder=FakeEncoder(second_dataset.electrodes),
        cache_dir=str(cache),
    )
    second_key = json.loads((cache / "embeddings.json").read_text())[
        "fingerprint"
    ]

    assert first_key != second_key


def test_write_run_creates_all_artifacts(result, tmp_path) -> None:
    run_dir = tmp_path / "run"
    written = write_run(run_dir, result)

    expected = [
        "config.json",
        "dataset_summary.json",
        "embeddings.meta.json",
        "trials.csv",
        "best_params.json",
        "metrics.json",
        "predictions.npz",
        "report.md",
        "confusion_matrix.png",
        "trials.png",
        "embedding_pca.png",
        "confidence.png",
    ]
    for name in expected:
        assert (run_dir / name).exists(), name

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["holdout"]["accuracy"] == pytest.approx(
        result["holdout"]["metrics"]["accuracy"]
    )
    assert metrics["cv"]["mean"] == pytest.approx(result["cv"]["mean"])

    report = (run_dir / "report.md").read_text()
    assert "Held-out test groups" in report
    assert "Confusion matrix" in report
    assert "balanced accuracy" in report.lower()

    with np.load(run_dir / "predictions.npz") as predictions:
        assert predictions["holdout_y_pred"].shape == (
            len(result["holdout"]["y_true"]),
        )

    assert written["report"].endswith("report.md")


def test_cli_builds_synthetic_dataset() -> None:
    from hotb_2026.classification.training.__main__ import (
        _load_dataset,
        build_parser,
    )

    args = build_parser().parse_args(
        [
            "--synthetic",
            "--synthetic-sessions",
            "2",
            "--synthetic-duration",
            "60",
            "--state-duration",
            "20",
        ]
    )
    dataset = _load_dataset(args)
    assert dataset.n_windows == 12
    assert set(dataset.y) == {0, 1, 2}
