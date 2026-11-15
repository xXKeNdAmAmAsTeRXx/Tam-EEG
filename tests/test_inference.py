from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from hotb_2026.features import ReveEncoder
from hotb_2026.features.cache import load_or_encode
from hotb_2026.inference import (
    BundleContract,
    Prediction,
    Predictor,
    load_bundle,
)
from hotb_2026.inference import loader as inference_loader
from hotb_2026.model_store import StoreConfig, save_model
from hotb_2026.model_store.bundle import METADATA_FILE, MODEL_FILE

ELECTRODES = ("E0", "E1", "E2", "E3")
LABELS = ("low", "high")
WINDOW_S = 1.0
RAW_SFREQ = 250.0
SFREQ = 200.0
RAW_SAMPLES = 250
N_SAMPLES = 200
PREPROCESSING = {
    "sfreq": RAW_SFREQ,
    "notch_freq": 50.0,
    "notch_q": 30.0,
    "bandpass": (0.5, 45.0),
    "filter_order": 4,
    "target_sfreq": SFREQ,
    "zscore": True,
    "clip": 15.0,
}


class _ThresholdModel(BaseEstimator, ClassifierMixin):
    """Tiny fitted-looking classifier: one feature, two classes."""

    def __init__(self, threshold: float = 0.0, scale: float = 5.0) -> None:
        self.threshold = threshold
        self.scale = scale
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = 1

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_ThresholdModel":
        self.classes_ = np.unique(np.asarray(y, dtype=int))
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        x = np.asarray(X, dtype=np.float64).reshape(len(X), -1)[:, 0]
        positive = 1.0 / (1.0 + np.exp(-self.scale * (x - self.threshold)))
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class _PredictOnly:
    classes_ = np.array([0, 1])
    n_features_in_ = 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=int)


class _RampEncoder:
    """Deterministic stand-in for REVE: rising vs falling windows differ."""

    def __init__(self, electrodes=ELECTRODES) -> None:
        self.electrodes = tuple(electrodes)
        self.model_name = "test-reve"
        self.pos_model_name = "test-positions"
        self.revision = "test-revision"
        self.patch_size = 1
        self.embedding_dim = 1
        self.load_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    def encode(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[None, ...]
        half = X.shape[-1] // 2
        feature = X[:, 0, half:].mean(axis=1) - X[:, 0, :half].mean(axis=1)
        return feature[:, None].astype(np.float32)


def _result() -> dict:
    return {
        "model": _ThresholdModel(),
        "label_names": list(LABELS),
        "labels": list(range(len(LABELS))),
        "embedding_meta": {
            "model_name": "test-reve",
            "pos_model_name": "test-positions",
            "embedding_dim": 1,
            "patch_size": 1,
            "revision": "test-revision",
            "electrodes": list(ELECTRODES),
            "window_s": WINDOW_S,
            "sfreq": SFREQ,
            "raw_sfreq": RAW_SFREQ,
            "n_windows": 12,
            "shape": [12, 1],
            "preprocessing": dict(PREPROCESSING),
        },
        "dataset_summary": {
            "n_windows": 12,
            "n_channels": len(ELECTRODES),
            "n_samples": N_SAMPLES,
            "sfreq": SFREQ,
            "raw_sfreq": RAW_SFREQ,
            "window_s": WINDOW_S,
            "class_counts": {LABELS[0]: 6, LABELS[1]: 6},
            "n_groups": 2,
            "groups": ["a", "b"],
            "label_names": list(LABELS),
            "electrodes": list(ELECTRODES),
            "preprocessing": dict(PREPROCESSING),
        },
        "config": {},
        "cv": {},
        "holdout": {},
        "split": {},
    }


@pytest.fixture(scope="module")
def bundle(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("bundle") / "model"
    save_model(
        _result(),
        directory,
        config=StoreConfig(version="1.2.3"),
    )
    return directory


@pytest.fixture()
def predictor(bundle: Path) -> Predictor:
    return Predictor.from_bundle(bundle, encoder=_RampEncoder())


def _ramp_window(
    up: bool = True,
    *,
    n_channels: int = len(ELECTRODES),
    samples: int = RAW_SAMPLES,
) -> np.ndarray:
    base = np.linspace(-1.0, 1.0, samples)
    if not up:
        base = base[::-1]
    scales = np.arange(1, n_channels + 1, dtype=np.float64)[:, None]
    return (base[None, :] * scales).astype(np.float32)


def _copy_bundle(bundle: Path, tmp_path: Path, name: str = "broken") -> Path:
    target = tmp_path / name
    shutil.copytree(bundle, target)
    return target


def _rewrite_metadata(directory: Path, mutate) -> None:
    path = directory / METADATA_FILE
    metadata = json.loads(path.read_text())
    mutate(metadata)
    path.write_text(json.dumps(metadata))


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def test_load_bundle_exposes_the_contract(bundle: Path) -> None:
    loaded = load_bundle(bundle)

    assert isinstance(loaded.contract, BundleContract)
    assert loaded.contract.model_version == "1.2.3"
    assert loaded.contract.electrodes == ELECTRODES
    assert loaded.contract.raw_samples == RAW_SAMPLES
    assert loaded.contract.n_channels == len(ELECTRODES)
    assert loaded.contract.n_samples == N_SAMPLES
    assert loaded.contract.label_names == LABELS
    assert loaded.contract.embedding_dim == 1
    assert loaded.contract.encoder_revision == "test-revision"
    assert loaded.contract.preprocessing.sfreq == RAW_SFREQ
    assert loaded.contract.metadata["model_version"] == "1.2.3"


def test_predictor_exposes_the_bundle(bundle: Path) -> None:
    predictor = Predictor.from_bundle(bundle, encoder=_RampEncoder())

    assert predictor.model_version == "1.2.3"
    assert predictor.bundle_dir == bundle
    assert predictor.electrodes == ELECTRODES
    assert predictor.label_names == LABELS
    assert predictor.window_s == WINDOW_S
    assert predictor.raw_sfreq == RAW_SFREQ
    assert predictor.sfreq == SFREQ
    assert predictor.raw_samples == RAW_SAMPLES
    assert predictor.metadata["labels"]["names"] == list(LABELS)
    assert "Predictor(model_version='1.2.3'" in repr(predictor)


def test_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="bundle directory not found"):
        Predictor.from_bundle(tmp_path / "missing", encoder=_RampEncoder())


def test_missing_model_file(bundle: Path, tmp_path: Path) -> None:
    target = _copy_bundle(bundle, tmp_path)
    (target / MODEL_FILE).unlink()

    with pytest.raises(FileNotFoundError, match="model.joblib"):
        Predictor.from_bundle(target, encoder=_RampEncoder())


def test_unsupported_schema(bundle: Path, tmp_path: Path) -> None:
    target = _copy_bundle(bundle, tmp_path)
    _rewrite_metadata(
        target,
        lambda metadata: metadata.update(artifact_schema_version=99),
    )

    with pytest.raises(ValueError, match="artifact schema"):
        Predictor.from_bundle(target, encoder=_RampEncoder())


def test_missing_model_version(bundle: Path, tmp_path: Path) -> None:
    target = _copy_bundle(bundle, tmp_path)
    _rewrite_metadata(target, lambda metadata: metadata.pop("model_version"))

    with pytest.raises(ValueError, match="model_version"):
        Predictor.from_bundle(target, encoder=_RampEncoder())


def test_invalid_json_metadata(bundle: Path, tmp_path: Path) -> None:
    target = _copy_bundle(bundle, tmp_path)
    (target / METADATA_FILE).write_text("{not json")

    with pytest.raises(ValueError, match="not valid JSON"):
        Predictor.from_bundle(target, encoder=_RampEncoder())


def test_estimator_without_predict_proba(bundle: Path, tmp_path: Path) -> None:
    target = _copy_bundle(bundle, tmp_path)
    joblib.dump(_PredictOnly(), target / MODEL_FILE)

    with pytest.raises(TypeError, match="predict_proba"):
        Predictor.from_bundle(target, encoder=_RampEncoder())


def test_model_classes_must_match_metadata(bundle: Path, tmp_path: Path) -> None:
    target = _copy_bundle(bundle, tmp_path)
    model = _ThresholdModel()
    model.classes_ = np.array([0, 2])
    joblib.dump(model, target / MODEL_FILE)

    with pytest.raises(ValueError, match="fitted on classes"):
        Predictor.from_bundle(target, encoder=_RampEncoder())


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda m: m["labels"].pop("names"), "labels.names"),
        (lambda m: m["labels"].update(names=["low"]), "classes"),
        (lambda m: m["labels"].update(classes=[0, 2]), "classes"),
        (lambda m: m["labels"].update(n_classes=3), "n_classes"),
        (lambda m: m["labels"].update(names=["low", "high", "high"]), "unique"),
        (
            lambda m: m["encoder"].update(embedding_dim=9),
            "embedding_dim",
        ),
        (lambda m: m["encoder"].update(model_name=""), "model_name"),
        (lambda m: m["encoder"].update(revision=""), "revision"),
        (lambda m: m["encoder"].update(patch_size=9999), "patch_size"),
        (lambda m: m["input"].update(electrodes=[]), "electrodes"),
        (lambda m: m["input"].update(window_s=0), "window_s"),
        (lambda m: m["input"].update(raw_sfreq=500.0), "raw_sfreq"),
        (lambda m: m["input"].update(sfreq=100.0), "input.sfreq"),
        (lambda m: m["input"].update(n_samples=123), "n_samples"),
        (lambda m: m["input"].update(time_units="ms"), "time_units"),
        (lambda m: m.update(preprocessing={}), "preprocessing"),
        (
            lambda m: m["preprocessing"].update(unknown=1),
            "unknown field",
        ),
        (
            lambda m: m["preprocessing"].update(sfreq=125.0),
            "preprocessing.sfreq",
        ),
        (
            lambda m: m["preprocessing"].update(bandpass=(0.5, 120.0)),
            "preprocessing",
        ),
    ],
)
def test_inconsistent_metadata_is_rejected(
    bundle: Path, tmp_path: Path, mutate, match: str
) -> None:
    target = _copy_bundle(bundle, tmp_path)
    _rewrite_metadata(target, mutate)

    with pytest.raises(ValueError, match=match):
        Predictor.from_bundle(target, encoder=_RampEncoder())


# --------------------------------------------------------------------------- #
# prediction contract
# --------------------------------------------------------------------------- #
def test_prediction_matches_the_fitted_model(predictor: Predictor) -> None:
    raw = _ramp_window(up=True)
    prediction = predictor.predict_raw(raw, current_time=10.0)

    processed = predictor.preprocessor.process(raw)
    features = predictor.encoder.encode(processed)
    expected = predictor.model.predict_proba(features)[0]

    assert isinstance(prediction, Prediction)
    assert prediction.label == int(np.argmax(expected))
    assert prediction.label_name == LABELS[prediction.label]
    assert prediction.confidence == pytest.approx(float(expected.max()))
    assert list(prediction.probabilities) == list(LABELS)
    assert prediction.probabilities[LABELS[0]] == pytest.approx(float(expected[0]))
    assert prediction.probabilities[LABELS[1]] == pytest.approx(float(expected[1]))
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0)
    assert prediction.current_time == 10.0
    assert prediction.window_start == 9.0
    assert prediction.window_end == 10.0
    assert prediction.state_change_time == 10.0
    assert prediction.model_version == "1.2.3"


def test_processed_and_raw_paths_agree(predictor: Predictor) -> None:
    raw = _ramp_window(up=False)
    processed = predictor.preprocessor.process(raw)

    from_processed = predictor.predict(processed, current_time=4.0)
    predictor.reset()
    from_raw = predictor.predict_raw(raw, current_time=4.0)

    assert from_processed == from_raw


def test_predict_raw_uses_the_trailing_samples(predictor: Predictor) -> None:
    raw = _ramp_window(up=True)
    padded = np.concatenate(
        [np.full((len(ELECTRODES), 50), 1e6, dtype=np.float32), raw], axis=1
    )

    expected = predictor.predict_raw(raw, current_time=2.0)
    predictor.reset()
    actual = predictor.predict_raw(padded, current_time=2.0)

    assert actual == expected


# --------------------------------------------------------------------------- #
# state tracking
# --------------------------------------------------------------------------- #
def test_state_change_time_tracks_predicted_labels(predictor: Predictor) -> None:
    rising = _ramp_window(up=True)
    falling = _ramp_window(up=False)

    first = predictor.predict_raw(rising, current_time=1.0)
    second = predictor.predict_raw(rising, current_time=2.0)
    assert second.label == first.label
    assert second.state_change_time == 1.0

    changed = predictor.predict_raw(falling, current_time=3.0)
    assert changed.label != first.label
    assert changed.state_change_time == 3.0

    stable = predictor.predict_raw(falling, current_time=4.0)
    assert stable.label == changed.label
    assert stable.state_change_time == 3.0


def test_backwards_time_resets_state_tracking(predictor: Predictor) -> None:
    rising = _ramp_window(up=True)
    predictor.predict_raw(rising, current_time=10.0)

    predictor.predict_raw(rising, current_time=2.0)
    restart = predictor.predict_raw(rising, current_time=3.0)

    assert restart.state_change_time == 2.0


def test_reset_starts_a_new_state(predictor: Predictor) -> None:
    rising = _ramp_window(up=True)
    predictor.predict_raw(rising, current_time=1.0)

    predictor.reset()
    after = predictor.predict_raw(rising, current_time=5.0)

    assert after.state_change_time == 5.0


# --------------------------------------------------------------------------- #
# electrode alignment
# --------------------------------------------------------------------------- #
def test_source_electrodes_are_reordered(bundle: Path) -> None:
    raw_training = _ramp_window(up=True)
    source = tuple(reversed(ELECTRODES))
    raw_source = raw_training[::-1]

    expected = Predictor.from_bundle(
        bundle, encoder=_RampEncoder()
    ).predict_raw(raw_training, current_time=1.0)
    actual = Predictor.from_bundle(
        bundle, electrodes=source, encoder=_RampEncoder()
    ).predict_raw(raw_source, current_time=1.0)

    assert actual == expected


def test_source_electrodes_may_have_extra_channels(bundle: Path) -> None:
    raw_training = _ramp_window(up=True)
    source = ("X",) + tuple(reversed(ELECTRODES))
    raw_source = np.vstack(
        [np.full((1, RAW_SAMPLES), 7.0, dtype=np.float32), raw_training[::-1]]
    )

    expected = Predictor.from_bundle(
        bundle, encoder=_RampEncoder()
    ).predict_raw(raw_training, current_time=1.0)
    actual = Predictor.from_bundle(
        bundle, electrodes=source, encoder=_RampEncoder()
    ).predict_raw(raw_source, current_time=1.0)

    assert actual == expected


def test_missing_source_electrode_is_rejected(bundle: Path) -> None:
    with pytest.raises(ValueError, match="missing training electrode"):
        Predictor.from_bundle(
            bundle,
            electrodes=ELECTRODES[:3],
            encoder=_RampEncoder(),
        )


def test_set_source_electrodes_can_be_reset(predictor: Predictor) -> None:
    source = tuple(reversed(ELECTRODES))
    assert predictor.set_source_electrodes(source) is predictor
    assert predictor.source_electrodes == source

    predictor.set_source_electrodes(None)
    assert predictor.source_electrodes == ELECTRODES


# --------------------------------------------------------------------------- #
# input validation
# --------------------------------------------------------------------------- #
def test_predict_rejects_bad_windows(predictor: Predictor) -> None:
    good = np.zeros((len(ELECTRODES), N_SAMPLES), dtype=np.float32)

    with pytest.raises(ValueError, match="shape"):
        predictor.predict(good[:-1], 1.0)
    with pytest.raises(ValueError, match="shape"):
        predictor.predict(good[:, :-1], 1.0)
    with pytest.raises(ValueError, match="2-D"):
        predictor.predict(good[None, ...], 1.0)
    with pytest.raises(ValueError, match="non-finite"):
        predictor.predict(np.full_like(good, np.nan), 1.0)
    with pytest.raises(ValueError, match="numeric"):
        predictor.predict(good.astype(complex), 1.0)


def test_predict_rejects_bad_times(predictor: Predictor) -> None:
    good = np.zeros((len(ELECTRODES), N_SAMPLES), dtype=np.float32)

    with pytest.raises(ValueError, match="finite"):
        predictor.predict(good, float("nan"))
    with pytest.raises(ValueError, match="finite number"):
        predictor.predict(good, "later")
    with pytest.raises(ValueError, match="finite"):
        predictor.predict_raw(_ramp_window(), float("inf"))


def test_predict_raw_rejects_bad_windows(predictor: Predictor) -> None:
    raw = _ramp_window()

    with pytest.raises(ValueError, match="channels"):
        predictor.predict_raw(raw[:-1], 1.0)
    with pytest.raises(ValueError, match="samples"):
        predictor.predict_raw(raw[:, :100], 1.0)
    with pytest.raises(ValueError, match="2-D"):
        predictor.predict_raw(raw[None, ...], 1.0)
    with pytest.raises(ValueError, match="non-finite"):
        predictor.predict_raw(np.full_like(raw, np.inf), 1.0)


# --------------------------------------------------------------------------- #
# streaming adapter
# --------------------------------------------------------------------------- #
def _feed_stream(processor, *, seconds: float = 8.0, sfreq: float = RAW_SFREQ):
    capacity = round((WINDOW_S + 1.0) * sfreq)
    buffer = np.zeros((len(ELECTRODES), 0), dtype=np.float32)
    total = 0
    chunk = 10
    steps = int(seconds * sfreq / chunk)
    scales = np.arange(1, len(ELECTRODES) + 1, dtype=np.float64)[:, None]

    for _ in range(steps):
        start = total / sfreq
        base = np.linspace(start, start + chunk / sfreq, chunk)
        block = base[None, :] * scales
        buffer = np.concatenate([buffer, block], axis=1)[:, -capacity:]
        total += chunk
        processor(buffer, total / sfreq)
    return buffer


def test_processor_emits_predictions(predictor: Predictor) -> None:
    predictions: list[Prediction] = []
    processor = predictor.processor(on_prediction=predictions.append)

    _feed_stream(processor, seconds=8.0)

    assert len(predictions) >= 7
    assert all(isinstance(prediction, Prediction) for prediction in predictions)
    times = [prediction.current_time for prediction in predictions]
    assert times == sorted(times)
    assert predictions[0].current_time == pytest.approx(1.0)
    assert predictions[0].window_start == pytest.approx(0.0)
    assert predictions[0].state_change_time == pytest.approx(1.0)


def test_processor_aligns_permuted_source(bundle: Path) -> None:
    source = tuple(reversed(ELECTRODES))
    source_predictor = Predictor.from_bundle(
        bundle, electrodes=source, encoder=_RampEncoder()
    )
    expected_predictions: list[Prediction] = []
    actual_predictions: list[Prediction] = []

    default_predictor = Predictor.from_bundle(bundle, encoder=_RampEncoder())
    expected_processor = default_predictor.processor(
        on_prediction=expected_predictions.append
    )
    actual_processor = source_predictor.processor(
        on_prediction=actual_predictions.append
    )

    _feed_stream(expected_processor, seconds=5.0)
    _feed_stream(actual_processor, seconds=5.0)

    assert len(actual_predictions) == len(expected_predictions)
    for actual, expected in zip(actual_predictions, expected_predictions):
        assert actual.label == expected.label
        assert actual.probabilities == pytest.approx(expected.probabilities)
        assert actual.current_time == expected.current_time


# --------------------------------------------------------------------------- #
# warmup and Hugging Face loading
# --------------------------------------------------------------------------- #
def test_warmup_loads_the_encoder(bundle: Path) -> None:
    encoder = _RampEncoder()
    predictor = Predictor.from_bundle(bundle, encoder=encoder)

    assert encoder.load_calls == 0
    assert predictor.warmup() is predictor
    assert encoder.load_calls == 1


def test_from_pretrained_forwards_download_options(
    bundle: Path, tmp_path: Path, monkeypatch
) -> None:
    captured: dict = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(bundle)

    monkeypatch.setattr(
        inference_loader, "snapshot_download", fake_snapshot_download
    )

    predictor = Predictor.from_pretrained(
        "owner/model",
        revision="v1",
        token="secret",
        cache_dir=tmp_path / "cache",
        encoder=_RampEncoder(),
    )

    assert predictor.model_version == "1.2.3"
    assert captured["repo_id"] == "owner/model"
    assert captured["revision"] == "v1"
    assert captured["token"] == "secret"
    assert captured["cache_dir"] == str(tmp_path / "cache")
    assert captured["allow_patterns"] == [
        "model.joblib",
        "metadata.json",
        "model_card.md",
    ]
    assert captured["local_files_only"] is False


def test_from_pretrained_requires_a_repo_id() -> None:
    with pytest.raises(ValueError, match="repo_id"):
        Predictor.from_pretrained("", encoder=_RampEncoder())


# --------------------------------------------------------------------------- #
# encoder revision pinning
# --------------------------------------------------------------------------- #
def test_reve_encoder_stores_the_revision() -> None:
    encoder = ReveEncoder(ELECTRODES, revision=" 2026-11 ")
    assert encoder.revision == "2026-11"
    assert ReveEncoder(ELECTRODES).revision is None

    with pytest.raises(ValueError, match="revision"):
        ReveEncoder(ELECTRODES, revision="  ")


def test_embedding_cache_invalidates_on_revision(tmp_path: Path) -> None:
    X = np.ones((2, len(ELECTRODES), 8), dtype=np.float32)
    cache = tmp_path / "embeddings"

    first = _RampEncoder()
    first.revision = "rev-a"
    load_or_encode(cache, first, X)
    first_key = json.loads((cache / "embeddings.json").read_text())["fingerprint"]

    second = _RampEncoder()
    second.revision = "rev-b"
    load_or_encode(cache, second, X)
    second_key = json.loads((cache / "embeddings.json").read_text())["fingerprint"]

    assert first_key != second_key


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def test_cli_requires_a_source() -> None:
    from hotb_2026.inference import __main__ as cli

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])
