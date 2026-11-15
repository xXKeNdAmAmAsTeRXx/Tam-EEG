from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from hotb_2026.classification.dataset import EegDataset
from hotb_2026.classification.training import TrainingConfig, train
from hotb_2026.model_store import (
    ARTIFACT_SCHEMA_VERSION,
    BundleInfo,
    StoreConfig,
    build_metadata,
    package_versions,
    publish,
    read_metadata,
    save_model,
    validate_result,
    validate_version,
)
from hotb_2026.model_store.__main__ import main
from hotb_2026.model_store.bundle import (
    METADATA_FILE,
    MODEL_CARD_FILE,
    MODEL_FILE,
)
from hotb_2026.model_store.metadata import git_commit, jsonable

PREPROCESSING = {
    "sfreq": 250.0,
    "notch_freq": 50.0,
    "notch_q": 30.0,
    "bandpass": (0.5, 45.0),
    "filter_order": 4,
    "target_sfreq": 200.0,
    "zscore": True,
    "clip": 15.0,
}


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
        preprocessing=dict(PREPROCESSING),
    )


@pytest.fixture(scope="module")
def result() -> dict:
    dataset = make_dataset()
    return train(
        dataset,
        config=TrainingConfig(n_splits=3, n_trials=3, seed=0),
        encoder=FakeEncoder(dataset.electrodes),
    )


@pytest.fixture()
def bundle(result: dict, tmp_path: Path) -> BundleInfo:
    return save_model(result, tmp_path / "bundle")


class FakeApi:
    """Offline stand-in for ``huggingface_hub.HfApi``."""

    def __init__(
        self,
        *,
        username: str = "tester",
        create_errors: list[Exception] | None = None,
        upload_errors: list[Exception] | None = None,
    ) -> None:
        self.username = username
        self.create_errors = list(create_errors or [])
        self.upload_errors = list(upload_errors or [])
        self.calls: list[tuple[str, object]] = []
        self.created: list[tuple[str, bool]] = []
        self.commit_message: str | None = None

    def create_repo(self, *, repo_id, repo_type, exist_ok, private):
        self.calls.append(("create_repo", repo_id))
        if self.create_errors:
            raise self.create_errors.pop(0)
        self.created.append((repo_id, private))

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs["repo_id"]))
        if self.upload_errors:
            raise self.upload_errors.pop(0)
        self.commit_message = kwargs["commit_message"]

        class Commit:
            oid = "abc123"

        return Commit()

    def create_tag(self, *, repo_id, tag, repo_type):
        self.calls.append(("create_tag", tag))

    def whoami(self):
        return {"name": self.username}


class AccessError(RuntimeError):
    """Mimics a Hub permission error."""


# --------------------------------------------------------------------------- #
# bundle contents
# --------------------------------------------------------------------------- #
def test_save_model_writes_bundle(bundle: BundleInfo) -> None:
    assert (bundle.directory / MODEL_FILE).is_file()
    assert (bundle.directory / METADATA_FILE).is_file()
    assert (bundle.directory / MODEL_CARD_FILE).is_file()
    assert bundle.version == "0.1.0"
    assert {path.name for path in bundle.files} == {
        MODEL_FILE,
        METADATA_FILE,
        MODEL_CARD_FILE,
    }


def test_saved_model_roundtrips_predictions(
    result: dict, bundle: BundleInfo
) -> None:
    loaded = joblib.load(bundle.model_path)
    X = result["embeddings"]

    np.testing.assert_array_equal(
        loaded.predict(X), result["model"].predict(X)
    )
    np.testing.assert_allclose(
        loaded.predict_proba(X), result["model"].predict_proba(X)
    )


def test_saved_metadata_is_json_and_self_describing(
    result: dict, bundle: BundleInfo
) -> None:
    on_disk = json.loads(bundle.metadata_path.read_text())
    assert on_disk == bundle.metadata
    assert read_metadata(bundle.directory) == on_disk

    metadata = bundle.metadata
    assert metadata["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert metadata["model_version"] == "0.1.0"
    assert metadata["labels"] == {
        "names": ["low", "medium", "high"],
        "classes": [0, 1, 2],
        "n_classes": 3,
    }

    contract = metadata["input"]
    assert contract["electrodes"] == ["E0", "E1", "E2", "E3"]
    assert contract["window_s"] == pytest.approx(1.0)
    assert contract["n_samples"] == 64
    assert contract["sfreq"] == pytest.approx(200.0)
    assert contract["raw_sfreq"] == pytest.approx(250.0)
    assert contract["time_units"] == "s"
    assert contract["label_rule"] == "state active at window start"

    encoder = metadata["encoder"]
    assert encoder["model_name"] == "fake-reve"
    assert encoder["pos_model_name"] == "fake-positions"
    assert encoder["patch_size"] == 64
    assert encoder["revision"] == "test-revision"
    assert encoder["embedding_dim"] == 4

    # JSON has no tuples, so the band-pass edges are stored as a list.
    assert metadata["preprocessing"]["notch_freq"] == 50.0
    assert metadata["preprocessing"]["bandpass"] == [0.5, 45.0]
    assert metadata["training"]["best_params"]["C"] > 0
    assert metadata["training"]["dataset"]["n_windows"] == 36
    assert metadata["model"]["steps"] == [
        {"name": "scaler", "type": "StandardScaler"},
        {"name": "svc", "type": "CalibratedClassifierCV"},
    ]
    assert metadata["model"]["n_features"] == result["embeddings"].shape[1]


def test_model_card_documents_the_contract(bundle: BundleInfo) -> None:
    card = bundle.model_card_path.read_text()
    assert card.startswith("---\n")
    assert "library_name: scikit-learn" in card
    assert "low, medium, high" in card
    assert "notch_freq" in card
    assert "0.1.0" in card
    assert "joblib.load" in card


def test_save_model_copies_run_artifacts(
    result: dict, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text("{}")
    (run_dir / "report.md").write_text("# Report")
    (run_dir / "unrelated.txt").write_text("not copied")
    extra = tmp_path / "confusion.png"
    extra.write_bytes(b"png")

    bundle = save_model(
        result, tmp_path / "bundle", run_dir=run_dir, extra_files=[extra]
    )
    assert (bundle.directory / "metrics.json").read_text() == "{}"
    assert (bundle.directory / "report.md").read_text() == "# Report"
    assert (bundle.directory / "confusion.png").read_bytes() == b"png"
    assert not (bundle.directory / "unrelated.txt").exists()


def test_save_model_reuses_existing_directory(
    result: dict, tmp_path: Path
) -> None:
    directory = tmp_path / "bundle"
    directory.mkdir()
    (directory / "notes.txt").write_text("keep me")

    bundle = save_model(result, directory)
    assert (directory / "notes.txt").read_text() == "keep me"
    assert bundle.version == "0.1.0"


def test_save_model_uses_configured_version(result: dict, tmp_path: Path) -> None:
    bundle = save_model(
        result,
        tmp_path / "bundle",
        config=StoreConfig(version="1.2.3"),
    )
    assert bundle.version == "1.2.3"
    assert read_metadata(bundle.directory)["model_version"] == "1.2.3"


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def test_validate_result_requires_a_dict() -> None:
    with pytest.raises(TypeError, match="result must be the dict"):
        validate_result([])


def test_validate_result_requires_a_model() -> None:
    with pytest.raises(ValueError, match="no 'model'"):
        validate_result({"label_names": ["low", "medium", "high"]})


def test_validate_result_rejects_unfitted_model() -> None:
    unfitted = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
    with pytest.raises(ValueError, match="not fitted"):
        validate_result(
            {"model": unfitted, "label_names": ["low", "medium", "high"]}
        )


def test_validate_result_rejects_class_count_mismatch(result: dict) -> None:
    broken = dict(result, label_names=["low", "medium"])
    with pytest.raises(ValueError, match="label_names has 2"):
        validate_result(broken)


def test_validate_result_rejects_non_contiguous_classes(result: dict) -> None:
    X = result["embeddings"]
    y = np.tile(np.array([1, 2, 3]), X.shape[0] // 3 + 1)[: X.shape[0]]
    model = CalibratedClassifierCV(
        SVC(), method="sigmoid", ensemble=False
    ).fit(X, y)

    with pytest.raises(ValueError, match=r"integers 0\.\."):
        validate_result({"model": model, "label_names": ["a", "b", "c"]})


def test_validate_result_rejects_embedding_dimension_mismatch(
    result: dict,
) -> None:
    broken = dict(result, embedding_meta={"embedding_dim": 999})
    with pytest.raises(ValueError, match="does not match"):
        validate_result(broken)


def test_save_model_does_not_write_invalid_result(tmp_path: Path) -> None:
    destination = tmp_path / "bundle"
    with pytest.raises(ValueError):
        save_model(
            {"label_names": ["low", "medium", "high"]},
            destination,
        )
    assert not destination.exists()


def test_save_model_rejects_missing_run_dir(result: dict, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run directory not found"):
        save_model(result, tmp_path / "bundle", run_dir=tmp_path / "missing")


def test_save_model_rejects_missing_extra_file(
    result: dict, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError, match="extra file not found"):
        save_model(
            result,
            tmp_path / "bundle",
            extra_files=[tmp_path / "missing.txt"],
        )


def test_read_metadata_rejects_invalid_json(tmp_path: Path) -> None:
    (tmp_path / METADATA_FILE).write_text("{")
    with pytest.raises(ValueError, match="not valid JSON"):
        read_metadata(tmp_path)


def test_read_metadata_requires_model_version(tmp_path: Path) -> None:
    (tmp_path / METADATA_FILE).write_text(
        json.dumps({"artifact_schema_version": ARTIFACT_SCHEMA_VERSION})
    )
    with pytest.raises(ValueError, match="no model_version"):
        read_metadata(tmp_path)


def test_build_metadata_rejects_invalid_version(result: dict) -> None:
    with pytest.raises(ValueError, match="version must be"):
        build_metadata(result, version="not a version")


# --------------------------------------------------------------------------- #
# StoreConfig
# --------------------------------------------------------------------------- #
def test_store_config_resolves_environment(monkeypatch) -> None:
    monkeypatch.setenv("HOTB_MODEL_REPO_ID", "ai-med/nback")
    monkeypatch.setenv("HF_TOKEN", "secret")
    config = StoreConfig()
    assert config.resolved_repo_id() == "ai-med/nback"
    assert config.resolved_token() == "secret"


def test_store_config_requires_repo_id(monkeypatch) -> None:
    monkeypatch.delenv("HOTB_MODEL_REPO_ID", raising=False)
    with pytest.raises(ValueError, match="no Hugging Face repo id"):
        StoreConfig().resolved_repo_id()


@pytest.mark.parametrize(
    "version", ["", "1 0", "-1.0", "1/2", None]
)
def test_store_config_rejects_invalid_versions(version) -> None:
    with pytest.raises(ValueError, match="version must be"):
        StoreConfig(version=version)


@pytest.mark.parametrize(
    "repo_id", ["model", "owner/", "/model", "owner/name/extra"]
)
def test_store_config_rejects_invalid_repo_ids(repo_id: str) -> None:
    with pytest.raises(ValueError, match="repo_id must look like"):
        StoreConfig(repo_id=repo_id)


def test_store_config_commit_message() -> None:
    assert StoreConfig().resolved_commit_message() == "add model version 0.1.0"
    assert (
        StoreConfig(commit_message="custom").resolved_commit_message("2.0.0")
        == "custom"
    )


# --------------------------------------------------------------------------- #
# publish
# --------------------------------------------------------------------------- #
def test_publish_uploads_bundle(bundle: BundleInfo) -> None:
    api = FakeApi()
    summary = publish(
        bundle.directory,
        StoreConfig(repo_id="ai-med/nback"),
        api=api,
    )

    assert summary == {
        "repo_id": "ai-med/nback",
        "repo_url": "https://huggingface.co/ai-med/nback",
        "version": "0.1.0",
        "tag": "0.1.0",
        "commit": "abc123",
    }
    assert api.created == [("ai-med/nback", False)]
    assert ("upload_folder", "ai-med/nback") in api.calls
    assert ("create_tag", "0.1.0") in api.calls
    assert api.commit_message == "add model version 0.1.0"


def test_publish_uses_bundle_version_not_config_version(
    bundle: BundleInfo,
) -> None:
    api = FakeApi()
    summary = publish(
        bundle.directory,
        StoreConfig(repo_id="ai-med/nback", version="9.9.9"),
        api=api,
    )
    assert summary["version"] == "0.1.0"
    assert summary["tag"] == "0.1.0"
    assert api.commit_message == "add model version 0.1.0"


def test_publish_can_skip_the_tag(bundle: BundleInfo) -> None:
    api = FakeApi()
    summary = publish(
        bundle.directory,
        StoreConfig(repo_id="ai-med/nback"),
        api=api,
        tag=False,
    )
    assert summary["tag"] is None
    assert all(name != "create_tag" for name, _ in api.calls)


def test_publish_tolerates_missing_tag_support(bundle: BundleInfo) -> None:
    class NoTagApi(FakeApi):
        create_tag = None

    summary = publish(
        bundle.directory,
        StoreConfig(repo_id="ai-med/nback"),
        api=NoTagApi(),
    )
    assert summary["tag"] is None
    assert summary["commit"] == "abc123"


def test_publish_falls_back_to_the_authenticated_user(
    bundle: BundleInfo,
) -> None:
    api = FakeApi(username="tester", create_errors=[AccessError("403 Forbidden")])
    summary = publish(
        bundle.directory,
        StoreConfig(repo_id="ai-med/nback"),
        api=api,
    )

    assert summary["repo_id"] == "tester/nback"
    assert summary["repo_url"] == "https://huggingface.co/tester/nback"
    assert api.created == [("tester/nback", False)]


def test_publish_falls_back_when_the_upload_is_forbidden(
    bundle: BundleInfo,
) -> None:
    api = FakeApi(username="tester", upload_errors=[AccessError("403 Forbidden")])
    summary = publish(
        bundle.directory,
        StoreConfig(repo_id="ai-med/nback"),
        api=api,
    )

    assert summary["repo_id"] == "tester/nback"
    uploads = [call for call in api.calls if call[0] == "upload_folder"]
    assert uploads == [
        ("upload_folder", "ai-med/nback"),
        ("upload_folder", "tester/nback"),
    ]


def test_publish_without_fallback_propagates_the_error(
    bundle: BundleInfo,
) -> None:
    api = FakeApi(create_errors=[AccessError("403 Forbidden")])
    with pytest.raises(AccessError):
        publish(
            bundle.directory,
            StoreConfig(repo_id="ai-med/nback"),
            api=api,
            fallback_to_user=False,
        )


def test_publish_does_not_fall_back_to_the_same_owner(
    bundle: BundleInfo,
) -> None:
    api = FakeApi(username="ai-med", create_errors=[AccessError("403")])
    with pytest.raises(AccessError):
        publish(
            bundle.directory,
            StoreConfig(repo_id="ai-med/nback"),
            api=api,
        )


def test_publish_propagates_unrelated_errors(bundle: BundleInfo) -> None:
    api = FakeApi(create_errors=[RuntimeError("server exploded")])
    with pytest.raises(RuntimeError, match="server exploded"):
        publish(
            bundle.directory,
            StoreConfig(repo_id="ai-med/nback"),
            api=api,
        )


def test_publish_rejects_incomplete_bundle(tmp_path: Path) -> None:
    directory = tmp_path / "bundle"
    directory.mkdir()
    (directory / MODEL_FILE).write_bytes(b"model")

    with pytest.raises(FileNotFoundError, match="metadata.json, model_card.md"):
        publish(directory, StoreConfig(repo_id="ai-med/nback"), api=FakeApi())


def test_publish_requires_a_repo_id(
    bundle: BundleInfo, monkeypatch
) -> None:
    monkeypatch.delenv("HOTB_MODEL_REPO_ID", raising=False)
    with pytest.raises(ValueError, match="no Hugging Face repo id"):
        publish(bundle.directory, StoreConfig(), api=FakeApi())


def test_publish_rejects_unknown_artifact_schema(bundle: BundleInfo) -> None:
    metadata = json.loads(bundle.metadata_path.read_text())
    metadata["artifact_schema_version"] = 99
    bundle.metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="artifact schema"):
        publish(
            bundle.directory,
            StoreConfig(repo_id="ai-med/nback"),
            api=FakeApi(),
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_show_prints_metadata(bundle: BundleInfo, capsys) -> None:
    assert main(["show", str(bundle.directory)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["model_version"] == "0.1.0"
    assert printed["labels"]["classes"] == [0, 1, 2]


def test_cli_publish_passes_options(bundle: BundleInfo, monkeypatch, capsys) -> None:
    from hotb_2026.model_store import __main__ as cli

    captured: dict = {}

    def fake_publish(bundle_dir, config, **kwargs):
        captured["bundle"] = Path(bundle_dir)
        captured["config"] = config
        captured["kwargs"] = kwargs
        return {
            "repo_id": "ai-med/nback",
            "repo_url": "https://huggingface.co/ai-med/nback",
            "version": "0.1.0",
            "tag": None,
            "commit": None,
        }

    monkeypatch.setattr(cli, "publish", fake_publish)
    code = cli.main(
        [
            "publish",
            str(bundle.directory),
            "--repo-id",
            "ai-med/nback",
            "--private",
            "--no-tag",
            "--no-fallback",
        ]
    )

    assert code == 0
    assert captured["bundle"] == bundle.directory
    assert captured["config"].repo_id == "ai-med/nback"
    assert captured["config"].private is True
    assert captured["kwargs"] == {"fallback_to_user": False, "tag": False}
    assert json.loads(capsys.readouterr().out)["repo_id"] == "ai-med/nback"


def test_cli_show_requires_metadata(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no metadata.json"):
        main(["show", str(tmp_path)])


# --------------------------------------------------------------------------- #
# metadata helpers
# --------------------------------------------------------------------------- #
@dataclass
class _Payload:
    path: Path
    values: np.ndarray
    score: np.float32


def test_jsonable_converts_numpy_dataclass_and_paths(tmp_path: Path) -> None:
    payload = _Payload(
        path=tmp_path / "artifact",
        values=np.array([1.5, 2.5], dtype=np.float32),
        score=np.float32(0.25),
    )
    converted = jsonable(payload)

    assert converted == {
        "path": str(tmp_path / "artifact"),
        "values": [1.5, 2.5],
        "score": 0.25,
    }
    # The result must be ready for ``json.dumps`` without a custom encoder.
    json.dumps(converted)


def test_package_versions_records_the_project_and_key_dependencies() -> None:
    versions = package_versions()
    assert "hotb-2026" in versions
    assert "scikit-learn" in versions
    assert "joblib" in versions


def test_git_commit_returns_the_current_revision() -> None:
    commit = git_commit()
    if commit is None:
        pytest.skip("not running inside a git checkout")
    assert len(commit) == 40
    assert all(char in "0123456789abcdef" for char in commit)


def test_git_commit_returns_none_outside_a_repository(tmp_path: Path) -> None:
    assert git_commit(tmp_path) is None


def test_build_metadata_accepts_a_directory_for_git_lookup(
    result: dict, tmp_path: Path
) -> None:
    metadata = build_metadata(result, version="1.0.0", directory=tmp_path)

    assert metadata["model_version"] == "1.0.0"
    assert metadata["git_commit"] is None


# --------------------------------------------------------------------------- #
# config helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "version", ["0.1.0", "v1", "2026.11.15", "exp_1-x"]
)
def test_validate_version_accepts_hub_safe_strings(version: str) -> None:
    assert validate_version(version) == version


def test_store_config_token_precedence(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "env-token")
    assert StoreConfig().resolved_token() == "env-token"
    assert StoreConfig(token="explicit").resolved_token() == "explicit"

    monkeypatch.delenv("HF_TOKEN")
    assert StoreConfig().resolved_token() is None


# --------------------------------------------------------------------------- #
# bundle hygiene
# --------------------------------------------------------------------------- #
def test_bundle_files_exclude_subdirectories(bundle: BundleInfo) -> None:
    (bundle.directory / "nested").mkdir()

    assert {path.name for path in bundle.files} == {
        MODEL_FILE,
        METADATA_FILE,
        MODEL_CARD_FILE,
    }


def test_save_model_leaves_no_temporary_files(bundle: BundleInfo) -> None:
    assert not list(bundle.directory.glob("*.tmp"))
