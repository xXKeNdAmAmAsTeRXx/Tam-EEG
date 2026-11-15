"""Load and validate a bundle written by :mod:`hotb_2026.model_store`.

``metadata.json`` is the contract between training and inference. The loader
checks it against the serialized estimator before any signal is processed, so a
stale or mismatched bundle fails at startup instead of producing quiet, wrong
predictions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from huggingface_hub import snapshot_download

from hotb_2026.features.reve import electrode_names
from hotb_2026.model_store import read_metadata
from hotb_2026.model_store.bundle import (
    METADATA_FILE,
    MODEL_CARD_FILE,
    MODEL_FILE,
)
from hotb_2026.preprocessing import PreprocessingConfig, Preprocessor

#: Files needed to run a bundle. Model cards and training reports are optional.
BUNDLE_FILES = (MODEL_FILE, METADATA_FILE, MODEL_CARD_FILE)


@dataclass(frozen=True)
class BundleContract:
    """Validated, runtime view of the contract stored in ``metadata.json``."""

    directory: Path
    model_version: str
    electrodes: tuple[str, ...]
    window_s: float
    sfreq: float
    raw_sfreq: float
    n_samples: int
    label_names: tuple[str, ...]
    classes: tuple[int, ...]
    encoder_model_name: str
    encoder_pos_model_name: str
    encoder_revision: str | None
    embedding_dim: int
    patch_size: int | None
    preprocessing: PreprocessingConfig
    metadata: dict[str, Any] = field(repr=False, compare=False)

    @property
    def raw_samples(self) -> int:
        """Samples in one raw window at :attr:`raw_sfreq`."""
        return max(1, round(self.window_s * self.raw_sfreq))

    @property
    def n_channels(self) -> int:
        return len(self.electrodes)


@dataclass(frozen=True)
class LoadedBundle:
    """A fitted estimator and its validated contract."""

    model: Any = field(repr=False)
    contract: BundleContract


def _mapping(metadata: dict[str, Any], key: str) -> dict[str, Any]:
    value = metadata.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"metadata.json is missing the {key!r} section")
    return value


def _name(section: dict[str, Any], key: str, where: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"metadata.json {where}.{key} must be a non-empty string, "
            f"got {value!r}"
        )
    return value.strip()


def _positive_float(
    section: dict[str, Any], key: str, where: str
) -> float:
    value = section.get(key)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"metadata.json {where}.{key} must be a number, got {value!r}"
        ) from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(
            f"metadata.json {where}.{key} must be positive and finite, "
            f"got {value!r}"
        )
    return number


def _positive_int(section: dict[str, Any], key: str, where: str) -> int:
    value = section.get(key)
    if isinstance(value, bool):
        raise ValueError(
            f"metadata.json {where}.{key} must be an integer, got {value!r}"
        )
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"metadata.json {where}.{key} must be an integer, got {value!r}"
        ) from exc
    if number != value or number <= 0:
        raise ValueError(
            f"metadata.json {where}.{key} must be a positive integer, "
            f"got {value!r}"
        )
    return number


def _electrodes(section: dict[str, Any]) -> tuple[str, ...]:
    raw = section.get("electrodes")
    if raw is None:
        raise ValueError("metadata.json input.electrodes is required")
    try:
        names = electrode_names(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"metadata.json input.electrodes is invalid: {exc}"
        ) from exc
    return tuple(names)


def _preprocessing(metadata: dict[str, Any]) -> PreprocessingConfig:
    raw = metadata.get("preprocessing")
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            "metadata.json has no preprocessing snapshot; re-save the model "
            "with hotb_2026.model_store.save_model"
        )

    known = set(PreprocessingConfig.__dataclass_fields__)
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            "metadata.json preprocessing has unknown field(s): "
            f"{', '.join(unknown)}"
        )
    try:
        return PreprocessingConfig(**raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"metadata.json preprocessing is invalid: {exc}"
        ) from exc


def _validate_estimator(
    model: Any, labels: dict[str, Any], encoder: dict[str, Any]
) -> tuple[tuple[int, ...], tuple[str, ...], int]:
    for method in ("predict", "predict_proba"):
        if not callable(getattr(model, method, None)):
            raise TypeError(
                f"the loaded model does not implement {method}(); "
                "was model.joblib written by hotb_2026.model_store?"
            )

    raw_names = labels.get("names")
    if not isinstance(raw_names, (list, tuple)) or not raw_names:
        raise ValueError("metadata.json labels.names must be a non-empty list")
    names = tuple(str(name) for name in raw_names)
    if any(not name for name in names):
        raise ValueError("metadata.json labels.names must not contain empty names")
    if len(set(names)) != len(names):
        raise ValueError("metadata.json labels.names must be unique")

    raw_classes = labels.get("classes")
    if not isinstance(raw_classes, (list, tuple)):
        raise ValueError("metadata.json labels.classes must be a list")
    try:
        classes = tuple(int(value) for value in raw_classes)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"metadata.json labels.classes must contain integers, "
            f"got {raw_classes!r}"
        ) from exc
    expected = tuple(range(len(names)))
    if classes != expected:
        raise ValueError(
            "metadata.json labels.classes must be contiguous from 0, "
            f"got {list(classes)}"
        )

    n_classes = labels.get("n_classes")
    if n_classes is not None and int(n_classes) != len(names):
        raise ValueError(
            f"metadata.json labels.n_classes={n_classes} does not match "
            f"{len(names)} label names"
        )

    model_classes = tuple(
        int(value) for value in np.asarray(getattr(model, "classes_", []))
    )
    if model_classes != classes:
        raise ValueError(
            "the loaded model was fitted on classes "
            f"{list(model_classes)} but metadata.json declares "
            f"{list(classes)}"
        )

    raw_dim = encoder.get("embedding_dim")
    try:
        embedding_dim = int(raw_dim)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "metadata.json encoder.embedding_dim must be a positive integer, "
            f"got {raw_dim!r}"
        ) from exc
    if embedding_dim <= 0:
        raise ValueError(
            "metadata.json encoder.embedding_dim must be a positive integer, "
            f"got {raw_dim!r}"
        )

    n_features = getattr(model, "n_features_in_", None)
    if n_features is None:
        raise ValueError(
            "the loaded model does not expose n_features_in_; cannot verify "
            "it against encoder.embedding_dim"
        )
    if int(n_features) != embedding_dim:
        raise ValueError(
            f"the loaded model expects {int(n_features)} features but "
            f"metadata.json records embedding_dim={embedding_dim}"
        )

    return classes, names, embedding_dim


def _validate_contract(
    directory: Path, metadata: dict[str, Any], model: Any
) -> BundleContract:
    labels = _mapping(metadata, "labels")
    encoder = _mapping(metadata, "encoder")
    inputs = _mapping(metadata, "input")

    classes, names, embedding_dim = _validate_estimator(model, labels, encoder)

    electrodes = _electrodes(inputs)
    window_s = _positive_float(inputs, "window_s", "input")
    sfreq = _positive_float(inputs, "sfreq", "input")
    raw_sfreq = _positive_float(inputs, "raw_sfreq", "input")
    n_samples = _positive_int(inputs, "n_samples", "input")

    time_units = inputs.get("time_units", "s")
    if time_units != "s":
        raise ValueError(
            "metadata.json input.time_units must be 's' for live inference, "
            f"got {time_units!r}"
        )

    model_name = _name(encoder, "model_name", "encoder")
    pos_model_name = _name(encoder, "pos_model_name", "encoder")
    revision = encoder.get("revision")
    if revision is not None and (
        not isinstance(revision, str) or not revision.strip()
    ):
        raise ValueError(
            "metadata.json encoder.revision must be a non-empty string or null, "
            f"got {revision!r}"
        )

    patch_size: int | None = None
    raw_patch = encoder.get("patch_size")
    if raw_patch is not None:
        patch_size = _positive_int(encoder, "patch_size", "encoder")
        if patch_size > n_samples:
            raise ValueError(
                f"metadata.json encoder.patch_size={patch_size} exceeds the "
                f"{n_samples} samples per window"
            )

    preprocessing = _preprocessing(metadata)
    if not math.isclose(preprocessing.sfreq, raw_sfreq):
        raise ValueError(
            "metadata.json preprocessing.sfreq="
            f"{preprocessing.sfreq:g} does not match input.raw_sfreq="
            f"{raw_sfreq:g}"
        )

    preprocessor = Preprocessor(preprocessing)
    if not math.isclose(preprocessor.output_sfreq, sfreq):
        raise ValueError(
            "metadata.json preprocessing produces "
            f"{preprocessor.output_sfreq:g} Hz but input.sfreq is {sfreq:g} Hz"
        )

    # Run the recorded preprocessing on a zero window. This catches a stale
    # snapshot whose filters would silently change the number of samples.
    raw_samples = max(1, round(window_s * raw_sfreq))
    if raw_samples < preprocessor.min_samples:
        raise ValueError(
            f"a {window_s:g} s window at {raw_sfreq:g} Hz has {raw_samples} "
            f"samples, but the recorded preprocessing needs at least "
            f"{preprocessor.min_samples}"
        )
    probe = np.zeros((len(electrodes), raw_samples), dtype=np.float32)
    try:
        output = preprocessor.process(probe)
    except ValueError as exc:
        raise ValueError(
            f"the recorded preprocessing rejects a {window_s:g} s window: {exc}"
        ) from exc
    expected_shape = (len(electrodes), n_samples)
    if output.shape != expected_shape:
        raise ValueError(
            "the recorded preprocessing returns "
            f"{output.shape[1]} samples per window but metadata.json declares "
            f"input.n_samples={n_samples}; the bundle's preprocessing snapshot "
            f"does not match its windows (expected shape {expected_shape}, got "
            f"{output.shape})"
        )

    return BundleContract(
        directory=directory,
        model_version=str(metadata["model_version"]),
        electrodes=electrodes,
        window_s=window_s,
        sfreq=sfreq,
        raw_sfreq=raw_sfreq,
        n_samples=n_samples,
        label_names=names,
        classes=classes,
        encoder_model_name=model_name,
        encoder_pos_model_name=pos_model_name,
        encoder_revision=revision.strip() if isinstance(revision, str) else None,
        embedding_dim=embedding_dim,
        patch_size=patch_size,
        preprocessing=preprocessing,
        metadata=metadata,
    )


def load_bundle(directory: str | Path) -> LoadedBundle:
    """Load and validate a local bundle directory.

    Parameters
    ----------
    directory:
        Folder containing ``model.joblib`` and ``metadata.json``.

    Raises
    ------
    FileNotFoundError
        When the directory, model or metadata file is missing.
    ValueError
        When ``metadata.json`` is unsupported or inconsistent with the model.
    TypeError
        When the serialized object is not a classifier.

    Notes
    -----
    ``joblib.load`` can execute arbitrary code. Only load bundles you trust.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"bundle directory not found: {directory}")

    metadata = read_metadata(directory)
    model_path = directory / MODEL_FILE
    if not model_path.is_file():
        raise FileNotFoundError(
            f"no {MODEL_FILE} in {directory}; expected a complete bundle"
        )

    try:
        model = joblib.load(model_path)
    except Exception as exc:  # noqa: BLE001 - add the bundle path to the error
        raise ValueError(f"could not load {model_path}: {exc}") from exc

    contract = _validate_contract(directory, metadata, model)
    return LoadedBundle(model=model, contract=contract)


def load_pretrained_bundle(
    repo_id: str,
    *,
    revision: str | None = None,
    token: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> LoadedBundle:
    """Download a bundle from the Hugging Face Hub and validate it.

    ``revision`` pins a tag, branch or commit. Only the three bundle files are
    downloaded; training reports and figures are skipped.
    """
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("repo_id must be a non-empty 'owner/name' string")
    downloaded = snapshot_download(
        repo_id=repo_id.strip(),
        revision=revision,
        token=token,
        cache_dir=None if cache_dir is None else str(cache_dir),
        allow_patterns=list(BUNDLE_FILES),
        local_files_only=local_files_only,
    )
    return load_bundle(downloaded)


__all__ = [
    "BUNDLE_FILES",
    "BundleContract",
    "LoadedBundle",
    "load_bundle",
    "load_pretrained_bundle",
]
