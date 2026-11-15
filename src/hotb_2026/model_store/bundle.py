"""Save a fitted REVE + SVM pipeline as a deployable bundle."""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib

from .config import StoreConfig
from .metadata import (
    ARTIFACT_SCHEMA_VERSION,
    build_metadata,
    jsonable,
    render_model_card,
)

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
MODEL_CARD_FILE = "model_card.md"

#: Files copied from a run directory when it is provided.
RUN_ARTIFACTS = ("metrics.json", "report.md", "predictions.npz")


@dataclass(frozen=True)
class BundleInfo:
    """Paths and metadata of a saved bundle."""

    directory: Path
    model_path: Path
    metadata_path: Path
    model_card_path: Path
    metadata: dict[str, Any]

    @property
    def version(self) -> str:
        return str(self.metadata["model_version"])

    @property
    def files(self) -> list[Path]:
        return sorted(path for path in self.directory.iterdir() if path.is_file())


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_dump_model(model: Any, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        joblib.dump(model, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _copy_run_artifacts(run_dir: Path, destination: Path) -> None:
    for name in RUN_ARTIFACTS:
        source = run_dir / name
        if source.is_file():
            shutil.copy2(source, destination / name)


def read_metadata(directory: str | Path) -> dict[str, Any]:
    """Read and validate ``metadata.json`` from a bundle directory."""
    path = Path(directory) / METADATA_FILE
    if not path.is_file():
        raise FileNotFoundError(f"no {METADATA_FILE} in {directory}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"{path} must contain a JSON object, got {type(payload).__name__}"
        )

    schema = payload.get("artifact_schema_version")
    if schema != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"{path} uses artifact schema {schema!r}; this package supports "
            f"{ARTIFACT_SCHEMA_VERSION}"
        )
    if not payload.get("model_version"):
        raise ValueError(f"{path} has no model_version")
    return payload


def save_model(
    result: dict[str, Any],
    out_dir: str | Path,
    *,
    config: StoreConfig | None = None,
    run_dir: str | Path | None = None,
    extra_files: Iterable[str | Path] | None = None,
) -> BundleInfo:
    """Write ``result`` to ``out_dir`` as a deployable model bundle.

    Parameters
    ----------
    result:
        Dict returned by ``hotb_2026.classification.training.train``. It must
        contain a fitted ``model`` and ``label_names``.
    out_dir:
        Destination directory. It is created when missing and reused when it
        already exists. Existing unrelated files are left untouched.
    config:
        Version and publishing settings. Only ``version`` affects saving.
    run_dir:
        Optional training run directory. ``metrics.json``, ``report.md`` and
        ``predictions.npz`` are copied from it when present.
    extra_files:
        Additional files to copy into the bundle.

    Returns
    -------
    :class:`BundleInfo` with the written paths and the metadata that was
    stored.

    Raises
    ------
    TypeError, ValueError
        When ``result`` does not contain a fitted, self-consistent pipeline.
    FileNotFoundError
        When ``run_dir`` or one of ``extra_files`` does not exist.
    """
    config = config or StoreConfig()
    directory = Path(out_dir)

    # Validate every input before writing anything, so a bad call never leaves
    # a half-written bundle behind.
    metadata = build_metadata(result, version=config.version)
    model = result["model"]

    if run_dir is not None:
        run_path = Path(run_dir)
        if not run_path.is_dir():
            raise FileNotFoundError(f"run directory not found: {run_path}")

    extra_paths = [Path(path) for path in extra_files or ()]
    for path in extra_paths:
        if not path.is_file():
            raise FileNotFoundError(f"extra file not found: {path}")

    # Render everything before writing so a serialization error cannot leave a
    # new model next to stale metadata.
    metadata_text = json.dumps(jsonable(metadata), indent=2, sort_keys=True) + "\n"
    card_text = render_model_card(metadata) + "\n"

    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / MODEL_FILE
    metadata_path = directory / METADATA_FILE
    model_card_path = directory / MODEL_CARD_FILE

    _atomic_dump_model(model, model_path)
    _atomic_write_text(metadata_path, metadata_text)
    _atomic_write_text(model_card_path, card_text)

    if run_dir is not None:
        _copy_run_artifacts(Path(run_dir), directory)
    for path in extra_paths:
        shutil.copy2(path, directory / path.name)

    return BundleInfo(
        directory=directory,
        model_path=model_path,
        metadata_path=metadata_path,
        model_card_path=model_card_path,
        metadata=metadata,
    )


__all__ = [
    "BundleInfo",
    "METADATA_FILE",
    "MODEL_CARD_FILE",
    "MODEL_FILE",
    "RUN_ARTIFACTS",
    "read_metadata",
    "save_model",
]
