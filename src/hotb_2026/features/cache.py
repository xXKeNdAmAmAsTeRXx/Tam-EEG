"""On-disk cache for frozen REVE embeddings.

Embeddings are deterministic once the model, the electrode order and the input
windows are fixed, so they can be computed once and reused across Optuna
trials and cross-validation folds.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

CACHE_VERSION = 1


def fingerprint(
    X: np.ndarray,
    electrodes: tuple[str, ...] | list[str],
    model_name: str,
    extra: dict[str, Any] | None = None,
) -> str:
    """Stable hash of the arrays and settings that determine the embeddings."""
    digest = hashlib.sha256()
    digest.update(f"v{CACHE_VERSION}\0".encode())
    digest.update(model_name.encode())
    digest.update(b"\0")
    digest.update("\0".join(electrodes).encode())
    digest.update(b"\0")
    if extra:
        digest.update(
            json.dumps(extra, sort_keys=True, default=str).encode()
        )
    array = np.ascontiguousarray(X, dtype=np.float32)
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _paths(path: str | os.PathLike[str]) -> tuple[Path, Path]:
    path = Path(path)
    if path.suffix == ".npy":
        return path, path.with_suffix(".json")
    return path / "embeddings.npy", path / "embeddings.json"


def load_or_encode(
    path: str | os.PathLike[str],
    encoder: Any,
    X: np.ndarray,
    *,
    model_name: str | None = None,
    extra: dict[str, Any] | None = None,
) -> np.ndarray:
    """Load cached embeddings or compute and store them.

    ``path`` may be a directory (``embeddings.npy`` + ``embeddings.json``) or a
    ``.npy`` file with a ``.json`` sidecar.
    """
    npy_path, meta_path = _paths(path)
    model_name = model_name or getattr(encoder, "model_name", "unknown")
    extra = dict(extra or {})
    pos_model_name = getattr(encoder, "pos_model_name", None)
    if pos_model_name:
        extra.setdefault("pos_model_name", pos_model_name)
    # A pinned revision is part of the encoder identity even when the model id
    # is unchanged (for example ``main`` moving forward).
    extra.setdefault("revision", getattr(encoder, "revision", None))
    key = fingerprint(X, tuple(encoder.electrodes), model_name, extra)

    if npy_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            cached_key = meta.get("fingerprint")
            if cached_key == key:
                embeddings = np.load(npy_path)
                if embeddings.shape[0] == X.shape[0]:
                    log.info("Loaded REVE embeddings from %s", npy_path)
                    return np.ascontiguousarray(embeddings, dtype=np.float32)
            else:
                log.info("Embedding cache is stale; recomputing %s", npy_path)
        except (OSError, ValueError, json.JSONDecodeError):
            log.warning("Could not read embedding cache at %s", npy_path)

    log.info("Encoding %d windows with REVE", X.shape[0])
    embeddings = np.asarray(encoder.encode(X), dtype=np.float32)

    npy_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = npy_path.with_suffix(".tmp.npy")
    np.save(tmp, embeddings)
    os.replace(tmp, npy_path)

    meta = {
        "version": CACHE_VERSION,
        "fingerprint": key,
        "model_name": model_name,
        "electrodes": list(encoder.electrodes),
        "shape": list(embeddings.shape),
        "embedding_dim": int(embeddings.shape[1]),
    }
    if extra:
        meta["extra"] = extra
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return embeddings


__all__ = ["CACHE_VERSION", "fingerprint", "load_or_encode"]
