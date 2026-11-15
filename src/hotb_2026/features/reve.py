"""Frozen REVE encoder shared by training and live inference.

The encoder wraps two pretrained Hugging Face models:

* ``brain-bzh/reve-positions`` maps electrode names to 3D scalp coordinates;
* ``brain-bzh/reve-base`` turns a window into attention-pooled embeddings.

Both models are frozen. Loading is lazy and cached per process so importing
this package (or ``hotb_2026.classification``) does not require ``torch``.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "brain-bzh/reve-base"
DEFAULT_POS_MODEL_NAME = "brain-bzh/reve-positions"

# (signal model name, position model name, revision) -> models and shape
_MODEL_CACHE: dict[
    tuple[str, str, str | None], tuple[Any, Any, int, int]
] = {}
_MODEL_LOCK = threading.Lock()


def electrode_names(electrodes: Any) -> list[str]:
    """Normalize a montage or an iterable of names into a validated list."""
    if hasattr(electrodes, "eeg_names"):
        names = list(electrodes.eeg_names)
    elif hasattr(electrodes, "names"):
        names = list(electrodes.names)
    else:
        names = list(electrodes)

    if not names:
        raise ValueError("electrodes must not be empty")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("electrode names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("electrode names must be unique")
    return names


def resolve_device(device: str | None) -> str:
    """Resolve ``None``/``"auto"`` to the best available torch device."""
    import torch

    if device not in (None, "auto"):
        return device
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _load_models(
    model_name: str,
    pos_model_name: str,
    revision: str | None = None,
) -> tuple[Any, Any, int, int]:
    key = (model_name, pos_model_name, revision)
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached

        from transformers import AutoModel

        log.info(
            "Loading REVE position model %s (revision %s)",
            pos_model_name,
            revision or "default",
        )
        pos_model = AutoModel.from_pretrained(
            pos_model_name, trust_remote_code=True, revision=revision
        )
        pos_model.eval()

        log.info(
            "Loading REVE signal model %s (revision %s)",
            model_name,
            revision or "default",
        )
        signal_model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, revision=revision
        )
        signal_model.eval()
        for param in signal_model.parameters():
            param.requires_grad_(False)

        embed_dim = int(getattr(signal_model, "embed_dim", 512))
        patch_size = int(getattr(signal_model, "patch_size", 200))
        _MODEL_CACHE[key] = (signal_model, pos_model, embed_dim, patch_size)
        return _MODEL_CACHE[key]


def validate_batch(
    X: Any, n_channels: int, min_samples: int
) -> np.ndarray:
    """Return ``X`` as a contiguous ``(n, n_channels, n_samples)`` float32 array.

    Accepts ``(n, C, T)`` or ``(n, T, C)`` batches, a single ``(C, T)`` window,
    and torch tensors. The channel axis may be either of the last two axes; it
    is moved to axis 1 when needed.
    """
    try:  # torch is optional for callers that only validate arrays
        import torch

        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
    except ImportError:  # pragma: no cover - torch is a declared dependency
        pass

    X = np.asarray(X)

    if X.ndim == 2:
        if X.shape[0] == n_channels:
            X = X[None, ...]
        elif X.shape[1] == n_channels:
            X = X.T[None, ...]
        else:
            raise ValueError(
                f"window has shape {X.shape}; expected (n_channels, n_samples) "
                f"with n_channels={n_channels}"
            )
    elif X.ndim == 3:
        if X.shape[1] == n_channels:
            pass
        elif X.shape[2] == n_channels:
            X = np.transpose(X, (0, 2, 1))
        else:
            raise ValueError(
                f"X has {X.shape[1]} channels on axis 1 and {X.shape[2]} on "
                f"axis 2, but {n_channels} electrodes were provided"
            )
    else:
        raise ValueError(
            "X must be 3-D (n_windows, n_channels, n_samples) or a single "
            f"(n_channels, n_samples) window, got shape {X.shape}"
        )

    if not np.issubdtype(X.dtype, np.number) or np.iscomplexobj(X):
        raise ValueError(f"X must contain real numeric values, got dtype {X.dtype}")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values (NaN or Inf)")
    if X.shape[2] < min_samples:
        raise ValueError(
            f"each window must contain at least {min_samples} samples, "
            f"got {X.shape[2]}"
        )

    return np.ascontiguousarray(X, dtype=np.float32)


class ReveEncoder:
    """Frozen REVE-Base encoder with attention-pooled 512-d output.

    Parameters
    ----------
    electrodes:
        Montage or ordered electrode names. The order must match the channel
        order of every array passed to :meth:`encode`.
    model_name:
        Hugging Face id of the signal transformer.
    pos_model_name:
        Hugging Face id of the electrode position bank.
    revision:
        Git revision, tag or branch of both Hugging Face repositories. Pin it
        when a model bundle records an encoder revision.
    device:
        ``None``/``"auto"``, ``"cpu"``, ``"mps"`` or ``"cuda"``.
    batch_size:
        Number of windows encoded per forward pass.
    """

    def __init__(
        self,
        electrodes: Any,
        model_name: str = DEFAULT_MODEL_NAME,
        pos_model_name: str = DEFAULT_POS_MODEL_NAME,
        device: str | None = None,
        batch_size: int = 16,
        revision: str | None = None,
    ):
        self.electrodes = tuple(electrode_names(electrodes))
        self.model_name = model_name
        self.pos_model_name = pos_model_name
        if revision is not None and not str(revision).strip():
            raise ValueError("revision must be a non-empty string or None")
        self.revision = str(revision).strip() if revision is not None else None
        self.device = device
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self._signal_model: Any = None
        self._pos_model: Any = None
        self._pos: Any = None
        self._device: str | None = None
        self.embedding_dim: int = 512
        self.patch_size: int = 200

    # ------------------------------------------------------------------ #
    # lazy model handling
    # ------------------------------------------------------------------ #
    def _ensure_loaded(self) -> None:
        if self._signal_model is not None:
            return

        signal_model, pos_model, embed_dim, patch_size = _load_models(
            self.model_name, self.pos_model_name, self.revision
        )
        device = resolve_device(self.device)

        pos_model.to(device)
        signal_model.to(device)

        missing = [
            name for name in self.electrodes if name not in pos_model.mapping
        ]
        if missing:
            raise ValueError(
                f"Unknown electrode names for position model: {missing}"
            )

        import torch

        with torch.inference_mode():
            pos = pos_model(list(self.electrodes))
        if not isinstance(pos, torch.Tensor):
            pos = torch.as_tensor(pos)

        self._signal_model = signal_model
        self._pos_model = pos_model
        self._pos = pos.detach().to(device).float()
        self._device = device
        self.embedding_dim = embed_dim
        self.patch_size = patch_size

    def _move_to(self, device: str) -> None:
        import torch

        self._signal_model.to(device)
        self._pos_model.to(device)
        self._pos = self._pos.to(device)
        self._device = device

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """Load the frozen weights and position bank into memory.

        Call this before starting a live stream so the first window does not
        pay the model-loading cost. :meth:`encode` loads lazily as well.
        """
        self._ensure_loaded()

    def encode(self, X: Any) -> np.ndarray:
        """Encode windows into attention-pooled embeddings.

        Parameters
        ----------
        X:
            ``(n, n_channels, n_samples)``, ``(n, n_samples, n_channels)`` or a
            single ``(n_channels, n_samples)`` window.

        Returns
        -------
        ``(n, embedding_dim)`` float32 array.
        """
        batch = validate_batch(X, len(self.electrodes), self.patch_size)
        self._ensure_loaded()
        try:
            return self._encode_batch_set(batch)
        except RuntimeError as exc:
            if self._device == "cpu":
                raise
            log.warning(
                "REVE failed on %s (%s); retrying on CPU", self._device, exc
            )
            self._move_to("cpu")
            return self._encode_batch_set(batch)

    def _encode_batch_set(self, X: np.ndarray) -> np.ndarray:
        import torch

        n_samples = X.shape[0]
        out = np.empty((n_samples, self.embedding_dim), dtype=np.float32)
        pos = self._pos.unsqueeze(0)

        for start in range(0, n_samples, self.batch_size):
            stop = min(start + self.batch_size, n_samples)
            tensor = torch.from_numpy(X[start:stop]).to(self._device)
            pos_batch = pos.expand(tensor.shape[0], -1, -1)

            with torch.inference_mode():
                output = self._signal_model(tensor, pos_batch)
                pooled = self._signal_model.attention_pooling(output)

            out[start:stop] = pooled.detach().cpu().numpy()

        return out

    # Names kept for compatibility with the existing sklearn wrapper/tests.
    @property
    def signal_embedding_model(self) -> Any:
        self._ensure_loaded()
        return self._signal_model

    @property
    def pos_embedding_model(self) -> Any:
        self._ensure_loaded()
        return self._pos_model

    @property
    def electrodes_pos(self) -> Any:
        self._ensure_loaded()
        return self._pos

    @property
    def is_loaded(self) -> bool:
        return self._signal_model is not None


__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_POS_MODEL_NAME",
    "ReveEncoder",
    "electrode_names",
    "resolve_device",
    "validate_batch",
]
