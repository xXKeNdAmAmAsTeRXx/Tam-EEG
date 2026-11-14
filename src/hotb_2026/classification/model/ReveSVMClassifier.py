from __future__ import annotations

from typing import Any, List

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from transformers import AutoModel


class ReveSVMClassifier(BaseEstimator, ClassifierMixin):
    """scikit-learn classifier built on top of a frozen REVE transformer.

    The transformer weights are pretrained and frozen.  A standard ``SVC`` is
    trained on the embeddings produced by REVE for each input window.
    """

    def __init__(
        self,
        electrodes: Any,
        pos_embedding_model_name: str = "brain-bzh/reve-positions",
        signal_embedding_model_name: str = "brain-bzh/reve-base",
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: Any = "scale",
    ):
        self.electrodes = electrodes
        self.pos_embedding_model_name = pos_embedding_model_name
        self.signal_embedding_model_name = signal_embedding_model_name
        self.C = C
        self.kernel = kernel
        self.gamma = gamma

        # Encode electrodes: this does not change during training or inference.
        if hasattr(electrodes, "eeg_names"):
            electrode_names = list(electrodes.eeg_names)
        elif hasattr(electrodes, "names"):
            electrode_names = list(electrodes.names)
        else:
            electrode_names = list(electrodes)

        if not electrode_names:
            raise ValueError("electrodes must not be empty")
        if any(not isinstance(name, str) or not name for name in electrode_names):
            raise ValueError("electrode names must be non-empty strings")
        if len(set(electrode_names)) != len(electrode_names):
            raise ValueError("electrode names must be unique")

        self._electrodes = electrode_names

        # Position embeddings (frozen lookup table).
        pos_embedding_model = AutoModel.from_pretrained(
            pos_embedding_model_name, trust_remote_code=True
        )
        pos_embedding_model.eval()
        self.electrodes_pos = pos_embedding_model(electrode_names)
        if not isinstance(self.electrodes_pos, torch.Tensor):
            self.electrodes_pos = torch.as_tensor(self.electrodes_pos)
        self.electrodes_pos = self.electrodes_pos.detach().cpu().float()

        missing = [
            name for name in electrode_names if name not in pos_embedding_model.mapping
        ]
        if missing:
            raise ValueError(f"Unknown electrode names for position model: {missing}")

        # Signal embedding model (frozen transformer).
        self.signal_embedding_model = AutoModel.from_pretrained(
            signal_embedding_model_name, trust_remote_code=True
        )
        self.signal_embedding_model.eval()
        for param in self.signal_embedding_model.parameters():
            param.requires_grad_(False)

        if not hasattr(self.signal_embedding_model, "embed_dim"):
            raise RuntimeError("Signal embedding model has no embed_dim attribute")
        self.embedding_dim_ = int(self.signal_embedding_model.embed_dim)

    # ------------------------------------------------------------------ #
    # input handling
    # ------------------------------------------------------------------ #
    def _validate_X(self, X: Any) -> np.ndarray:
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        X = np.asarray(X)

        if X.ndim != 3:
            raise ValueError(
                f"X must be 3-D (n_samples, n_channels, n_timesteps), got shape {X.shape}"
            )

        n_channels = len(self._electrodes)

        if X.shape[1] == n_channels:
            pass
        elif X.shape[2] == n_channels:
            X = np.transpose(X, (0, 2, 1))
        else:
            raise ValueError(
                f"X has {X.shape[1]} channels on axis 1 and {X.shape[2]} on axis 2, "
                f"but {n_channels} electrodes were provided"
            )

        if not np.issubdtype(X.dtype, np.number) or np.iscomplexobj(X):
            raise ValueError(f"X must contain real numeric values, got dtype {X.dtype}")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains non-finite values (NaN or Inf)")

        min_samples = int(getattr(self.signal_embedding_model, "patch_size", 200))
        if X.shape[2] < min_samples:
            raise ValueError(
                f"Each window must contain at least {min_samples} samples, got {X.shape[2]}"
            )

        return np.ascontiguousarray(X, dtype=np.float64)

    def _encode(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        embeddings = np.empty((n_samples, self.embedding_dim_), dtype=np.float64)

        batch_size = 16
        pos_embeddings = self.electrodes_pos

        model = self.signal_embedding_model
        params = list(model.parameters())
        device = params[0].device if params else torch.device("cpu")
        pos_embeddings = pos_embeddings.to(device)

        for start in range(0, n_samples, batch_size):
            stop = min(start + batch_size, n_samples)
            batch = X[start:stop]
            tensor = torch.from_numpy(
                np.ascontiguousarray(batch, dtype=np.float32)
            ).to(device)

            pos_batch = pos_embeddings.unsqueeze(0).expand(tensor.shape[0], -1, -1)

            with torch.no_grad():
                output = model(tensor, pos_batch)
                pooled = model.attention_pooling(output)

            embeddings[start:stop] = pooled.detach().cpu().numpy().astype(np.float64)

        return embeddings

    # ------------------------------------------------------------------ #
    # sklearn API
    # ------------------------------------------------------------------ #
    def _check_fitted(self) -> None:
        if not hasattr(self, "_svc"):
            raise RuntimeError("ReveSVMClassifier is not fitted yet")

    def fit(self, X: Any, y: Any) -> "ReveSVMClassifier":
        X = self._validate_X(X)

        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"X and y have inconsistent numbers of samples: {X.shape[0]} vs {y.shape[0]}"
            )

        embeddings = self._encode(X)

        self._svc = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            probability=True,
            random_state=None,
        )
        self._svc.fit(embeddings, y)

        self.classes_ = self._svc.classes_
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = embeddings.shape[1]
        return self

    def predict(self, X: Any) -> np.ndarray:
        self._check_fitted()
        X = self._validate_X(X)
        embeddings = self._encode(X)
        return self._svc.predict(embeddings)

    def predict_proba(self, X: Any) -> np.ndarray:
        self._check_fitted()
        X = self._validate_X(X)
        embeddings = self._encode(X)
        return self._svc.predict_proba(embeddings)

    def decision_function(self, X: Any) -> np.ndarray:
        self._check_fitted()
        X = self._validate_X(X)
        embeddings = self._encode(X)
        return self._svc.decision_function(embeddings)
