"""scikit-learn wrapper: frozen REVE encoder + SVM classifier."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC

from hotb_2026.features.reve import (
    DEFAULT_MODEL_NAME,
    DEFAULT_POS_MODEL_NAME,
    ReveEncoder,
    electrode_names,
    validate_batch,
)


class ReveSVMClassifier(BaseEstimator, ClassifierMixin):
    """scikit-learn classifier built on top of a frozen REVE transformer.

    The transformer weights are pretrained and frozen. A standard ``SVC`` is
    trained on the embeddings produced by REVE for each input window.

    This class is the convenience wrapper for interactive use. Offline
    training (``hotb_2026.classification.training``) works on cached
    embeddings with a plain scikit-learn pipeline instead, so it does not pay
    the model-loading cost on every estimator clone.

    Parameters
    ----------
    electrodes:
        Montage or ordered electrode names. Must match the channel order of
        every array passed to ``fit``/``predict``.
    pos_embedding_model_name, signal_embedding_model_name:
        Hugging Face ids of the position bank and signal transformer.
    C, kernel, gamma:
        Forwarded to :class:`sklearn.svm.SVC`.
    batch_size:
        Windows encoded per forward pass.
    device:
        ``None``/``"auto"``, ``"cpu"``, ``"mps"`` or ``"cuda"``.
    random_state:
        Forwarded to ``SVC`` (affects Platt probability calibration).
    """

    def __init__(
        self,
        electrodes: Any,
        pos_embedding_model_name: str = DEFAULT_POS_MODEL_NAME,
        signal_embedding_model_name: str = DEFAULT_MODEL_NAME,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: Any = "scale",
        batch_size: int = 16,
        device: str | None = None,
        random_state: int | None = None,
    ):
        self.electrodes = electrodes
        self.pos_embedding_model_name = pos_embedding_model_name
        self.signal_embedding_model_name = signal_embedding_model_name
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state

        self._electrodes = tuple(electrode_names(electrodes))
        self._encoder: ReveEncoder | None = None

    # ------------------------------------------------------------------ #
    # encoder
    # ------------------------------------------------------------------ #
    @property
    def encoder(self) -> ReveEncoder:
        if self._encoder is None:
            self._encoder = ReveEncoder(
                self._electrodes,
                model_name=self.signal_embedding_model_name,
                pos_model_name=self.pos_embedding_model_name,
                device=self.device,
                batch_size=self.batch_size,
            )
        return self._encoder

    @property
    def signal_embedding_model(self) -> Any:
        return self.encoder.signal_embedding_model

    @property
    def pos_embedding_model(self) -> Any:
        return self.encoder.pos_embedding_model

    @property
    def electrodes_pos(self) -> Any:
        return self.encoder.electrodes_pos

    # ------------------------------------------------------------------ #
    # input handling
    # ------------------------------------------------------------------ #
    def _validate_X(self, X: Any) -> np.ndarray:
        encoder = self.encoder
        return validate_batch(X, len(self._electrodes), encoder.patch_size)

    def _encode(self, X: np.ndarray) -> np.ndarray:
        return self.encoder.encode(X)

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
                f"X and y have inconsistent numbers of samples: "
                f"{X.shape[0]} vs {y.shape[0]}"
            )

        embeddings = self._encode(X)

        self._svc = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            probability=True,
            random_state=self.random_state,
        )
        self._svc.fit(embeddings, y)

        self.classes_ = self._svc.classes_
        self.n_classes_ = len(self.classes_)
        self.embedding_dim_ = embeddings.shape[1]
        self.n_features_in_ = embeddings.shape[1]
        return self

    def predict(self, X: Any) -> np.ndarray:
        self._check_fitted()
        X = self._validate_X(X)
        return self._svc.predict(self._encode(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        self._check_fitted()
        X = self._validate_X(X)
        return self._svc.predict_proba(self._encode(X))

    def decision_function(self, X: Any) -> np.ndarray:
        self._check_fitted()
        X = self._validate_X(X)
        return self._svc.decision_function(self._encode(X))


__all__ = ["ReveSVMClassifier"]
