"""Classify live EEG windows with the model stored in a bundle."""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from hotb_2026.features.reve import ReveEncoder, electrode_names
from hotb_2026.preprocessing import Preprocessor, WindowedPreprocessor

from .loader import BundleContract, load_bundle, load_pretrained_bundle
from .schema import Prediction

log = logging.getLogger(__name__)

PredictionCallback = Callable[[Prediction], None]


def _finite_time(current_time: Any) -> float:
    try:
        value = float(current_time)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"current_time must be a finite number of seconds, got "
            f"{current_time!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"current_time must be finite, got {current_time!r}"
        )
    return value


def _require_numeric(window: Any, where: str) -> np.ndarray:
    array = np.asarray(window)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(
            f"{where} must contain real numeric values, got dtype "
            f"{array.dtype}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{where} contains non-finite values (NaN or Inf)")
    return array


class _ChannelSelectingPreprocessor:
    """Select training channels before running the recorded preprocessing."""

    def __init__(
        self, preprocessor: Preprocessor, indices: np.ndarray | None
    ) -> None:
        self.preprocessor = preprocessor
        self.indices = indices

    def process(self, window: np.ndarray) -> np.ndarray:
        if self.indices is not None:
            window = np.take(window, self.indices, axis=0)
        return self.preprocessor.process(window)


class Predictor:
    """Load a saved bundle and classify one window at a time.

    The predictor owns the frozen encoder, the recorded preprocessing and the
    fitted classifier. It also tracks when the predicted label last changed so
    every :class:`~hotb_2026.inference.schema.Prediction` carries a
    ``state_change_time``.

    Parameters
    ----------
    model:
        Fitted classifier loaded from ``model.joblib``.
    contract:
        Validated metadata returned by :func:`~hotb_2026.inference.loader.load_bundle`.
    encoder:
        Optional encoder double. When omitted, a
        :class:`~hotb_2026.features.ReveEncoder` is built from the bundle.
    preprocessor:
        Optional preprocessing override. When omitted, the recorded
        ``PreprocessingConfig`` is used.
    device:
        Device passed to :class:`~hotb_2026.features.ReveEncoder`.
    source_electrodes:
        Channel order of the live signal, if it differs from the training
        order. See :meth:`set_source_electrodes`.

    Notes
    -----
    ``predict`` consumes one already-preprocessed window in training electrode
    order. Use :meth:`predict_raw` when you have the raw stream window.
    """

    def __init__(
        self,
        model: Any,
        contract: BundleContract,
        *,
        encoder: Any | None = None,
        preprocessor: Preprocessor | None = None,
        device: str | None = None,
        source_electrodes: Any | None = None,
    ) -> None:
        self.model = model
        self.contract = contract
        self.device = device

        if encoder is None:
            encoder = ReveEncoder(
                contract.electrodes,
                model_name=contract.encoder_model_name,
                pos_model_name=contract.encoder_pos_model_name,
                revision=contract.encoder_revision,
                device=device,
            )
        self.encoder = encoder
        self.preprocessor = preprocessor or Preprocessor(contract.preprocessing)

        self._source_electrodes: tuple[str, ...] = tuple(contract.electrodes)
        self._channel_index: np.ndarray | None = None
        if source_electrodes is not None:
            self.set_source_electrodes(source_electrodes)

        self._last_time: float | None = None
        self._last_label: int | None = None
        self._state_change_time: float | None = None

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_bundle(
        cls,
        directory: str | Path,
        *,
        electrodes: Any | None = None,
        encoder: Any | None = None,
        preprocessor: Preprocessor | None = None,
        device: str | None = None,
    ) -> "Predictor":
        """Load a local bundle directory.

        Parameters
        ----------
        directory:
            Folder containing ``model.joblib`` and ``metadata.json``.
        electrodes:
            Channel order of incoming raw windows (a montage or a list of
            names). Defaults to the training order.
        encoder, preprocessor, device:
            Overrides for testing or a custom deployment.
        """
        loaded = load_bundle(directory)
        return cls(
            loaded.model,
            loaded.contract,
            encoder=encoder,
            preprocessor=preprocessor,
            device=device,
            source_electrodes=electrodes,
        )

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        revision: str | None = None,
        token: str | None = None,
        cache_dir: str | Path | None = None,
        electrodes: Any | None = None,
        encoder: Any | None = None,
        preprocessor: Preprocessor | None = None,
        device: str | None = None,
        local_files_only: bool = False,
    ) -> "Predictor":
        """Download a bundle from the Hugging Face Hub and load it.

        ``revision`` selects a tag, branch or commit, so a deployment can pin
        the exact bundle it was tested with.
        """
        loaded = load_pretrained_bundle(
            repo_id,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        return cls(
            loaded.model,
            loaded.contract,
            encoder=encoder,
            preprocessor=preprocessor,
            device=device,
            source_electrodes=electrodes,
        )

    # ------------------------------------------------------------------ #
    # metadata
    # ------------------------------------------------------------------ #
    @property
    def bundle_dir(self) -> Path:
        return self.contract.directory

    @property
    def model_version(self) -> str:
        return self.contract.model_version

    @property
    def metadata(self) -> dict[str, Any]:
        return self.contract.metadata

    @property
    def electrodes(self) -> tuple[str, ...]:
        """Electrode order expected by the encoder (training order)."""
        return self.contract.electrodes

    @property
    def source_electrodes(self) -> tuple[str, ...]:
        """Channel order of raw windows passed to :meth:`predict_raw`."""
        return self._source_electrodes

    @property
    def window_s(self) -> float:
        return self.contract.window_s

    @property
    def n_samples(self) -> int:
        return self.contract.n_samples

    @property
    def raw_samples(self) -> int:
        return self.contract.raw_samples

    @property
    def sfreq(self) -> float:
        """Sample rate after preprocessing."""
        return self.contract.sfreq

    @property
    def raw_sfreq(self) -> float:
        """Sample rate of the live signal before preprocessing."""
        return self.contract.raw_sfreq

    @property
    def label_names(self) -> tuple[str, ...]:
        return self.contract.label_names

    def set_source_electrodes(self, electrodes: Any | None) -> "Predictor":
        """Declare the channel order of raw windows.

        Training electrodes may appear in a different order or alongside extra
        channels. The predictor selects and reorders them before preprocessing;
        a required electrode that is missing raises ``ValueError``.
        """
        if electrodes is None:
            self._source_electrodes = tuple(self.electrodes)
            self._channel_index = None
            return self

        source = tuple(electrode_names(electrodes))
        missing = [name for name in self.electrodes if name not in source]
        if missing:
            raise ValueError(
                "source signal is missing training electrode(s): "
                f"{', '.join(missing)}"
            )
        index = np.asarray(
            [source.index(name) for name in self.electrodes], dtype=int
        )
        if np.array_equal(index, np.arange(len(self.electrodes))):
            index = None
        self._source_electrodes = source
        self._channel_index = index
        return self

    # ------------------------------------------------------------------ #
    # prediction
    # ------------------------------------------------------------------ #
    def predict(
        self, processed_window: Any, current_time: float
    ) -> Prediction:
        """Classify one preprocessed window.

        Parameters
        ----------
        processed_window:
            ``(n_channels, n_samples)`` array in training electrode order, as
            returned by the recorded :class:`~hotb_2026.preprocessing.Preprocessor`.
        current_time:
            Stream time at the end of the window, in seconds.
        """
        t = _finite_time(current_time)
        window = self._validate_processed(processed_window)

        if self._last_time is not None and t < self._last_time:
            log.debug("Stream time moved backwards (%.3f s); resetting", t)
            self.reset()

        features = self._encode(window)
        probabilities, label, confidence = self._probabilities(features)

        if self._last_label != label:
            self._last_label = label
            self._state_change_time = t
        self._last_time = t
        assert self._state_change_time is not None  # set on the first prediction

        return Prediction(
            label=label,
            label_name=self.label_names[label],
            confidence=confidence,
            probabilities=probabilities,
            current_time=t,
            state_change_time=self._state_change_time,
            window_start=t - self.window_s,
            window_end=t,
            model_version=self.model_version,
        )

    def predict_raw(
        self, raw_window: Any, current_time: float
    ) -> Prediction:
        """Preprocess and classify one raw window.

        ``raw_window`` is ``(n_channels, n_samples)`` in
        :attr:`source_electrodes` order. When it is longer than
        :attr:`raw_samples`, the trailing samples are used.
        """
        t = _finite_time(current_time)
        window = _require_numeric(raw_window, "raw_window")
        if window.ndim != 2:
            raise ValueError(
                "raw_window must be 2-D (n_channels, n_samples), "
                f"got shape {window.shape}"
            )
        if window.shape[0] != len(self._source_electrodes):
            raise ValueError(
                f"raw_window has {window.shape[0]} channels but "
                f"{len(self._source_electrodes)} source electrodes were "
                f"declared: {', '.join(self._source_electrodes)}"
            )
        if window.shape[1] < self.raw_samples:
            raise ValueError(
                f"raw_window has {window.shape[1]} samples but a "
                f"{self.window_s:g} s window at {self.raw_sfreq:g} Hz needs "
                f"{self.raw_samples}"
            )

        if self._channel_index is not None:
            window = np.take(window, self._channel_index, axis=0)
        window = window[:, -self.raw_samples :]
        processed = self.preprocessor.process(window)
        return self.predict(processed, t)

    def processor(
        self, on_prediction: PredictionCallback | None = None
    ) -> WindowedPreprocessor:
        """Build the ``EEGStreamer`` processor for this bundle.

        The returned adapter preprocesses every completed window and calls
        ``predict``. Pass ``streamer.window_s = predictor.window_s + 1.0`` (or
        any larger value) so the adapter can slice exact, non-overlapping
        windows out of the streamer's ring buffer.

        ``on_prediction`` is called with each :class:`Prediction` on the
        streamer thread.
        """
        preprocessor = _ChannelSelectingPreprocessor(
            self.preprocessor, self._channel_index
        )

        def _on_window(window: np.ndarray, t: float) -> None:
            prediction = self.predict(window, t)
            if on_prediction is not None:
                on_prediction(prediction)

        return WindowedPreprocessor(
            preprocessor,
            sfreq=self.raw_sfreq,
            window_s=self.window_s,
            on_window=_on_window,
        )

    def warmup(self) -> "Predictor":
        """Load the encoder before streaming to avoid a first-window stall."""
        load = getattr(self.encoder, "load", None)
        if callable(load):
            load()
        else:
            probe = np.zeros(
                (1, len(self.electrodes), self.n_samples), dtype=np.float32
            )
            self.encoder.encode(probe)
        return self

    def reset(self) -> None:
        """Forget label timing, e.g. before restarting a stream."""
        self._last_time = None
        self._last_label = None
        self._state_change_time = None

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _validate_processed(self, processed_window: Any) -> np.ndarray:
        window = _require_numeric(processed_window, "processed_window")
        expected = (len(self.electrodes), self.n_samples)
        if window.ndim != 2:
            raise ValueError(
                "processed_window must be 2-D (n_channels, n_samples), "
                f"got shape {window.shape}"
            )
        if window.shape != expected:
            raise ValueError(
                f"processed_window has shape {window.shape}; expected "
                f"{expected} in training electrode order"
            )
        return np.asarray(window, dtype=np.float32)

    def _encode(self, window: np.ndarray) -> np.ndarray:
        features = np.asarray(
            self.encoder.encode(window), dtype=np.float32
        )
        if features.ndim == 1:
            features = features[None, :]
        expected = (1, self.contract.embedding_dim)
        if features.shape != expected:
            raise ValueError(
                f"the encoder returned features with shape {features.shape}; "
                f"expected {expected} for this bundle"
            )
        if not np.all(np.isfinite(features)):
            raise ValueError("the encoder returned non-finite features")
        return features

    def _probabilities(
        self, features: np.ndarray
    ) -> tuple[dict[str, float], int, float]:
        raw = np.asarray(self.model.predict_proba(features))
        expected = (1, len(self.label_names))
        if raw.shape != expected:
            raise ValueError(
                f"predict_proba returned shape {raw.shape}; expected "
                f"{expected} for {len(self.label_names)} classes"
            )
        row = raw[0]
        if not np.all(np.isfinite(row)):
            raise ValueError("predict_proba returned non-finite probabilities")
        probabilities = {
            name: float(row[index])
            for index, name in enumerate(self.label_names)
        }
        label = int(np.argmax(row))
        confidence = float(row[label])
        return probabilities, label, confidence

    def __repr__(self) -> str:
        return (
            f"Predictor(model_version={self.model_version!r}, "
            f"window_s={self.window_s:g}, electrodes={list(self.electrodes)!r})"
        )


__all__ = ["Predictor", "PredictionCallback"]
