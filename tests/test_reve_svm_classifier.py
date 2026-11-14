from __future__ import annotations

import numpy as np
import pytest

from hotb_2026.classification import ReveSVMClassifier

@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 8, 256)).astype(np.float32)
    y = np.array([0] * 6 + [1] * 6)
    clf = ReveSVMClassifier(
        electrodes=["F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"],
    )
    try:
        clf.fit(X, y)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"REVE model unavailable: {exc}")
    return clf, X, y


def test_fit_predict_proba(fitted):
    clf, X, y = fitted
    proba = clf.predict_proba(X)
    assert proba.shape == (12, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert clf.score(X, y) >= 0.5


def test_frozen_weights(fitted):
    clf, *_ = fitted
    model = clf.signal_embedding_model
    assert model.training is False
    for param in model.parameters():
        assert param.requires_grad is False


def test_clone_and_params(fitted):
    from sklearn.base import clone

    clf, *_ = fitted
    cloned = clone(clf)
    assert not hasattr(cloned, "_svc")
    params = clf.get_params()
    assert "C" in params and "kernel" in params