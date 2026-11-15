from __future__ import annotations

import pytest

from hotb_2026.brainaccess_connection import DEFAULT_MONTAGE, Electrode, Montage


def test_default_montage_is_mini_layout() -> None:
    # Electrode labels depend on how the cap is wired, so only assert the fixed
    # MINI layout: 8 inputs on pins 0-7 with bias on the last channel.
    assert isinstance(DEFAULT_MONTAGE, Montage)
    assert DEFAULT_MONTAGE.n_channels == 8
    assert [e.pin for e in DEFAULT_MONTAGE.electrodes] == list(range(8))
    assert len(set(DEFAULT_MONTAGE.names)) == DEFAULT_MONTAGE.n_channels
    assert all(e.name.strip() for e in DEFAULT_MONTAGE.electrodes)
    assert DEFAULT_MONTAGE.bias_electrode == 7


def test_duplicate_names_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate electrode names"):
        Montage(
            electrodes=(Electrode("F3", 0), Electrode("F3", 1)),
            bias_electrode=None,
        )


def test_duplicate_pins_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate device pins"):
        Montage(
            electrodes=(Electrode("F3", 0), Electrode("F4", 0)),
            bias_electrode=None,
        )


def test_empty_montage_rejected() -> None:
    with pytest.raises(ValueError, match="at least one electrode"):
        Montage(electrodes=())


def test_negative_pin_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Montage(electrodes=(Electrode("F3", -1),), bias_electrode=None)


def test_bias_must_index_an_electrode() -> None:
    with pytest.raises(ValueError, match="bias_electrode"):
        Montage(electrodes=(Electrode("F3", 0),), bias_electrode=1)


def test_bias_can_be_disabled() -> None:
    montage = Montage(electrodes=(Electrode("F3", 0),), bias_electrode=None)
    assert montage.bias_electrode is None
