from __future__ import annotations

import argparse

import pytest

from hotb_2026.brainaccess_connection import DEVICE_NAME, MONTAGE
from hotb_2026.brainaccess_connection.main import (
    _bias_value,
    _device_name,
    _warn_ignored_simulate_options,
    build_parser,
)


def test_defaults_come_from_config() -> None:
    args = build_parser().parse_args([])
    assert args.device is None
    assert _device_name(args) == DEVICE_NAME
    assert args.window == 4.0
    assert args.sfreq == round(MONTAGE.sfreq)
    assert args.chunk_size == 25
    assert args.gain == 8
    assert args.bias == MONTAGE.bias_electrode
    assert args.name == "BrainAccess EEG"
    assert args.source_id == "brainaccess-eeg-1"


def test_explicit_flags_override_config() -> None:
    args = build_parser().parse_args(
        [
            "--device",
            "BA MINI 001",
            "--sfreq",
            "500",
            "--bias",
            "3",
            "--gain",
            "4",
        ]
    )
    assert _device_name(args) == "BA MINI 001"
    assert args.sfreq == 500
    assert args.bias == 3
    assert args.gain == 4


def test_simulate_does_not_warn_about_the_configured_device(
    caplog: pytest.LogCaptureFixture,
) -> None:
    args = build_parser().parse_args(["--simulate"])
    with caplog.at_level("WARNING"):
        _warn_ignored_simulate_options(args)
    assert caplog.text == ""


def test_simulate_warns_about_explicit_device(
    caplog: pytest.LogCaptureFixture,
) -> None:
    args = build_parser().parse_args(["--simulate", "--device", "BA MINI 001"])
    with caplog.at_level("WARNING"):
        _warn_ignored_simulate_options(args)
    assert "ignoring --device 'BA MINI 001'" in caplog.text


@pytest.mark.parametrize("text", ["none", "NONE", "off", "no"])
def test_bias_can_be_disabled(text: str) -> None:
    assert _bias_value(text) is None


def test_bias_accepts_index() -> None:
    assert _bias_value("3") == 3


def test_bias_rejects_text() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _bias_value("middle")
