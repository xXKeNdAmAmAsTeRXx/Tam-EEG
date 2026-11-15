from __future__ import annotations

import pytest

from hotb_2026.record import cli as record_cli
from hotb_2026.record.cli import build_parser


def test_cli_defaults():
    args = build_parser().parse_args([])
    assert args.sessions == 1
    assert args.session_idx is None
    assert args.out_dir == "data"
    assert args.simulate is False
    assert args.n_blocks == 3


def test_cli_session_flag_selects_single_session():
    args = build_parser().parse_args(["--session", "2"])
    assert args.session_idx == 2
    assert args.sessions == 1


def test_cli_session_idx_alias():
    args = build_parser().parse_args(["--session-idx", "4"])
    assert args.session_idx == 4
    assert args.sessions == 1


def test_cli_sessions_flag_is_a_count():
    args = build_parser().parse_args(["--sessions", "3"])
    assert args.sessions == 3
    assert args.session_idx is None


def test_cli_abbreviations_are_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--sessio", "2"])


def test_cli_aborts_before_recording_when_session_exists(tmp_path, monkeypatch):
    (tmp_path / "subj" / "session_2").mkdir(parents=True)
    called = False

    def fake_run_experiment(**kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(record_cli, "run_experiment", fake_run_experiment)

    with pytest.raises(SystemExit) as excinfo:
        record_cli.main(
            [
                "--subject",
                "subj",
                "--session",
                "2",
                "--out-dir",
                str(tmp_path),
                "--simulate",
            ]
        )
    assert "next free: 3" in str(excinfo.value)
    assert called is False
