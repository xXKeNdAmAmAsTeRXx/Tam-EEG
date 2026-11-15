from __future__ import annotations

from pathlib import Path

import pytest

import hotb_2026.record.api as api


def _use_fake_session(monkeypatch, calls: list[dict]) -> None:
    def fake_run_session(**kwargs):
        calls.append(kwargs)
        return Path(kwargs["out_dir"]) / f"session_{kwargs['session_idx']}"

    monkeypatch.setattr(api, "_run_session_impl", fake_run_session)


def test_run_experiment_starts_at_requested_index(tmp_path: Path, monkeypatch):
    calls: list[dict] = []
    _use_fake_session(monkeypatch, calls)

    results = api.run_experiment(
        "subj",
        n_sessions=2,
        start_session_idx=3,
        driver=object(),
        montage=object(),
        out_dir=tmp_path,
    )

    assert [call["session_idx"] for call in calls] == [3, 4]
    assert [Path(result).name for result in results] == ["session_3", "session_4"]
    assert all(call["out_dir"] == tmp_path for call in calls)


def test_run_experiment_aborts_before_recording_on_conflict(
    tmp_path: Path, monkeypatch
):
    (tmp_path / "subj" / "session_4").mkdir(parents=True)
    calls: list[dict] = []
    _use_fake_session(monkeypatch, calls)

    with pytest.raises(FileExistsError, match="next free: 5"):
        api.run_experiment(
            "subj",
            n_sessions=3,
            start_session_idx=3,
            driver=object(),
            montage=object(),
            out_dir=tmp_path,
        )

    assert calls == []


def test_run_session_aborts_when_index_exists(tmp_path: Path, monkeypatch):
    (tmp_path / "subj" / "session_1").mkdir(parents=True)
    calls: list[dict] = []
    _use_fake_session(monkeypatch, calls)

    with pytest.raises(FileExistsError):
        api.run_session(
            "subj",
            session_idx=1,
            driver=object(),
            montage=object(),
            out_dir=tmp_path,
        )

    assert calls == []


def test_run_experiment_rejects_invalid_start_index(tmp_path: Path):
    with pytest.raises(ValueError):
        api.run_experiment("subj", n_sessions=1, start_session_idx=0, out_dir=tmp_path)
