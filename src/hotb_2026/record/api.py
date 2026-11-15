"""High-level helpers for notebooks and scripts."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .task.psychopy_task import run_session as _run_session_impl


def _resolve_driver_and_montage(driver, montage, simulate):
    """Fill in default montage/driver, mirroring CLI/notebook conveniences."""
    if montage is None:
        from hotb_2026.brainaccess_connection import MONTAGE

        montage = MONTAGE
    if driver is None:
        if simulate:
            from hotb_2026.brainaccess_connection import SimulatedDriver

            driver = SimulatedDriver(montage)
        else:
            raise ValueError("driver is required unless simulate=True")
    return driver, montage


def run_session_helper(
    subject_id: str,
    session_idx: int = 1,
    *,
    driver=None,
    montage=None,
    out_dir: str | Path = "data",
    simulate: bool = False,
    seed: int | None = None,
    **kwargs: Any,
) -> Path:
    driver, montage = _resolve_driver_and_montage(driver, montage, simulate)
    rng = random.Random(seed) if seed is not None else None
    return _run_session_impl(
        subject_id=subject_id,
        session_idx=session_idx,
        driver=driver,
        montage=montage,
        out_dir=out_dir,
        rng=rng,
        **kwargs,
    )


def run_experiment(
    subject_id: str,
    n_sessions: int = 1,
    *,
    driver=None,
    montage=None,
    out_dir: str | Path = "data",
    simulate: bool = False,
    seed: int | None = None,
    **kwargs: Any,
) -> list[Path]:
    driver, montage = _resolve_driver_and_montage(driver, montage, simulate)
    results: list[Path] = []
    for i in range(1, n_sessions + 1):
        child_seed = None if seed is None else seed + i
        child_rng = random.Random(child_seed) if child_seed is not None else None
        results.append(
            _run_session_impl(
                subject_id=subject_id,
                session_idx=i,
                driver=driver,
                montage=montage,
                out_dir=out_dir,
                rng=child_rng,
                **kwargs,
            )
        )
    return results


run_session = run_session_helper