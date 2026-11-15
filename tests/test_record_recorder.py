from __future__ import annotations

import time

import numpy as np

from hotb_2026.brainaccess_connection import SimulatedDriver, MONTAGE
from hotb_2026.record import EEGRecorder


def test_recorder_cutout():
    driver = SimulatedDriver(MONTAGE)
    rec = EEGRecorder(driver, MONTAGE, poll_interval=0.01)
    rec.start()
    rec.set_state("rest")
    time.sleep(0.15)
    rec.set_state(None)
    time.sleep(0.1)
    rec.set_state("1back")
    time.sleep(0.15)
    rec.stop()
    signal, time_arr, changes = rec.get_recording()
    assert signal.ndim == 2
    assert signal.shape[1] == MONTAGE.n_channels
    assert time_arr.shape[0] == signal.shape[0]
    assert [c[1] for c in changes] == ["rest", "1back"]
    # cut-out period should reduce samples vs total stream time
    assert rec.sample_count > signal.shape[0]
