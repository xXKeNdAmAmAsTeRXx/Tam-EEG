# `record` package

Runs the PsychoPy n-back procedure (rest / 1-back / 3-back) and records EEG
sessions for downstream analysis.

## Subpackages

- `core` — settings, randomized procedure generation, background `EEGRecorder`
- `task` — PsychoPy window, stimuli, and session orchestration
- `io` — session folder writing
- `api` — notebook-friendly helpers
- `cli` — command-line entry point

## Quick start

### CLI

```bash
uv run hotb-record --simulate --subject demo --sessions 1 --out-dir data
```

### Notebook

```python
from hotb_2026.brainaccess_connection import MONTAGE, SimulatedDriver
from hotb_2026.record import run_session

driver = SimulatedDriver(MONTAGE)
session_dir = run_session(
    subject_id="demo",
    session_idx=1,
    driver=driver,
    montage=MONTAGE,
    out_dir="data",
    simulate=True,
)
print(session_dir)
```

## Output layout

```
session_<session_idx>/
├── meta.json            # session_id, subject_id, electrodes, sfreq
├── signal.npy           # (n_samples, n_channels), µV, electrode order
├── time.npy             # (n_samples,), stream time in seconds
└── state_changes.csv    # time,label — when the active label changes
```

## Montage setup

Pass the same `Montage` to both the driver and the recorder/session API. The
signal columns follow `montage.names`, and `montage.sfreq` defines the time base.

## Cut-out behavior

Instruction, feedback, welcome, and finish screens call `set_state(None)`. During
those periods EEG samples are discarded (cut out), while the stream sample counter
keeps advancing so timestamps remain continuous. `state_changes.csv` contains
only active task labels: `rest`, `1back`, `3back`.

## Reproducibility

Pass `seed` (CLI: `--seed`) to make block order and durations reproducible.

## Notes

- PsychoPy is imported lazily so `import hotb_2026.record` works without a
  display.
- For real hardware, build a `BrainAccessDriver` instead of `SimulatedDriver`.
- On Linux without GLU, set `LD_LIBRARY_PATH` to a libGLU location if needed.