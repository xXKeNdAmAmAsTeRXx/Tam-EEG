# Preprocessing

Per-window EEG cleaning on top of `brainaccess_connection`. The package does not
open a device and does not own a signal buffer: it transforms the rolling window
that `EEGStreamer` already provides.

```text
EEGStreamer window (n_channels, n_samples)
        │
        ▼
notch 50 Hz ──▶ band-pass 0.5–45 Hz ──▶ resample 250→200 Hz ──▶ z-score ──▶ clip ±15
        │
        ▼
processed window (n_channels, 200 Hz × window_s)
```

## Pipeline

| Step | Implementation | Default |
| --- | --- | --- |
| Notch | `scipy.signal.iirnotch` → `tf2sos` → `sosfiltfilt` | 50 Hz, Q = 30 |
| Band-pass | `scipy.signal.butter(..., output="sos")` → `sosfiltfilt` | 0.5–45 Hz, order 4 |
| Resample | `scipy.signal.resample_poly(..., padtype="line")` | 250 → 200 Hz |
| Z-score | per channel, over samples, `ddof=0` | enabled |
| Clip | `np.clip` after z-score | ±15 |

All filters are zero-phase (`filtfilt`) and designed once when the
`Preprocessor` is created. Processing happens on `float64`, the result is
`float32`. The input array is never modified. Set any stage to `None` in
`PreprocessingConfig` (or `zscore=False`) to skip it. Disabled stages still have
their parameters validated, so a bad value fails at config construction instead
of when the stage is re-enabled.

## Usage

`MONTAGE` stays the single source of truth for the input rate:

```python
from hotb_2026.brainaccess_connection import (
    MONTAGE,
    EEGStreamer,
    SimulatedDriver,
)
from hotb_2026.preprocessing import PreprocessingConfig, Preprocessor

pre = Preprocessor(PreprocessingConfig(sfreq=MONTAGE.sfreq))
window = ...  # (n_channels, n_samples) from the streamer or a file
clean = pre.process(window)

print(clean.shape, pre.output_sfreq)  # (n_channels, 200 Hz * window_s), 200.0
```

### Inside the streamer

`WindowedPreprocessor` is a regular `EEGStreamer` processor. It waits for a full
window and then preprocesses it, so the pipeline runs once per `window_s`
instead of once per chunk. Pass `window_s=streamer.window_s` to process the
whole established streamer window:

```python
from collections import deque

from hotb_2026.preprocessing import WindowedPreprocessor

results = deque(maxlen=10)
streamer = EEGStreamer(
    SimulatedDriver(MONTAGE),
    MONTAGE,
    window_s=4.0,
)
streamer.add_processor(
    WindowedPreprocessor(
        Preprocessor(PreprocessingConfig(sfreq=MONTAGE.sfreq)),
        sfreq=MONTAGE.sfreq,
        window_s=streamer.window_s,  # whole window, no hidden sub-window
        on_window=lambda clean, t: results.append((clean, t)),
    )
)
streamer.start()
...
streamer.stop()
```

The adapter emits one result per completed window. If the streamer window is
longer than `window_s`, it slices the exact completed window out of the ring
buffer. With equal windows, a chunk that overshoots the boundary means the
first samples are already overwritten; the adapter then uses the latest full
window and counts it in `shifted_windows`. Pick a chunk size that divides
`window_s * sfreq` (e.g. 25 samples at 250 Hz for a 4 s window) to avoid that.
`skipped_windows` counts windows lost when one chunk crosses several boundaries.
Both counters are reset automatically when stream time moves backwards (streamer
restart).

## Parameters

`PreprocessingConfig` validates itself, so a bad combination fails at
construction, not mid-stream:

| Field | Default | Notes |
| --- | --- | --- |
| `sfreq` | `250.0` | pass `MONTAGE.sfreq` |
| `notch_freq` | `50.0` | `None` disables the notch |
| `notch_q` | `30.0` | ≈1.7 Hz stop band at 50 Hz |
| `bandpass` | `(0.5, 45.0)` | `None` disables the band-pass |
| `filter_order` | `4` | Butterworth order (SOS) |
| `target_sfreq` | `200.0` | `None` keeps the input rate |
| `zscore` | `True` | per channel, per window |
| `clip` | `15.0` | after z-score, `None` disables clipping |

The band-pass high edge must stay below both the input and the target Nyquist
frequency; the default 45 Hz works for 250 Hz input and 200 Hz output.
`clip` is applied to whatever reaches that stage, so with `zscore=False` it
truncates the filtered signal in µV rather than in standard deviations.

## Limitations

- **Per-window filtering**: `filtfilt` and `resample_poly` see only one window,
  so each window has small edge transients. Use windows of at least 4 s (8–10 s
  is better for the 0.5 Hz high-pass). For a continuous, artifact-free stream,
  use stateful filters (`sosfilt` with `zi`) instead.
- **Z-score removes absolute amplitude**: the output is in standard deviations,
  not µV. Compute RMS or band power **before** the z-score if downstream code
  needs physical units.
- **Resampling ratio** is approximated with
  `Fraction(target / sfreq).limit_denominator(1000)`, which is exact for
  250 → 200 Hz (4/5).
- **Notch only at the configured frequency** (50 Hz by default); harmonics such
  as 100 Hz are not removed automatically.
- **Preprocessing does not modify the LSL stream**: a processor only observes
  data. Store or forward the result yourself (the `on_window` callback).

## Tests

```bash
uv run pytest -q tests/test_preprocessing.py
```
