# BrainAccess Connection

Connect to a BrainAccess MINI EEG headset over Bluetooth LE and publish the raw
signal as an [LSL](https://labstreaminglayer.readthedocs.io/) stream, so any
LSL-aware tool (LabRecorder, OpenViBE, MNE, custom consumers) can record or
consume it.

```text
headset ──BLE──▶ BrainAccess SDK ──▶ BrainAccessDriver ──▶ EEGStreamer ──▶ LSL
                                                          │
                                                          └──▶ processors
```

The package also ships a `SimulatedDriver` that generates a real-time fake
signal, so the rest of the system can be developed without a headset.

## Hardware

Target device: **BrainAccess MINI (V2)**.

| Property | Value |
| --- | --- |
| EEG inputs | 8 (`0`–`7`) |
| Sample rates | 250 Hz or 500 Hz |
| Reference | Dedicated **R** connector |
| Bias | Dedicated **B** connector |

`R` and `B` are wired electrodes, not stream channels. The default montage
follows the SDK's 8-channel 10-20 layout:

```text
pin 0..7 → F3, F4, C3, C4, P3, P4, O1, O2
```

Change the labels in [`config.py`](config.py) if your cap is wired differently.
Bias feedback is selected in software with `bias_electrode` (default: electrode
`7`, `O2`); the physical bias electrode is always the `B` connector.

## Requirements

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/)
- A BrainAccess MINI, switched on and not connected to another computer or phone
- Linux: a working BlueZ stack (`bluetoothd` running, adapter not blocked)
- macOS / Windows: system Bluetooth enabled

## Install

From the repository root:

```bash
uv sync
```

This installs the BrainAccess SDK, `pylsl`, and the `hotb-brainaccess` command.
No separate LSL installation is required for streaming.

## Configuration

Project defaults live in [`config.py`](config.py) — that is the only file you
normally need to edit:

| Setting | Config attribute | CLI override | Default |
| --- | --- | --- | --- |
| Headset name | `DEVICE_NAME` | `--device` | `BA MINI 078` |
| Electrode layout | `MONTAGE.electrodes` | — | `F3, F4, C3, C4, P3, P4, O1, O2` on pins 0–7 |
| Stream rate | `MONTAGE.sfreq` | `--sfreq` | `200` Hz |
| Bias electrode | `MONTAGE.bias_electrode` | `--bias` | `7` (0-based) |

Precedence is **explicit CLI flag → `config.py` → built-in `DEFAULT_MONTAGE`**,
so `uv run hotb-brainaccess` uses the configured headset and montage without
extra flags.

> **Sample rate:** the BrainAccess MINI hardware supports only **250 Hz** or
> **500 Hz**. The shipped config uses `200` Hz so `--simulate` runs at 200 Hz,
> which is why hardware runs need `--sfreq 250` (or `500`). The driver rejects
> other rates before connecting and tells you what to pass.

`DEFAULT_MONTAGE` in `models/montage.py` is the library's 250 Hz baseline.
`MONTAGE` in `config.py` is this project's configuration: edit it freely, but
keep `sfreq` at 250 or 500 when you use a real headset.

## Quick start

1. Scan for nearby headsets. The call blocks for about 5 seconds:

   ```bash
   uv run hotb-brainaccess --scan
   ```

   ```text
   BA MINI 001
   ```

2. Stream the configured headset (`DEVICE_NAME` in `config.py`) at a rate the
   hardware supports:

   ```bash
   uv run hotb-brainaccess --sfreq 250
   ```

   Pass `--device "BA MINI 001"` to override the configured name. The process
   prints per-channel RMS once per second and runs until `Ctrl+C`.

3. Verify the stream from another terminal:

   ```bash
   uv run python - <<'PY'
   from pylsl import StreamInlet, resolve_byprop

   streams = resolve_byprop("name", "BrainAccess EEG", timeout=5)
   inlet = StreamInlet(streams[0])
   info = inlet.info()
   print(info.name(), info.type(), info.channel_count(), info.nominal_srate())

   chunk, timestamps = inlet.pull_chunk(timeout=2.0, max_samples=64)
   print(len(chunk), "samples x", len(chunk[0]), "channels")
   PY
   ```

   `resolve_byprop` returns stream metadata without the full XML description;
   use `inlet.info().desc()` if you need channel labels.

## Simulated headset

`--simulate` replaces the device with a fake signal (10 Hz alpha plus noise)
paced in real time at `MONTAGE.sfreq` — 200 Hz by default:

```bash
uv run hotb-brainaccess --simulate
```

Use it to develop LSL consumers, the REVE processor, or the state-aware agent
without tying up the headset. Simulation accepts any positive rate, e.g.
`--simulate --sfreq 500`.

## CLI reference

```text
hotb-brainaccess [--device NAME | --simulate | --scan] [options]
```

| Option | Default | Description |
| --- | --- | --- |
| `--device NAME` | `DEVICE_NAME` (`BA MINI 078`) | Exact advertised name, e.g. `"BA MINI 001"`. Case and spaces matter. Overrides `config.py`. |
| `--scan` | off | Scan for headsets, print one name per line, exit. Blocks ~5 s. |
| `--simulate` | off | Use the simulated driver instead of hardware. Accepts any `--sfreq`. |
| `--adapter N` | system default | Linux only: use Bluetooth adapter `hci<N>` (e.g. `--adapter 1`). Applies to `--scan` too. |
| `--name NAME` | `BrainAccess EEG` | LSL stream name. |
| `--source-id ID` | `brainaccess-eeg-1` | LSL `source_id`; use a unique value per headset. |
| `--window SECONDS` | `4.0` | Maximum rolling-window length passed to processors. Does not affect the LSL stream. |
| `--sfreq HZ` | `MONTAGE.sfreq` (`200`) | Stream rate. Hardware supports 250 or 500 Hz only; simulation accepts any positive rate. |
| `--chunk-size N` | `25` | Samples per BLE chunk, 1–100. Lower = lower latency, more overhead. |
| `--gain N` | `8` | Amplifier gain: 1, 2, 4, 6, 8 or 12. |
| `--bias N\|none` | `MONTAGE.bias_electrode` (`7`) | 0-based EEG electrode used for bias feedback. `none` disables feedback. |
| `--log-level LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`. |

Examples:

```bash
# Stream the configured headset at 250 Hz (hardware-supported)
uv run hotb-brainaccess --sfreq 250

# Use adapter hci1, custom stream identity, 500 Hz, smaller chunks
uv run hotb-brainaccess --device "BA MINI 001" --adapter 1 --sfreq 500 \
  --name "Subject 07" --source-id subject-07 --chunk-size 20

# Scan on a specific adapter
uv run hotb-brainaccess --scan --adapter 1

# Disable bias feedback
uv run hotb-brainaccess --device "BA MINI 001" --sfreq 250 --bias none
```

The module form works as well:

```bash
uv run python -m hotb_2026.brainaccess_connection --simulate
```

### Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Clean shutdown after `Ctrl+C` |
| `1` | Device or stream failure (connect, BLE drop, SDK error) |
| `2` | Invalid arguments |

## LSL stream layout

| Property | Value |
| --- | --- |
| Name | `--name` (default `BrainAccess EEG`) |
| Type | `EEG` |
| Channel count | Number of montage electrodes (8 by default) |
| Channel format | `float32` |
| Nominal rate | `MONTAGE.sfreq` (200 Hz in the shipped config; 250/500 on hardware) |
| Unit | µV |
| `source_id` | `--source-id` |

Channel labels follow the montage order, e.g.
`F3, F4, C3, C4, P3, P4, O1, O2`. Reference and bias connectors are **not**
published as data channels.

Chunks are pushed to LSL as they arrive. A BLE disconnect stops the streamer
and is reported as an error instead of silently repeating the last data.

## Using it as a library

`DEVICE_NAME` and `MONTAGE` are the same values the CLI uses. The driver
requires a hardware-supported rate, so replace `MONTAGE.sfreq` when using a real
headset:

```python
from dataclasses import replace

from hotb_2026.brainaccess_connection import (
    DEVICE_NAME,
    MONTAGE,
    BrainAccessDriver,
    EEGStreamer,
)

montage = replace(MONTAGE, sfreq=250.0)  # hardware: 250 or 500
driver = BrainAccessDriver(DEVICE_NAME, montage)
streamer = EEGStreamer(
    driver,
    montage,
    stream_name="BrainAccess EEG",
    source_id="brainaccess-eeg-1",
    window_s=4.0,
)

streamer.add_processor(lambda window, t: print(t, window.shape))
streamer.start()
try:
    while streamer.running:
        ...  # do work; callbacks run on the streamer thread
finally:
    streamer.stop()

if streamer.error:
    raise RuntimeError(streamer.error)
```

For a self-contained 250 Hz default that ignores `config.py`, use
`DEFAULT_MONTAGE` from `models/montage.py` instead of `MONTAGE`.

Processors receive `(window, t)` where:

- `window` is a private copy with shape `(n_channels, n_samples)` in montage
  order. Before the rolling window is full, `n_samples` is smaller than
  `window_s * sfreq`; check `streamer.is_full`.
- `t` is the stream time in seconds since `start()`.
- Exceptions from one processor are logged and do not stop the EEG stream.

### Custom montage

Editing `MONTAGE` in [`config.py`](config.py) is the normal way to change the
layout. For programmatic use, build a `Montage` directly:

```python
from hotb_2026.brainaccess_connection import Electrode, Montage

montage = Montage(
    electrodes=(
        Electrode("F3", pin=0),
        Electrode("F4", pin=1),
    ),
    sfreq=250.0,       # hardware: 250 or 500
    bias_electrode=1,  # 0-based index into electrodes, or None to disable
)
```

`pin` is the 0-based electrode input reported by the headset, not a label
printed on the cap. At `open()` the driver checks every pin against
`electrode_count()` and refuses to start on a mismatch.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `No devices found` | Headset is off, out of range, or already connected to another machine/phone. Power-cycle it. |
| Device missing from `--scan` | Renamed headsets are not advertised as `BA ...`; connect using the current advertised name instead. |
| `Could not connect` | Close the vendor app or any other BLE client holding the device. On Linux try `bluetoothctl` or a different adapter. |
| `Bluetooth cannot be used` | Start `bluetoothd`, unblock the adapter (`rfkill unblock bluetooth`), check permissions. |
| `BrainAccess SDK missing` | Run `uv sync` from the repository root. |
| `pylsl is missing` | Run `uv sync`; `pylsl` is a declared dependency. |
| `Montage uses electrode N, but the headset has M` | Fix `pin` values; the driver will not start with a wrong mapping. |
| `BrainAccess MINI supports 250 or 500 Hz, not 200 Hz` | The shipped config is simulation-only. Pass `--sfreq 250` (or `500`), or use `--simulate`. |
| `MINI_V2 does not support 500 Hz sample rate` | Use `--sfreq 250`; check the model's supported rates. |
| LSL stream never appears | Check the process is still running and the consumer resolves the same `--name`. |
| RMS lines stop printing | Check `--log-level DEBUG` for a disconnect or queue-overrun warning. |
| `bacore.log` appears in the project root | The BrainAccess SDK writes its log to the current directory. It is git-ignored; delete it whenever you like. |

## Package layout

```text
src/hotb_2026/brainaccess_connection/
├── __init__.py       # public exports
├── __main__.py       # python -m hotb_2026.brainaccess_connection
├── config.py         # DEVICE_NAME and MONTAGE: the project defaults
├── main.py           # CLI
├── drivers/
│   ├── __init__.py
│   └── driver.py     # Driver protocol, BrainAccessDriver, SimulatedDriver
├── stream/
│   ├── __init__.py
│   └── streamer.py   # EEGStreamer: polling thread, rolling window, LSL outlet
└── models/
    ├── __init__.py
    └── montage.py    # Electrode, Montage, DEFAULT_MONTAGE
```

The streamer depends only on the `Driver` protocol, so simulated and future
devices plug in without changing the LSL or processing code.

## Tests

```bash
uv run pytest -q
```

The suite covers montage validation, chunk mapping, disconnect handling, ring
buffer behaviour, processor isolation, config defaults, config-derived CLI
defaults, and the hardware sample-rate guard.
