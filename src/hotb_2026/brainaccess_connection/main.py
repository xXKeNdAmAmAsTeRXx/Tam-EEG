from __future__ import annotations

import argparse
import dataclasses
import logging
import time

import numpy as np

from . import (
    DEVICE_NAME,
    MONTAGE,
    BrainAccessDriver,
    DeviceError,
    EEGStreamer,
    SimulatedDriver,
    list_devices,
)

log = logging.getLogger(__name__)


def _bias_value(text: str) -> int | None:
    if text.lower() in {"none", "off", "no"}:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "bias must be an electrode index (0-7) or 'none'"
        ) from exc


def _device_name(args: argparse.Namespace) -> str:
    """Headset name: an explicit ``--device`` wins over ``config.py``."""
    return args.device or DEVICE_NAME


def _warn_ignored_simulate_options(args: argparse.Namespace) -> None:
    if not args.simulate:
        return
    if args.device:
        log.warning("--simulate given; ignoring --device %r", args.device)
    if args.adapter is not None:
        log.warning("--simulate given; ignoring --adapter")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream BrainAccess MINI EEG to LSL over BLE"
    )
    parser.add_argument(
        "--device",
        help=(
            'Headset name as printed on it, e.g. "BA MINI 001" '
            f"(default from config.py: {DEVICE_NAME!r})"
        ),
    )
    parser.add_argument(
        "--adapter", type=int, help="Bluetooth adapter index (Linux only, hci<N>)"
    )
    parser.add_argument("--scan", action="store_true", help="List nearby headsets and exit")
    parser.add_argument("--simulate", action="store_true", help="Use the fake headset")
    parser.add_argument("--name", default="BrainAccess EEG", help="LSL stream name")
    parser.add_argument(
        "--source-id", default="brainaccess-eeg-1", help="LSL source_id"
    )
    parser.add_argument(
        "--window", type=float, default=4.0, help="Rolling window length (s)"
    )
    parser.add_argument(
        "--sfreq",
        type=int,
        default=round(MONTAGE.sfreq),
        help=(
            "Stream sample rate in Hz "
            f"(default from config.py: {MONTAGE.sfreq:g}; "
            "MINI hardware: 250 or 500)"
        ),
    )
    parser.add_argument(
        "--chunk-size", type=int, default=25, help="Samples per BLE chunk (1-100)"
    )
    parser.add_argument(
        "--gain",
        type=int,
        choices=(1, 2, 4, 6, 8, 12),
        default=8,
        help="Amplifier gain (default: 8)",
    )
    parser.add_argument(
        "--bias",
        type=_bias_value,
        default=MONTAGE.bias_electrode,
        metavar="N|none",
        help=(
            "0-based electrode used for bias feedback "
            f"(default from config.py: {MONTAGE.bias_electrode})"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level (default: INFO)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.scan:
        try:
            names = list_devices(adapter_index=args.adapter)
        except DeviceError as exc:
            raise SystemExit(str(exc)) from exc
        for name in names:
            print(name)
        return

    device = _device_name(args)
    if not args.simulate and not device:
        parser.error("Pass --device, --simulate, or --scan")
    _warn_ignored_simulate_options(args)

    try:
        montage = dataclasses.replace(
            MONTAGE, bias_electrode=args.bias, sfreq=float(args.sfreq)
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not 1 <= args.chunk_size <= 100:
        parser.error("--chunk-size must be in 1..100")

    try:
        driver = (
            SimulatedDriver(montage)
            if args.simulate
            else BrainAccessDriver(
                device,
                montage,
                adapter_index=args.adapter,
                chunk_size=args.chunk_size,
                gain=args.gain,
            )
        )
    except DeviceError as exc:
        raise SystemExit(str(exc)) from exc

    streamer = EEGStreamer(
        driver,
        montage,
        stream_name=args.name,
        source_id=args.source_id,
        window_s=args.window,
    )

    last_print = [0.0]

    def rms_monitor(window: np.ndarray, t: float) -> None:
        if not streamer.is_full or t - last_print[0] < 1.0:
            return
        last_print[0] = t
        rms = np.sqrt((window ** 2).mean(axis=1))
        print(
            "  ".join(
                f"{name}:{value:6.1f}"
                for name, value in zip(montage.eeg_names, rms)
            ),
            flush=True,
        )

    streamer.add_processor(rms_monitor)
    try:
        streamer.start()
    except (DeviceError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    try:
        while streamer.running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        streamer.stop()

    if streamer.error:
        raise SystemExit(f"Streamer failed: {streamer.error}")


if __name__ == "__main__":
    main()
