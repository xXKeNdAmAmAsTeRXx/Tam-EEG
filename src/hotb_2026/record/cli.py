"""Command-line entry point for the record package."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from .api import run_experiment
from .io.paths import ensure_sessions_available, subject_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run recorded n-back EEG sessions",
        allow_abbrev=False,
    )
    parser.add_argument("--subject", default="", help="Subject/participant id")
    parser.add_argument(
        "--session",
        "--session-idx",
        dest="session_idx",
        type=int,
        default=None,
        help="Session index to record (default: 1, or the value from the dialog)",
    )
    parser.add_argument(
        "--sessions", type=int, default=1, help="Number of sessions to record"
    )
    parser.add_argument("--out-dir", default="data", help="Output root directory")
    parser.add_argument("--simulate", action="store_true", help="Use simulated EEG driver")
    parser.add_argument("--device", default=None, help="Headset name (real device)")
    parser.add_argument("--adapter", type=int, default=None, help="Bluetooth adapter index")
    parser.add_argument("--sfreq", type=int, default=None, help="Sample rate Hz")
    parser.add_argument("--chunk-size", type=int, default=25, help="BLE chunk size")
    parser.add_argument("--gain", type=int, default=8, help="Amplifier gain")
    parser.add_argument("--bias", default=None, help="Bias electrode index or 'none'")
    parser.add_argument("--fullscreen", action="store_true", default=True)
    parser.add_argument("--windowed", action="store_true", help="Run in a window")
    parser.add_argument("--min-duration", type=int, default=30)
    parser.add_argument("--max-duration", type=int, default=90)
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.sessions < 1:
        raise SystemExit("--sessions must be >= 1")

    session_idx = args.session_idx
    subject = args.subject
    if not subject:
        try:
            from psychopy import gui

            info = {"participant": "", "session": "001"}
            dlg = gui.DlgFromDict(info, title="N-back recording")
            if not dlg.OK:
                raise SystemExit(0)
            subject = info["participant"] or "subject"
            if session_idx is None:
                try:
                    session_idx = int(info.get("session", "1"))
                except (TypeError, ValueError):
                    session_idx = 1
        except Exception:  # noqa: BLE001
            raise SystemExit("--subject is required")

    if session_idx is None:
        session_idx = 1
    if session_idx < 1:
        raise SystemExit("--session must be >= 1")

    out_root = Path(args.out_dir)
    subject_root = subject_dir(out_root, subject)
    try:
        ensure_sessions_available(subject_root, session_idx, args.sessions)
    except FileExistsError as exc:
        raise SystemExit(str(exc)) from exc

    last_session_idx = session_idx + args.sessions - 1
    print(
        f"Recording session(s) {session_idx}..{last_session_idx} "
        f"for subject '{subject}' into {subject_root}"
    )

    from hotb_2026.brainaccess_connection import DEVICE_NAME, MONTAGE

    device = args.device or DEVICE_NAME
    fullscreen = args.fullscreen and not args.windowed

    # Build montage.
    montage = MONTAGE
    if args.sfreq is not None:
        montage = dataclasses.replace(montage, sfreq=float(args.sfreq))

    if args.bias is not None:
        bias = None if str(args.bias).lower() in {"none", "off", "no"} else int(args.bias)
        montage = dataclasses.replace(montage, bias_electrode=bias)

    if args.simulate:
        from hotb_2026.brainaccess_connection import SimulatedDriver

        driver = SimulatedDriver(montage)
    else:
        from hotb_2026.brainaccess_connection import BrainAccessDriver

        driver = BrainAccessDriver(
            device,
            montage,
            adapter_index=args.adapter,
            chunk_size=args.chunk_size,
            gain=args.gain,
        )

    if not args.simulate:
        print(f"Connecting to {device} ... (device connection occurs when the experiment window opens)")

    from hotb_2026.brainaccess_connection import DeviceError

    try:
        run_experiment(
            subject_id=subject,
            n_sessions=args.sessions,
            start_session_idx=session_idx,
            driver=driver,
            montage=montage,
            out_dir=args.out_dir,
            simulate=args.simulate,
            seed=args.seed,
            fullscreen=fullscreen,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            n_blocks=args.n_blocks,
        )
    except (DeviceError, KeyboardInterrupt) as exc:
        print(f"Recording stopped: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
