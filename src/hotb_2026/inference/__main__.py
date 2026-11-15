"""Simulated live inference from the command line.

Examples
--------
Classify a local bundle with the fake headset::

    uv run python -m hotb_2026.inference --bundle bundles/nback --windows 3

Download a pinned bundle from the Hugging Face Hub::

    uv run python -m hotb_2026.inference \
        --repo-id ai-med/hotb-2026-nback-classifier --revision 0.1.0

Each completed window is printed as one JSON object (a ``Prediction``).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

from hotb_2026.brainaccess_connection import (
    MONTAGE,
    EEGStreamer,
    SimulatedDriver,
)

from .predictor import Predictor

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m hotb_2026.inference",
        description=(
            "Stream simulated EEG through a saved model bundle and print one "
            "Prediction per completed window."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--bundle", type=Path, help="local bundle directory"
    )
    source.add_argument(
        "--repo-id",
        help="Hugging Face model repository, e.g. ai-med/nback-classifier",
    )

    parser.add_argument(
        "--revision",
        default=None,
        help="Hub tag, branch or commit to pin when using --repo-id",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token; defaults to HF_TOKEN or the cached login",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Hugging Face download cache",
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=3,
        help="number of predictions to collect before stopping",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="wall-clock limit in seconds (default: 10 + 2 * window_s per window)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device for REVE: auto, cpu, mps or cuda",
    )
    parser.add_argument(
        "--chunk-ms",
        type=float,
        default=40.0,
        help="simulated device chunk length",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="simulated device seed"
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="more logging"
    )
    return parser


def _load_predictor(args: argparse.Namespace) -> Predictor:
    # The simulator emits MONTAGE order; real recordings can use a subset or a
    # permutation of it.
    common = {
        "electrodes": MONTAGE.names,
        "device": args.device,
    }
    if args.bundle is not None:
        return Predictor.from_bundle(args.bundle, **common)
    return Predictor.from_pretrained(
        args.repo_id,
        revision=args.revision,
        token=args.token,
        cache_dir=args.cache_dir,
        **common,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.windows < 1:
        parser.error("--windows must be at least 1")
    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be positive")

    logging.basicConfig(
        level=logging.INFO if args.verbose == 0 else logging.DEBUG,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    predictor = _load_predictor(args)
    print(
        f"loaded {predictor.model_version} from {predictor.bundle_dir}; "
        f"window={predictor.window_s:g}s at {predictor.raw_sfreq:g} Hz, "
        f"classes={list(predictor.label_names)}",
        file=sys.stderr,
    )
    predictor.warmup()

    timeout = args.timeout
    if timeout is None:
        timeout = 10.0 + 2.0 * predictor.window_s * args.windows

    predictions = []
    failures: list[BaseException] = []

    def on_prediction(prediction) -> None:
        predictions.append(prediction)
        print(json.dumps(asdict(prediction), sort_keys=True), flush=True)

    adapter = predictor.processor(on_prediction=on_prediction)

    def guarded(window, t) -> None:
        # EEGStreamer logs and swallows processor errors; keep the first one so
        # the CLI can still fail with a useful exit status.
        try:
            adapter(window, t)
        except BaseException as exc:  # noqa: BLE001 - reported below
            failures.append(exc)
            raise

    driver = SimulatedDriver(
        MONTAGE, chunk_ms=args.chunk_ms, seed=args.seed
    )
    streamer = EEGStreamer(
        driver,
        MONTAGE,
        stream_name="HOTB inference (simulated)",
        source_id="hotb-inference-simulated",
        window_s=predictor.window_s + 1.0,
    )
    streamer.add_processor(guarded)

    deadline = time.monotonic() + timeout
    streamer.start()
    try:
        while len(predictions) < args.windows:
            if failures:
                break
            if streamer.error is not None:
                raise streamer.error
            if time.monotonic() > deadline:
                log.warning(
                    "stopped after %.1f s with %d/%d predictions",
                    timeout,
                    len(predictions),
                    args.windows,
                )
                break
            time.sleep(0.05)
    finally:
        streamer.stop()

    if failures:
        raise failures[0]
    if not predictions:
        print(
            "no windows were classified; increase --timeout or --windows",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
