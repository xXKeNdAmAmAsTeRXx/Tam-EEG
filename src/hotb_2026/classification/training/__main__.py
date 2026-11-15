"""Command line entry point for one training run.

Examples
--------
Offline smoke test with synthetic recordings::

    uv run python -m hotb_2026.classification.training --synthetic --trials 20

Train on recorded sessions::

    uv run python -m hotb_2026.classification.training \
        --recordings data/sessions --out runs/nback --trials 60

Reuse a pre-built windowed dataset::

    uv run python -m hotb_2026.classification.training \
        --data runs/dataset.npz --out runs/nback
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from hotb_2026.classification.dataset import (
    EegDataset,
    build_dataset,
    load_recordings,
    make_synthetic_recording,
)
from hotb_2026.features import ReveEncoder
from hotb_2026.reporting import SUPPORTED_METRICS, write_run

from .config import TrainingConfig
from .trainer import train

log = logging.getLogger(__name__)


def _default_out() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("runs") / f"classification-{stamp}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m hotb_2026.classification.training",
        description=(
            "Train a REVE + SVM cognitive-overload classifier with grouped "
            "cross-validation and an Optuna hyperparameter search."
        ),
    )

    source = parser.add_argument_group("data source")
    source.add_argument(
        "--synthetic",
        action="store_true",
        help="generate synthetic recordings instead of loading real data",
    )
    source.add_argument(
        "--data",
        type=Path,
        help="pre-built EegDataset .npz file",
    )
    source.add_argument(
        "--recordings",
        type=Path,
        help="directory containing one session directory per recording",
    )

    window = parser.add_argument_group("windowing")
    window.add_argument(
        "--window", type=float, default=10.0, help="window length in seconds"
    )
    window.add_argument(
        "--stride", type=float, default=None, help="stride in seconds"
    )
    window.add_argument(
        "--group-by",
        choices=("auto", "session", "state_block"),
        default="auto",
        help="unit kept intact by cross-validation (ignored with --data)",
    )
    window.add_argument(
        "--keep-boundary-windows",
        action="store_true",
        help="keep windows whose label changes before the window ends",
    )

    training = parser.add_argument_group("training")
    training.add_argument("--trials", type=int, default=60, help="Optuna trials")
    training.add_argument(
        "--folds", type=int, default=5, help="cross-validation folds"
    )
    training.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="fraction of groups held out for the final score",
    )
    training.add_argument("--seed", type=int, default=0)
    training.add_argument(
        "--metric", choices=SUPPORTED_METRICS, default="balanced_accuracy"
    )
    training.add_argument(
        "--timeout", type=float, default=None, help="Optuna timeout in seconds"
    )

    encoder = parser.add_argument_group("encoder")
    encoder.add_argument(
        "--device",
        default=None,
        choices=("auto", "cpu", "mps", "cuda"),
        help="torch device for REVE (default: auto)",
    )
    encoder.add_argument("--batch-size", type=int, default=16)
    encoder.add_argument(
        "--no-cache", action="store_true", help="do not cache embeddings"
    )

    synthetic = parser.add_argument_group("synthetic data")
    synthetic.add_argument("--synthetic-sessions", type=int, default=10)
    synthetic.add_argument("--synthetic-duration", type=float, default=300.0)
    synthetic.add_argument("--state-duration", type=float, default=30.0)
    synthetic.add_argument("--synthetic-sfreq", type=float, default=250.0)

    output = parser.add_argument_group("output")
    output.add_argument(
        "--out", type=Path, default=None, help="run directory"
    )
    output.add_argument(
        "--save-dataset",
        action="store_true",
        help="write the windowed dataset next to the run artifacts",
    )
    output.add_argument(
        "--no-plots", action="store_true", help="skip figure generation"
    )
    output.add_argument(
        "-v", "--verbose", action="count", default=0, help="more logging"
    )
    return parser


def _load_dataset(args: argparse.Namespace) -> EegDataset:
    if args.data is not None:
        log.info("Loading dataset from %s", args.data)
        return EegDataset.load(args.data)

    if args.recordings is not None:
        log.info("Loading recordings from %s", args.recordings)
        recordings = load_recordings(str(args.recordings))
    elif args.synthetic:
        recordings = []
        for index in range(args.synthetic_sessions):
            log.info(
                "Generating synthetic session %d/%d",
                index + 1,
                args.synthetic_sessions,
            )
            recordings.append(
                make_synthetic_recording(
                    duration_s=args.synthetic_duration,
                    sfreq=args.synthetic_sfreq,
                    state_duration_s=args.state_duration,
                    seed=args.seed + index,
                    session_id=f"synthetic-{index + 1}",
                    subject_id="synthetic",
                )
            )
    else:
        raise SystemExit(
            "choose a data source: --synthetic, --data or --recordings"
        )

    return build_dataset(
        recordings,
        window_s=args.window,
        stride_s=args.stride,
        drop_boundary=not args.keep_boundary_windows,
        group_by=args.group_by,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if sum(item is not None for item in (args.data, args.recordings)) + int(
        args.synthetic
    ) != 1:
        parser.error("pass exactly one of --synthetic, --data or --recordings")

    level = logging.INFO if args.verbose == 0 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    out_dir = Path(args.out) if args.out is not None else _default_out()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_dataset(args)
    log.info(
        "Dataset: %s",
        ", ".join(
            f"{name}={count}"
            for name, count in dataset.summary()["class_counts"].items()
        ),
    )
    if args.save_dataset:
        dataset_path = out_dir / "dataset.npz"
        dataset.save(dataset_path)
        log.info("Saved dataset to %s", dataset_path)

    config = TrainingConfig(
        window_s=dataset.window_s if args.data is not None else args.window,
        stride_s=args.stride,
        n_splits=args.folds,
        n_trials=args.trials,
        seed=args.seed,
        test_size=args.test_size,
        metric=args.metric,
        group_by=args.group_by,
        drop_boundary=not args.keep_boundary_windows,
        timeout_s=args.timeout,
    )
    encoder = ReveEncoder(
        dataset.electrodes,
        device=args.device,
        batch_size=args.batch_size,
    )
    cache_dir = None if args.no_cache else out_dir / "embeddings"

    result = train(
        dataset,
        config=config,
        encoder=encoder,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    written = write_run(out_dir, result, make_plots=not args.no_plots)

    cv = result["cv"]
    holdout = result["holdout"]["metrics"]
    print()
    print(f"Run directory : {out_dir}")
    print(
        f"CV           : {cv['mean']:.3f} ± {cv['std']:.3f} "
        f"({cv['metric']}, {cv['n_splits']} folds, "
        f"{len(result['trials'])} trials)"
    )
    print(
        f"Held-out     : balanced_accuracy={holdout['balanced_accuracy']:.3f}, "
        f"accuracy={holdout['accuracy']:.3f}, "
        f"macro_f1={holdout['f1_macro']:.3f}"
    )
    print(f"Report       : {written['report']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
