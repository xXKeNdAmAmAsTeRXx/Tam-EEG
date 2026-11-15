"""Command line entry point for model bundles.

Examples
--------
Inspect a bundle without uploading::

    uv run python -m hotb_2026.model_store show bundles/nback

Publish it (public repository under the configured owner)::

    HF_TOKEN=... uv run python -m hotb_2026.model_store publish bundles/nback \
        --repo-id ai-med/hotb-2026-nback-classifier

Saving a bundle is a Python call, because the fitted pipeline lives in the
dict returned by training::

    from hotb_2026.model_store import save_model
    save_model(result, "bundles/nback", run_dir="runs/nback")
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .bundle import read_metadata
from .config import REPO_ID_ENV, StoreConfig
from .publish import publish


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m hotb_2026.model_store",
        description="Inspect and publish saved cognitive-overload models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    show = subparsers.add_parser("show", help="print bundle metadata as JSON")
    show.add_argument("bundle", type=Path)

    push = subparsers.add_parser(
        "publish", help="upload a bundle to the Hugging Face Hub"
    )
    push.add_argument("bundle", type=Path)
    push.add_argument(
        "--repo-id",
        default=None,
        help=f"owner/name; defaults to ${REPO_ID_ENV}",
    )
    push.add_argument(
        "--private",
        action="store_true",
        help="create a private repository",
    )
    push.add_argument(
        "--no-tag",
        action="store_true",
        help="do not create a git tag for the bundle version",
    )
    push.add_argument(
        "--no-fallback",
        action="store_true",
        help="fail instead of publishing under the authenticated user",
    )
    push.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="more logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "show":
        metadata = read_metadata(args.bundle)
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return 0

    logging.basicConfig(
        level=logging.INFO if args.verbose == 0 else logging.DEBUG,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config = StoreConfig(repo_id=args.repo_id, private=args.private)
    summary = publish(
        args.bundle,
        config,
        fallback_to_user=not args.no_fallback,
        tag=not args.no_tag,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
