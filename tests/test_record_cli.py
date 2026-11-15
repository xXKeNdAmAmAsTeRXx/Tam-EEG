from __future__ import annotations

from hotb_2026.record.cli import build_parser


def test_cli_defaults():
    args = build_parser().parse_args([])
    assert args.sessions == 1
    assert args.out_dir == "data"
    assert args.simulate is False
    assert args.n_blocks == 3
