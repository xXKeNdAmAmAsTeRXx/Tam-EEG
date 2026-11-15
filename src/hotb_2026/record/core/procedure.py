"""Randomized n-back procedure generation."""

from __future__ import annotations

import random
from typing import Any

from .settings import BLOCK_TYPES, Block, MAX_BLOCK_DURATION, MIN_BLOCK_DURATION, N_BLOCKS


def make_procedure(
    rng: random.Random | None = None,
    *,
    n_blocks: int = N_BLOCKS,
    min_duration: int = MIN_BLOCK_DURATION,
    max_duration: int = MAX_BLOCK_DURATION,
    block_types: list[dict[str, Any]] | None = None,
) -> list[Block]:
    """Build a randomized list of blocks (sampling with replacement)."""
    if rng is None:
        rng = random.Random()
    if block_types is None:
        block_types = BLOCK_TYPES
    if n_blocks < 1:
        raise ValueError("n_blocks must be >= 1")
    if min_duration > max_duration:
        raise ValueError("min_duration must be <= max_duration")

    blocks: list[Block] = []
    for i in range(n_blocks):
        template = rng.choice(block_types).copy()
        duration = rng.randint(min_duration, max_duration)
        blocks.append(
            Block(
                name=template["name"],
                n=template["n"],
                duration=float(duration),
                occurrence=i + 1,
            )
        )
    return blocks
