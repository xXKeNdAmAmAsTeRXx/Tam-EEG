from __future__ import annotations

import random

from hotb_2026.record import make_procedure
from hotb_2026.record.settings import MIN_BLOCK_DURATION, MAX_BLOCK_DURATION


def test_procedure_length_and_bounds():
    rng = random.Random(1)
    blocks = make_procedure(rng=rng, n_blocks=5, min_duration=10, max_duration=20)
    assert len(blocks) == 5
    for b in blocks:
        assert 10 <= b.duration <= 20
        assert b.name in {"rest", "1back", "3back"}


def test_procedure_reproducible():
    a = make_procedure(rng=random.Random(42), n_blocks=3)
    b = make_procedure(rng=random.Random(42), n_blocks=3)
    assert a == b


def test_procedure_defaults():
    blocks = make_procedure(rng=random.Random(0))
    assert len(blocks) == 3
    for b in blocks:
        assert MIN_BLOCK_DURATION <= b.duration <= MAX_BLOCK_DURATION
