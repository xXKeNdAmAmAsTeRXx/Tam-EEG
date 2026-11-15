"""Constants for the n-back recording procedure."""

from __future__ import annotations

import random
from dataclasses import dataclass

LETTERS = list("ABCDEHIKLMOPRST")

STIM_DURATION = 0.5
ITI = 2.5
TRIAL_DURATION = STIM_DURATION + ITI

RESPONSE_KEY = "m"
MATCH_PROB = 1 / 3

MIN_BLOCK_DURATION = 30
MAX_BLOCK_DURATION = 90

N_BLOCKS = 3

SHOW_FEEDBACK_DEFAULT = True
FULLSCREEN_DEFAULT = True

LETTER_HEIGHT = 0.15
BG_COLOR = "grey"

BLOCK_TYPES = [
    {"name": "rest", "n": None},
    {"name": "1back", "n": 1},
    {"name": "3back", "n": 3},
]


@dataclass(frozen=True)
class Block:
    name: str
    n: int | None
    duration: float
    occurrence: int
