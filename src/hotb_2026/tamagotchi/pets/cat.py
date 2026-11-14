"""The ``cat`` pet plugin: ASCII art + engine parameters."""

from __future__ import annotations

from hotb_2026.tamagotchi.pets import register_pet

register_pet(
    "cat",
    art={'EGG': {'HAPPY': '   _____\n  ( ovo )\n   -----', 'NEUTRAL': '   _____\n  ( ovo )\n   -----', 'SAD': '   _____\n  ( ovo )\n   -----', 'CRITICAL': '   _____\n  ( xvx )\n   -----'}, 'BABY': {'HAPPY': '  /\\_/\\\n ( ^w^ )\n', 'NEUTRAL': '  /\\_/\\\n ( -.- )\n', 'SAD': '  /\\_/\\\n ( ;_; )\n', 'CRITICAL': '  /\\_/\\\n ( x x )\n'}, 'ADULT': {'HAPPY': '  /\\_/\\_______\n ( ^w^ )      )~\n', 'NEUTRAL': '  /\\_/\\_______\n ( -.- )      )~\n', 'SAD': '  /\\_/\\_______\n ( ;_; )      )~\n', 'CRITICAL': '  /\\_/\\_______\n ( x x )      )~\n'}},
    params={
        "decay_rates": {'hunger': 8.0, 'energy': 6.0, 'hygiene': 5.0},
        "evolution_thresholds": {'BABY': {'seconds': 300.0, 'avg': 40.0}, 'ADULT': {'seconds': 3600.0, 'avg': 60.0}},
    },
)
