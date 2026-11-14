"""The ``slime`` pet plugin: ASCII art + engine parameters."""

from __future__ import annotations

from hotb_2026.tamagotchi.pets import register_pet

register_pet(
    "slime",
    art={'EGG': {'HAPPY': "   .-.\n  ( o )\n   '-'", 'NEUTRAL': "   .-.\n  ( o )\n   '-'", 'SAD': "   .-.\n  ( o )\n   '-'", 'CRITICAL': "   .-.\n  ( x )\n   '-'"}, 'BABY': {'HAPPY': "  .-~-.\n ( ^w^ )\n  '~ ~'", 'NEUTRAL': "  .-~-.\n ( -.- )\n  '~ ~'", 'SAD': "  .-~-.\n ( ;.; )\n  '~ ~'", 'CRITICAL': "  .-~-.\n ( x.x )\n  '~ ~'"}, 'ADULT': {'HAPPY': "  .-~~~~-.\n (  ^w^   )\n  '~~~~~~~'", 'NEUTRAL': "  .-~~~~-.\n (  -.-   )\n  '~~~~~~~'", 'SAD': "  .-~~~~-.\n (  ;.;   )\n  '~~~~~~~'", 'CRITICAL': "  .-~~~~-.\n (  x.x   )\n  '~~~~~~~'"}},
    params={
        "decay_rates": {'hunger': 4.0, 'energy': 4.0, 'hygiene': 3.0},
        "evolution_thresholds": {'BABY': {'seconds': 300.0, 'avg': 35.0}, 'ADULT': {'seconds': 3600.0, 'avg': 50.0}},
    },
)
