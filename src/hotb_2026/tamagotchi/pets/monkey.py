from __future__ import annotations

from hotb_2026.tamagotchi.pets import register_pet

register_pet(
    name="monkey",
    art = {
        'EGG': {
            'HAPPY': '   _____\n  ( ovo )\n   -----',
            'NEUTRAL': '   _____\n  ( ovo )\n   -----',
            'SAD': '   _____\n  ( ovo )\n   -----',
            'CRITICAL': '   _____\n  ( xvx )\n   -----'
        },
        'BABY': {
            'HAPPY': '   _~_\n @( ^o^ )@\n',
            'NEUTRAL': '   _~_\n @( -.- )@\n',
            'SAD': '   _~_\n @( ;_; )@\n',
            'CRITICAL': '   _~_\n @( x x )@\n'
        },
        'ADULT': {
            'HAPPY': '    _~_\n  @( ^o^ )@  _\n   /(   )\\__/\n',
            'NEUTRAL': '    _~_\n  @( -.- )@  _\n   /(   )\\__/\n',
            'SAD': '    _~_\n  @( ;_; )@  _\n   /(   )\\__/\n',
            'CRITICAL': '    _~_\n  @( x x )@  _\n   /(   )\\__/\n'
        }
    },
    params={
        "decay_rates": {'hunger': 10.0, 'energy': 4.0, 'hygiene': 7.0},
        "evolution_thresholds": {'BABY': {'seconds': 60.0, 'avg': 20.0}, 'ADULT': {'seconds': 5000.0, 'avg': 90.0}},
    },
)