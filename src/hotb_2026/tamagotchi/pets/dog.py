"""The ``dog`` pet plugin: ASCII art + engine parameters."""

from __future__ import annotations

from hotb_2026.tamagotchi.pets import register_pet

register_pet(
    "dog",
    art={'EGG': {'HAPPY': "   .-----.\n  (  ovo  )\n   '-----'", 'NEUTRAL': "   .-----.\n  (  ovo  )\n   '-----'", 'SAD': "   .-----.\n  (  ovo  )\n   '-----'", 'CRITICAL': "   .-----.\n  (  xvx  )\n   '-----'"}, 'BABY': {'HAPPY': '  / \\__\n (    @\\\n', 'NEUTRAL': '  / \\__\n (    -\\\n', 'SAD': '  / \\__\n (    ;\\\n', 'CRITICAL': '  / \\__\n (    x\\\n'}, 'ADULT': {'HAPPY': '  / \\_____/ \\\n ( ^w^      )\n', 'NEUTRAL': '  / \\_____/ \\\n ( -.-      )\n', 'SAD': '  / \\_____/ \\\n ( ;_;      )\n', 'CRITICAL': '  / \\_____/ \\\n ( x x      )\n'}},
    params={
        "decay_rates": {'hunger': 10.0, 'energy': 8.0, 'hygiene': 6.0},
        "evolution_thresholds": {'BABY': {'seconds': 300.0, 'avg': 40.0}, 'ADULT': {'seconds': 3600.0, 'avg': 60.0}},
    },
)
