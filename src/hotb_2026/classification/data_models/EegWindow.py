from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class EegWindow:
    electrodes: List[str]
    signal: np.ndarray[float]