from typing import List

from hotb_2026.brainaccess_connection import (
    Montage,
)

def montage_to_list_electrode(montage: Montage) -> list[str]:
    return [electrode.name for electrode in montage.electrodes]