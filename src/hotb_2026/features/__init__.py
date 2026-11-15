"""Signal -> feature extraction shared by training and inference."""
from .cache import fingerprint, load_or_encode
from .reve import (
    DEFAULT_MODEL_NAME,
    DEFAULT_POS_MODEL_NAME,
    ReveEncoder,
    electrode_names,
    resolve_device,
    validate_batch,
)

__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_POS_MODEL_NAME",
    "ReveEncoder",
    "electrode_names",
    "fingerprint",
    "load_or_encode",
    "resolve_device",
    "validate_batch",
]
