from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Electrode:
    """One measured EEG input of the headset.

    ``pin`` is the 0-based electrode input index reported by the device
    (``0..electrode_count - 1``). Reference and bias connectors are not
    electrodes and must not be listed here.
    """

    name: str
    pin: int


@dataclass(frozen=True)
class Montage:
    """Channel layout of the EEG stream.

    Columns of every device read follow the order of ``electrodes``. The
    reference is a physical connector; ``bias_electrode`` selects which measured
    electrode is used for bias feedback.
    """

    electrodes: tuple[Electrode, ...]
    sfreq: float = 250.0
    bias_electrode: int | None = 7

    def __post_init__(self) -> None:
        if not self.electrodes:
            raise ValueError("Montage must contain at least one electrode")
        names = [e.name for e in self.electrodes]
        pins = [e.pin for e in self.electrodes]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate electrode names")
        if len(set(pins)) != len(pins):
            raise ValueError("Duplicate device pins")
        if any(pin < 0 for pin in pins):
            raise ValueError("Electrode pins must be non-negative")
        if self.sfreq <= 0:
            raise ValueError("sfreq must be positive")
        if self.bias_electrode is not None and not (
            0 <= self.bias_electrode < len(self.electrodes)
        ):
            raise ValueError(
                f"bias_electrode must be in 0..{len(self.electrodes) - 1} "
                f"or None, got {self.bias_electrode}"
            )

    @property
    def names(self) -> list[str]:
        return [e.name for e in self.electrodes]

    # Kept as an alias: every column of this montage is an EEG channel.
    @property
    def eeg_names(self) -> list[str]:
        return self.names

    @property
    def eeg_columns(self) -> list[int]:
        return list(range(len(self.electrodes)))

    @property
    def n_channels(self) -> int:
        return len(self.electrodes)


# BrainAccess MINI: 8 EEG inputs (0-7) plus dedicated R (reference) and
# B (bias) connectors. Labels follow the SDK's default 8-channel 10-20 layout;
# change them if your cap is wired differently.
DEFAULT_MONTAGE = Montage(
    electrodes=(
        Electrode("F3", pin=0),
        Electrode("F4", pin=1),
        Electrode("C3", pin=2),
        Electrode("C4", pin=3),
        Electrode("P3", pin=4),
        Electrode("P4", pin=5),
        Electrode("O1", pin=6),
        Electrode("O2", pin=7),
    ),
    sfreq=250.0,
    bias_electrode=7,
)
