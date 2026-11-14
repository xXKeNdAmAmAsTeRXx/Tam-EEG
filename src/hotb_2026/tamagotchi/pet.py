"""Pet model: stats, life stages and JSON persistence."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum, unique
from pathlib import Path

from hotb_2026.tamagotchi.constants import (
    CRITICAL_THRESHOLD,
    STAT_MAX,
    STAT_MIN,
    PET_FILE_NAME,
    state_dir,
)


@unique
class Stage(str, Enum):
    """Life stages, in evolution order."""

    EGG = "EGG"
    BABY = "BABY"
    ADULT = "ADULT"

    @property
    def next(self) -> Stage | None:
        stages = list(Stage)
        idx = stages.index(self)
        return stages[idx + 1] if idx + 1 < len(stages) else None


@unique
class Mood(str, Enum):
    """Resolved from stats by the engine, never stored long-term."""

    HAPPY = "HAPPY"
    NEUTRAL = "NEUTRAL"
    SAD = "SAD"
    CRITICAL = "CRITICAL"


def clamp(value: float) -> float:
    """Clamp a stat into ``[STAT_MIN, STAT_MAX]``."""
    return max(STAT_MIN, min(STAT_MAX, float(value)))


@dataclass
class Stats:
    """Vital statistics of a pet, each clamped to 0–100."""

    happiness: float = 50.0
    hunger: float = 80.0
    energy: float = 80.0
    hygiene: float = 80.0

    def clamp_all(self) -> None:
        for name in ("happiness", "hunger", "energy", "hygiene"):
            setattr(self, name, clamp(getattr(self, name)))

    @property
    def average(self) -> float:
        return sum(
            getattr(self, name)
            for name in ("happiness", "hunger", "energy", "hygiene")
        ) / 4.0

    def is_critical(self) -> bool:
        """True when any vital stat (except happiness) drops to the threshold."""
        return any(
            getattr(self, name) <= CRITICAL_THRESHOLD
            for name in ("hunger", "energy", "hygiene")
        )


@dataclass
class Pet:
    """Persisted pet state."""

    name: str
    pet_kind: str
    stage: Stage = Stage.EGG
    stats: Stats = field(default_factory=Stats)
    #: Unix timestamp of the last engine update (decay/mood/evolution/save).
    last_update: float = field(default_factory=time.time)
    #: Unix timestamp of when the current stage was entered.
    stage_started_at: float = field(default_factory=time.time)

    # -- persistence -----------------------------------------------------

    def to_dict(self) -> dict:
        data = asdict(self)
        data["stage"] = self.stage.value
        return data

    def save(self, directory: Path | None = None) -> Path:
        target_dir = Path(directory) if directory else state_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / PET_FILE_NAME
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict) -> Pet:
        return cls(
            name=data["name"],
            pet_kind=data["pet_kind"],
            stage=Stage(data["stage"]),
            stats=Stats(**data["stats"]),
            last_update=float(data["last_update"]),
            stage_started_at=float(data["stage_started_at"]),
        )

    @classmethod
    def load(cls, directory: Path | None = None) -> Pet:
        path = (Path(directory) if directory else state_dir()) / PET_FILE_NAME
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def load_or_none(cls, directory: Path | None = None) -> Pet | None:
        try:
            return cls.load(directory)
        except (FileNotFoundError, KeyError, ValueError):
            return None
