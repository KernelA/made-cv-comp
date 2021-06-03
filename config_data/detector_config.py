from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class TrainDetectorConfig:
    batch_size: int = MISSING
