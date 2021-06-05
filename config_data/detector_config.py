from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class TrainDetectorConfig:
    train_batch_size: int = MISSING
    valid_batch_size: int = MISSING
