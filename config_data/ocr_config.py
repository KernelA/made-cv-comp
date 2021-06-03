from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class TrainOCRConfig:
    batch_size: int = MISSING
