from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class TrainOCRConfig:
    train_batch_size: int = MISSING
    valid_batch_size: int = MISSING
    image_dir: str = MISSING
