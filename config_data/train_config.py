from dataclasses import dataclass
from typing import Any, Optional

from omegaconf import MISSING

from .detector_config import TrainDetectorConfig
from .ocr_config import TrainOCRConfig


@dataclass
class TrainConfig:
    data_dir: str = MISSING
    train_size: float = MISSING
    detector: TrainDetectorConfig = MISSING
    ocr_config: TrainOCRConfig = MISSING
    seed: int = MISSING
    num_workers: int = MISSING
    optimizer: Any = MISSING
    scheduler: Optional[Any] = None
    exp_dir: str = MISSING
    check_val_every_n_epoch: int = MISSING
    flush_logs_every_n_steps: int = MISSING
    fast_dev_run: bool = False
    precision: int = 16
    max_epochs: int = MISSING
    max_image_size: int = MISSING
    val_check_interval: float = MISSING
