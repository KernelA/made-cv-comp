from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class ConvertConfig:
    input_zip: str = MISSING
    out_dir: str = MISSING
