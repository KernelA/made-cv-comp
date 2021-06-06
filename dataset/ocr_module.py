from pytorch_lightning import LightningDataModule
import torch
from torch.utils import data

from .ocr_data import OCRDataset
from utils import load_json

from .plates_module import collate_fn


class OCRDetectionDataModule(LightningDataModule):
    def __init__(self, *,
                 archive_name: str,
                 train_size: float,
                 seed: int,
                 num_workers: int,
                 train_batch_size: int,
                 valid_batch_size: int,
                 train_transforms=None,
                 val_transforms=None,
                 test_transforms=None,
                 dims=None):
        super().__init__(train_transforms=train_transforms,
                         val_transforms=val_transforms, test_transforms=test_transforms, dims=dims)

        self._seed = seed
        self._train_batch_size = train_batch_size
        self._valid_batch_size = valid_batch_size
        self._num_workers = num_workers
        self._archive_name = archive_name
        self._train_size = train_size
        self._train_data = self._val_data = None

    def setup(self, stage) -> None:
        all_data = OCRDataset(self._archive_name, self.train_transforms)

        train_size = round(self._train_size * len(all_data))
        valid_size = len(all_data) - train_size

        self._train_data, self._val_data = data.random_split(
            all_data, [train_size, valid_size], generator=torch.Generator().manual_seed(self._seed))

        self._val_data.transforms = self._val_transforms

    def val_dataloader(self):
        return data.DataLoader(self._val_data,
                               batch_size=self._valid_batch_size,
                               num_workers=self._num_workers,
                               pin_memory=True,
                               collate_fn=collate_fn,
                               drop_last=False, shuffle=False)

    def train_dataloader(self):
        return data.DataLoader(self._train_data,
                               batch_size=self._train_batch_size,
                               num_workers=self._num_workers,
                               pin_memory=True,
                               collate_fn=collate_fn,
                               drop_last=True, shuffle=True)
