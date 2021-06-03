import os

from pytorch_lightning import LightningDataModule
import torch
from torch.utils import data

from .plates_data import DetectionDataset
from utils import load_json


def collate_fn(batch):
    return tuple(map(list, zip(*batch)))


class PlatesDetectionDataModule(LightningDataModule):
    def __init__(self, *, data_dir: str, train_size: float,
                 seed: int,
                 num_workers: int,
                 batch_size: int,
                 train_transforms=None,
                 val_transforms=None,
                 test_transforms=None,
                 dims=None):
        super().__init__(train_transforms=train_transforms,
                         val_transforms=val_transforms, test_transforms=test_transforms, dims=dims)

        self._seed = seed
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._data_dir = data_dir
        self._train_size = train_size
        self._marks = load_json(os.path.join(self._data_dir, "train.json"))
        self._train_data = self._val_data = None

    def setup(self, stage) -> None:
        all_data = DetectionDataset(self._marks, self._data_dir, transforms=self._train_transforms)

        train_size = round(self._train_size * len(all_data))
        valid_size = len(all_data) - train_size

        self._train_data, self._val_data = data.random_split(
            all_data, [train_size, valid_size], generator=torch.Generator().manual_seed(self._seed))

        self._val_data.transforms = self._val_transforms

    def val_dataloader(self):
        return data.DataLoader(self._val_data,
                               batch_size=self._batch_size,
                               num_workers=self._num_workers,
                               pin_memory=True,
                               collate_fn=collate_fn,
                               drop_last=False, shuffle=False)

    def train_dataloader(self):
        return data.DataLoader(self._train_data,
                               batch_size=self._batch_size,
                               num_workers=self._num_workers,
                               pin_memory=True,
                               collate_fn=collate_fn,
                               drop_last=True, shuffle=True)
