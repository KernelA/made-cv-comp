import os
import copy

from pytorch_lightning import LightningDataModule
import torch
from torch.utils import data
import numpy as np
import pandas as pd

from .plates_data import DetectionDataset
from utils import load_json


def collate_fn(batch):
    return tuple(map(list, zip(*batch)))


def filter_marks(marks: dict) -> list:
    remove_indices = []

    for example_num, item in enumerate(marks):
        boxes = item["nums"]
        remove_boxes_indices = []
        for i, box in enumerate(boxes):
            points = np.round(np.array(box['box'])).astype(np.int32)
            x0, y0 = np.min(points[:, 0]), np.min(points[:, 1])
            x2, y2 = np.max(points[:, 0]), np.max(points[:, 1])
            bbox_width = x2 - x0
            bbox_height = y2 - y0

            if bbox_width / bbox_height <= 0.8:
                remove_boxes_indices.append(i)

        if remove_boxes_indices:
            print("Remove incorrcet boxes: ", len(remove_boxes_indices))

        boxes = [box for i, box in enumerate(boxes) if i not in remove_boxes_indices]

        item["nums"] = boxes

        # Empty boxes
        if not boxes:
            remove_indices.append(example_num)

    print("Images with incorrect aspect ratio: ", len(remove_indices))

    return [mark for i, mark in enumerate(marks) if i not in remove_indices]


class PlatesDetectionDataModule(LightningDataModule):
    def __init__(self, *, data_dir: str, train_size: float,
                 seed: int,
                 num_workers: int,
                 train_batch_size: int,
                 valid_batch_size: int,
                 max_size: int,
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
        self._data_dir = data_dir
        self._train_size = train_size
        self._marks = load_json(os.path.join(self._data_dir, "train.json"))
        self._marks = filter_marks(self._marks)
        self._max_image_size = max_size
        self._train_data = self._val_data = None

    def setup(self, stage) -> None:
        all_data = DetectionDataset(self._marks, self._data_dir,
                                    max_size=self._max_image_size, transforms=self._train_transforms)

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
