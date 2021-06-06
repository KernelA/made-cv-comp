from typing import List
import string
import os
import zipfile

import torch
from torch.utils import data
import numpy as np
from torchvision import io
from torchvision.io.image import ImageReadMode

ALPHABET = list(string.digits) + ["A", "B", "E", "K", "M", "H", "O", "P", "C", "T", "Y", "X"]


class OCRDataset(data.Dataset):
    def __init__(self, archive_name: str, aug_transforms=None):
        super().__init__()
        self.transforms = aug_transforms
        self._plate_images = []
        self._archive = zipfile.ZipFile(archive_name, "r")
        self._items = [item for item in self._archive.infolist(
        ) if item.filename.lower().endswith(".png")]

    def __getitem__(self, index):
        archive_item = self._items[index]

        with self._archive.open(archive_item.filename, "r") as img_zip:
            buffer = np.frombuffer(img_zip.read(), dtype=np.uint8)

        raw_data = torch.tensor(buffer)
        text = os.path.splitext(os.path.basename(archive_item.filename))[0]
        img = io.decode_image(raw_data, mode=ImageReadMode.RGB).permute(1, 2, 0).numpy()

        if self.transforms is not None:
            img = self.transforms(image=img)["image"]

        output = {
            'img': img,
            'text': text,
            'seq': list(text),
            'seq_len': len(text)
        }

        return output

    def __len__(self):
        return len(self._items)
