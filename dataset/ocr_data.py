import string
import os
from typing import Mapping
import zipfile

import torch
from torch.utils import data
import numpy as np
from torchvision import io
from torchvision.io.image import ImageReadMode

BLANK_TOKEN = "-"
PLATE_ALPHABET = list(string.digits) + ["A", "B", "E", "K", "M",
                                        "H", "O", "P", "C", "T", "Y", "X"] + [BLANK_TOKEN]

MAPPING_RUS_ASCII = {
    'с': 'C',
    'о': 'O',
    'н': 'H',
    'м': 'M',
    'в': 'B',
    'а': 'A',
    'е': 'E',
    'т': 'T',
    'р': 'P',
    'у': 'Y',
    'х': 'X',
    'к': 'K'
}


class OCRDataset(data.Dataset):
    def __init__(self, img_dir: str, aug_transforms=None):
        super().__init__()
        self.transforms = aug_transforms
        self._plate_images = []
        self._images = [os.path.join(img_dir, file)
                        for file in os.listdir(img_dir) if file.endswith(".png")]

    def __getitem__(self, index):
        path_to_image = self._images[index]
        text = os.path.splitext(os.path.basename(path_to_image))[0]
        img = io.read_image(path_to_image, mode=ImageReadMode.RGB).permute(1, 2, 0).numpy()

        if self.transforms is not None:
            img = self.transforms(img)

        output = {
            'img': img,
            'text': "".join(MAPPING_RUS_ASCII.get(char, char) for char in text),
            'seq': [PLATE_ALPHABET.index(MAPPING_RUS_ASCII.get(char, char)) for char in text],
            'seq_len': len(text)
        }

        return output

    def __len__(self):
        return len(self._images)
