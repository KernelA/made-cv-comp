import os

import cv2
from torch.utils import data
import numpy as np
from matplotlib.path import Path
import torch


class DetectionDataset(data.Dataset):
    def __init__(self, marks, img_folder: str, transforms=None):
        self.marks = marks
        self.img_folder = img_folder
        self.transforms = transforms

    def __getitem__(self, index):
        item = self.marks[index]
        img_path = os.path.join(self.img_folder, item["file"])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width = img.shape[:2]

        box_coords = item['nums']
        boxes = []
        labels = torch.ones(len(box_coords), dtype=torch.long)
        masks = []

        for box in box_coords:
            points = np.array(box['box'])
            x0, y0 = np.min(points[:, 0]), np.min(points[:, 1])
            x2, y2 = np.max(points[:, 0]), np.max(points[:, 1])

            x0 = np.clip(x0, 0, width - 1)
            x2 = np.clip(x2, 0, width - 1)

            y0 = np.clip(y0, 0, height - 1)
            y2 = np.clip(y2, 0, height - 1)

            boxes.append([x0, y0, x2, y2])

            # Здесь мы наши 4 точки превращаем в маску
            # Это нужно, чтобы кроме bounding box предсказывать и, соответственно, маску :)
            nx, ny = width, height
            poly_verts = points
            x, y = np.meshgrid(np.arange(nx), np.arange(ny))
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x, y)).T
            path = Path(poly_verts)
            grid = path.contains_points(points)
            grid = grid.reshape((ny, nx)).astype(int)
            masks.append(grid)

        boxes = torch.tensor(boxes)
        masks = torch.tensor(masks)

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.marks)
