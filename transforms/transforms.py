import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


TORCHVISION_RGB_STD_LIST = [0.229, 0.224, 0.225]
TORCHVISION_RGB_MEAN_LIST = [0.485, 0.456, 0.406]

TORCHVISION_RGB_MEAN = torch.tensor(TORCHVISION_RGB_MEAN_LIST, dtype=torch.float32)
TORCHVISION_RGB_STD = torch.tensor(TORCHVISION_RGB_STD_LIST, dtype=torch.float32)


class AugTransform:
    def __init__(self, aug_trasnform):
        self._aug_transform = aug_trasnform

    def __call__(self, img):
        return self._aug_transform(image=img)["image"]


def get_train_transform():
    return transforms.Compose([
        transforms.ConvertImageDtype(dtype=torch.get_default_dtype()),
        transforms.Normalize(mean=TORCHVISION_RGB_MEAN_LIST,
                             std=TORCHVISION_RGB_STD_LIST, inplace=True)
    ]
    )


def get_ocr_valid_transform_list(height: int = 64, width: int = 320):
    return [
        A.Resize(height, width),
        A.Normalize(mean=TORCHVISION_RGB_MEAN_LIST, std=TORCHVISION_RGB_STD_LIST),
        ToTensorV2()
    ]
