import torch
from torchvision import transforms

TORCHVISION_RGB_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
TORCHVISION_RGB_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

TORCHVISION_RGB_STD_LIST = [0.229, 0.224, 0.225]
TORCHVISION_RGB_MEAN_LIST = [0.485, 0.456, 0.406]


def get_train_transform():
    return transforms.Compose([
        transforms.ConvertImageDtype(dtype=torch.get_default_dtype()),
        transforms.Normalize(mean=TORCHVISION_RGB_MEAN_LIST,
                             std=TORCHVISION_RGB_STD_LIST, inplace=True)
    ]
    )
