import torch
from torchvision import transforms

TORCHVISION_RGB_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
TORCHVISION_RGB_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


def get_train_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=TORCHVISION_RGB_MEAN, std=TORCHVISION_RGB_STD, inplace=True)
        ]
    )
