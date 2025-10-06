from typing import Callable, Optional, Tuple, List
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except:
    HAS_ALBUMENTATIONS = False

try:
    import cv2
    HAS_CV2 = True
except:
    HAS_CV2 = False


def build_torchvision_transforms(image_size=224, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225), train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2,0.2,0.2,0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size*1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def build_albumentations_transforms(image_size=224, train=True):
    if not HAS_ALBUMENTATIONS:
        raise RuntimeError("albumentations not installed")
    if train:
        return A.Compose([
            A.RandomResizedCrop(image_size, image_size, scale=(0.8,1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.ShiftScaleRotate(0.0625,0.1,15,p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(),
            ToTensorV2(),
        ])


class SimpleImageDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], transform: Optional[Callable]=None):
        assert len(image_paths)==len(labels)
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        path, label = self.image_paths[idx], self.labels[idx]
        if HAS_CV2:
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        else:
            from PIL import Image
            img = Image.open(path).convert("RGB")

        if self.transform is not None:
            if HAS_ALBUMENTATIONS and isinstance(self.transform, A.Compose):
                img = self.transform(image=img)["image"]
            else:
                img = self.transform(img)
        return img, label
