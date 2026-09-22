"""
data.py - Dataset loading, ingestion and preprocessing for SNN Bench .
"""





from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

from .config import BATCH_SIZE

DATASETS = {
    "mnist":        (torchvision.datasets.MNIST, 1, 28, 10),
    "fashionmnist": (torchvision.datasets.FahionMNIST, 1, 28, 10),
    "kmnist":       (torchvision.datasets.KMNIST, 1, 28, 10),
    "cifar10":      (torchvision.datasets.CIFAR10, 3, 32, 10), # Needs a 3-ch Model
}

# an organized way to structure how the data and labels are loaded into the model
class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None): # constructor: what to do when it is created
    # transform resizes images to th same size
    # ImageFolder assumes any subfolders in the path are the class name of the image
    # and will handle creating all the labels for us
        self.data = ImageFolder(data_dir, transform=transform)

    # The dataloader will need to know how many examples we have in a dataset
    # once we create it
    def __len__(self):
        return len(self.data)

    # takes an index location in dataset and returns one item
    def __getitem__(self, idx):
        return self.data[idx]

    # return data classes from image folder
    # since the folder in which the actual images are stored
    # is the actual class
    @property
    def classes(self):
        return self.data.classes

    @property
    def num_classes(self):
        return len(self.classes)

    def target_to_class(self):
        return {v: k for k, v in self.data.class_to_idx.items()}

@dataclass
class DatasetMeta:
    name: str
    num_classes: int
    in_channels: int
    img_size: int  # assumes square H == W

 
def get_loaders(
    name: str = "mnist", root: str = "./dataset", batch_size: int = BATCH_SIZE,
    val_fraction: float = 0.5, seed: int = 42, num_workers: int = 2
):
    """
    Return :(train_loader, val_loader, test_loader, meta).

    The official test split is halved into a validation set and a held-out test
    set (matching the notebook's 50/50 split), so model selection never touches
    the final test numbers.
    """
    name = name.lower()
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Available: {list(DATASETS)}")
    cls, in_ch, img_size, n_cls = DATASETS[name]

    tf = v2.Compose([v2.ToTensor()])  # pixels in [0, 1]
    train_set = cls(root=root, train=True,  download=True, transform=tf)
    test_full = cls(root=root, train=False, download=True, transform=tf)

    val_size = int(len(test_full) * val_fraction)
    test_size = len(test_full) - val_size
    val_set, test_set = random_split(
        test_full, [val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True,  drop_last=True,  **common)
    val_loader   = DataLoader(val_set,   shuffle=False, drop_last=False, **common)
    test_loader  = DataLoader(test_set,  shuffle=False, drop_last=False, **common)

    meta = DatasetMeta(name=name, num_classes=n_cls, in_channels=in_ch, img_size=img_size)
    return train_loader, val_loader, test_loader, meta