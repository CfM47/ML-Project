from pathlib import Path
from typing import Any, Dict, List, Tuple

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Grayscale,
    Normalize,
    RandomCrop,
    RandomRotation,
    ToTensor,
)

from ..configs.schemas import ApproachConfig


class BinaryImageFolder(ImageFolder):
    """An ImageFolder that only loads images from 'Brittle' and 'Ductile' classes."""

    def find_classes(self, directory: str | Path) -> Tuple[List[str], Dict[str, int]]:
        """Find the specific classes 'Brittle' and 'Ductile'."""
        classes = ["Brittle", "Ductile"]
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


def get_data(config: ApproachConfig) -> Tuple[BinaryImageFolder, BinaryImageFolder]:
    """Create training and validation datasets with appropriate transformations."""
    train_transform = Compose(
        [
            Grayscale(num_output_channels=1),
            RandomRotation(degrees=(-90, 90)),
            RandomCrop(256),
            ToTensor(),
            Normalize((0.5,), (0.5,)),
        ],
    )

    val_transform = Compose(
        [
            Grayscale(num_output_channels=1),
            CenterCrop(256),
            ToTensor(),
            Normalize((0.5,), (0.5,)),
        ],
    )

    train_dataset = BinaryImageFolder(root=config.data_path, transform=train_transform)
    val_dataset = BinaryImageFolder(root=config.data_path, transform=val_transform)

    return train_dataset, val_dataset


def split_exclusive(
    train_dataset: BinaryImageFolder,
    val_dataset: BinaryImageFolder,
    config: ApproachConfig,
    test_size_override: float | None = None,
) -> Tuple[Subset, Subset]:
    """Perform a stratified split on the datasets."""
    try:
        targets = train_dataset.targets
    except AttributeError:
        targets = [s[1] for s in train_dataset.samples]

    test_size = test_size_override or (1.0 - config.train_test_split)

    num_images = len(targets)
    train_indices, val_indices = train_test_split(
        range(num_images),
        test_size=test_size,
        stratify=targets,
        random_state=42,
    )

    train_subset: Subset[Any] = Subset(train_dataset, train_indices)
    val_subset: Subset[Any] = Subset(val_dataset, val_indices)

    print(f"Total images (Brittle & Ductile): {num_images}")
    print(f"Training set size: {len(train_subset)}")
    print(f"Validation set size: {len(val_subset)}")

    return train_subset, val_subset


def get_dataloader(
    config: ApproachConfig,
    subset: Subset,
    train: bool,
) -> DataLoader:
    """Create a DataLoader for a given subset."""
    return DataLoader(
        subset,
        batch_size=config.batch_size,
        shuffle=train,
        num_workers=2,
    )
