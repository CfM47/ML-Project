"""Base utilities for model implementations."""

from typing import Tuple

import torch
from torch.utils.data import Dataset

from auto_ml.interfaces import SegmentationDatasetInterface


class InMemoryPyTorchDataset(Dataset):
    """Bridge between AutoML DatasetInterface and PyTorch Dataset."""

    def __init__(self, dataset: SegmentationDatasetInterface) -> None:
        """Initialize the Dataset."""
        self.dataset = dataset

    def __len__(self) -> int:
        """Get Amount of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa: D105
        # reuse logic from vit/dataset.py but adapted for in-memory data
        input_interface, output_interface = self.dataset[idx]

        # Prepare inputs
        # Assuming images are numpy arrays (H, W, 3) or (H, W)
        img = input_interface.image
        if input_interface.is_grayscale:
            # Convert to tensor (1, H, W)
            img_tensor = torch.from_numpy(img).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)
        else:
            # Convert RGB (H, W, 3) -> (3, H, W)
            img_tensor = torch.from_numpy(img).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1)

        # Prepare targets
        mask = output_interface.mask
        mask_tensor = torch.from_numpy(mask).long()

        return img_tensor, mask_tensor
