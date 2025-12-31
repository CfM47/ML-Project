"""ViT model implementation for segmentation."""

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from auto_ml.implementations.segmentators.base import InMemoryPyTorchDataset
from auto_ml.interfaces import (
    MaskPair,
    MetricsResultInterface,
    SegmentationDatasetInterface,
    SegmentationModelInterface,
)
from auto_ml.models.vit.segmentation import ViTSegmentation


class ViTModel(SegmentationModelInterface):
    """
    ViT Model implementation for AutoML.

    Wraps the ViTSegmentation model from vit.model.
    """

    def __init__(  # noqa: D107
        self,
        epochs: int = 10,
        batch_size: int = 4,
        lr: float = 1e-4,
        device: str = "auto",
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        if device == "auto":
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = device

        self.model = ViTSegmentation(
            image_size=512,
            patch_size=16,
            num_classes=3,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=1024,
            channels=1,  # Warning: Hardcoded for grayscale, should be dynamic if needed
            dropout=0.1,
            emb_dropout=0.1,
        ).to(self.device)

    def train(self, dataset: SegmentationDatasetInterface) -> MetricsResultInterface:
        """Train the model."""
        pytorch_dataset = InMemoryPyTorchDataset(dataset)
        dataloader = DataLoader(
            pytorch_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()

        total_loss: float = 0

        # Simplified training loop for AutoML context
        for epoch in range(self.epochs):
            epoch_loss = 0
            # iterate over batches
            for inputs, masks in dataloader:
                inputs = inputs.to(self.device)
                masks = masks.to(self.device)

                # Check for channel mismatch if we hardcoded channels=1
                if inputs.shape[1] != 1 and self.model.patch_embed.in_channels == 1:
                    # Force grayscale conversion if model expects 1 channel
                    # (B, 3, H, W) -> (B, 1, H, W) using luminosity method
                    inputs = (
                        inputs[:, 0:1, :, :] * 0.299
                        + inputs[:, 1:2, :, :] * 0.587
                        + inputs[:, 2:3, :, :] * 0.114
                    )

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0
            total_loss = float(avg_loss)  # Report the last epoch's loss
            print(f"Epoch {epoch + 1}/{self.epochs} Loss: {avg_loss:.4f}")

        return MetricsResultInterface(
            loss=total_loss,
            accuracy=0.0,  # Placeholder
            additional_metrics={"epochs_trained": self.epochs},
        )

    def evaluate(self, dataset: SegmentationDatasetInterface) -> List[MaskPair]:
        """Evaluate the model and return predicted/real mask pairs."""
        pytorch_dataset = InMemoryPyTorchDataset(dataset)
        dataloader = DataLoader(pytorch_dataset, batch_size=1, shuffle=False)

        self.model.eval()
        mask_pairs: List[MaskPair] = []

        with torch.no_grad():
            for inputs, masks in dataloader:
                inputs = inputs.to(self.device)
                masks = masks.to(self.device)

                if inputs.shape[1] != 1 and self.model.patch_embed.in_channels == 1:
                    inputs = (
                        inputs[:, 0:1, :, :] * 0.299
                        + inputs[:, 1:2, :, :] * 0.587
                        + inputs[:, 2:3, :, :] * 0.114
                    )

                outputs = self.model(inputs)

                # Get predicted mask
                predictions = torch.argmax(outputs, dim=1)
                predicted_mask = predictions.squeeze().cpu().numpy().astype(np.uint8)
                real_mask = masks.squeeze().cpu().numpy().astype(np.uint8)

                mask_pairs.append((predicted_mask, real_mask))

        return mask_pairs
