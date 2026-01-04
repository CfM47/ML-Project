from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from auto_ml.interfaces import (
    ClassificationDatasetInterface,
    ClassificationModelInterface,
    ImageArray,
    MetricsResultInterface,
)
from auto_ml.models.swin.classification import SwinClassifier


class SwinModel(ClassificationModelInterface):
    """
    Swin Transformer Model implementation for AutoML.

    Wraps the SwinClassifier model to implement the
    ClassificationModelInterface for region-based classification.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 4,
        num_classes: int = 3,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: int = 7,
        channels: int = 1,
        dropout: float = 0.0,
        device: str = "auto",
        train_epochs: int = 10,
        train_batch_size: int = 0,
        train_learning_rate: float = 0.001,
    ) -> None:
        """
        Initialize the SwinModel.

        Args:
            image_size: Input image size for the transformer.
            patch_size: Size of each patch.
            num_classes: Number of output classes.
            embed_dim: Embedding dimension.
            depths: Number of transformer layers in each stage.
            num_heads: Number of attention heads in each stage.
            window_size: Attention window size.
            channels: Number of input channels (1 for grayscale, 3 for RGB).
            dropout: Dropout rate.
            device: Device to run on ("auto", "cuda", "mps", "cpu").
            train_epochs: Number of training epochs.
            train_batch_size: Training batch size. Adaptive if 0.
            train_learning_rate: Learning rate for training.

        """
        self.num_classes = num_classes
        self.channels = channels
        self.image_size = image_size
        self.train_epochs = train_epochs
        self.train_learning_rate = train_learning_rate

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

        if train_batch_size == 0:
            if self.device == "cuda":
                self.train_batch_size = 32
            elif self.device == "mps":
                self.train_batch_size = 16
            else:
                self.train_batch_size = 8
        else:
            self.train_batch_size = train_batch_size

        self.model = SwinClassifier(
            image_size=image_size,
            patch_size=(patch_size, patch_size),
            num_classes=num_classes,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=[window_size, window_size],
            channels=channels,
            dropout=dropout,
        ).to(self.device)

        self.model.eval()

    def classify(
        self,
        image: ImageArray,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> Tuple[int, float]:
        """Classify an image region."""
        region = image[y : y + height, x : x + width]
        region_tensor = self._preprocess_region(region)
        region_tensor = region_tensor.to(self.device)

        with torch.no_grad():
            probabilities = self.model(region_tensor, return_logits=False)

        confidence, class_label = torch.max(probabilities, dim=1)

        return int(class_label.item()), float(confidence.item())

    def train(
        self,
        dataset: ClassificationDatasetInterface,
    ) -> MetricsResultInterface:
        """Train the Swin model."""
        self.model.train()

        images_list, labels_tensor = dataset.to_tensors()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_learning_rate,
        )

        final_loss = 0.0
        final_accuracy = 0.0

        for epoch in range(self.train_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            indices = np.arange(len(images_list))
            perm = np.random.permutation(len(indices))
            shuffled_indices = indices[perm]

            for batch_start in range(
                0,
                len(shuffled_indices),
                self.train_batch_size,
            ):
                batch_indices = shuffled_indices[
                    batch_start : batch_start + self.train_batch_size
                ]

                batch_images = []
                for idx in batch_indices:
                    img = images_list[int(idx)]
                    if (
                        img.shape[1] != self.image_size
                        or img.shape[2] != self.image_size
                    ):
                        img = F.interpolate(
                            img.unsqueeze(0),
                            size=(self.image_size, self.image_size),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)
                    batch_images.append(img)

                batch_images_tensor = torch.stack(batch_images).to(self.device)
                batch_labels = torch.tensor(
                    [labels_tensor[int(i)] for i in batch_indices],
                    dtype=torch.long,
                ).to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_images_tensor, return_logits=True)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_images_tensor.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(batch_labels).sum().item()
                total += batch_labels.size(0)

            final_loss = epoch_loss / total if total > 0 else 0.0
            final_accuracy = correct / total if total > 0 else 0.0

            print(f"Epoch {epoch + 1}/{self.train_epochs}, Loss: {final_loss:.6f}")

        self.model.eval()
        return MetricsResultInterface(
            accuracy=final_accuracy,
            loss=final_loss,
        )

    def _preprocess_region(self, region: ImageArray) -> torch.Tensor:
        """Preprocess a region for the Swin model."""
        region_float = region.astype(np.float32) / 255.0

        if region_float.ndim == 2:
            tensor = torch.from_numpy(region_float).unsqueeze(0).unsqueeze(0)
        else:
            tensor = torch.from_numpy(region_float).permute(2, 0, 1).unsqueeze(0)

        if tensor.shape[1] != self.channels:
            if self.channels == 1 and tensor.shape[1] == 3:
                tensor = (
                    tensor[:, 0:1, :, :] * 0.299
                    + tensor[:, 1:2, :, :] * 0.587
                    + tensor[:, 2:3, :, :] * 0.114
                )
            elif self.channels == 3 and tensor.shape[1] == 1:
                tensor = tensor.repeat(1, 3, 1, 1)

        if tensor.shape[2] != self.image_size or tensor.shape[3] != self.image_size:
            tensor = F.interpolate(
                tensor,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        return tensor

    def load_weights(self, path: Path) -> None:
        """Load model weights from a file."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def save_weights(self, path: Path) -> None:
        """Save model weights to a file."""
        torch.save(self.model.state_dict(), path)
