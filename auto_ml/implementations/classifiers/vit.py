from pathlib import Path
from typing import Tuple

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
from auto_ml.models.vit.classification import ViTClassification


class ViTModel(ClassificationModelInterface):
    """
    ViT Model implementation for AutoML.

    Wraps the ViTClassification model from vit.classification to implement
    the ClassificationModelInterface for region-based classification.
    """

    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 16,
        num_classes: int = 3,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_dim: int = 3072,
        channels: int = 1,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        device: str = "auto",
        train_epochs: int = 10,
        train_batch_size: int = 0,
        train_learning_rate: float = 0.001,
    ) -> None:
        """
        Initialize the ViT Model.

        Args:
            image_size: Input image size (must be divisible by patch_size).
            patch_size: Size of each patch.
            num_classes: Number of output classes.
            dim: Transformer embedding dimension.
            depth: Number of transformer encoder layers.
            heads: Number of attention heads.
            mlp_dim: Dimension of the MLP feedforward layer.
            channels: Number of input channels (1 for grayscale, 3 for RGB).
            dropout: Dropout rate in transformer and classification head.
            emb_dropout: Dropout rate after positional embedding.
            device: Device to run the model on ("auto", "cuda", "mps", "cpu").
            train_epochs: Number of training epochs.
            train_batch_size: Training batch size. If 0, uses adaptive sizing
                            (32 for CUDA, 16 for MPS, 8 for CPU).
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

        # Set adaptive batch size if not specified
        if train_batch_size == 0:
            if self.device == "cuda":
                self.train_batch_size = 32
            elif self.device == "mps":
                self.train_batch_size = 16
            else:
                self.train_batch_size = 8
        else:
            self.train_batch_size = train_batch_size

        self.model = ViTClassification(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=channels,
            dropout=dropout,
            emb_dropout=emb_dropout,
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
        """
        Classify an image region.

        Args:
            image: Image as numpy array.
            x: x-coordinate of the region.
            y: y-coordinate of the region.
            width: Width of the region.
            height: Height of the region.

        Returns:
            Tuple of (class_label, confidence),
            where class_label ∈ {0,1,2} and confidence ∈ [0,1].

        """
        # Extract region from image
        region = image[y : y + height, x : x + width]

        # Convert to tensor and resize to expected image_size
        region_tensor = self._preprocess_region(region)
        region_tensor = region_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            probabilities = self.model(region_tensor, return_logits=False)

        # Get class label and confidence
        confidence, class_label = torch.max(probabilities, dim=1)

        return int(class_label.item()), float(confidence.item())

    def train(
        self,
        dataset: ClassificationDatasetInterface,
    ) -> MetricsResultInterface:
        """
        Train the ViT model on provided data.

        Args:
            dataset: The training dataset.

        Returns:
            MetricsResultInterface containing training metrics.

        """
        # Set model to training mode
        self.model.train()

        # Convert dataset to tensors (returns list of tensors + labels)
        images_list, labels_tensor = dataset.to_tensors()

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_learning_rate,
        )

        final_loss = 0.0
        final_accuracy = 0.0

        # Train over epochs
        for epoch in range(self.train_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            # Shuffle indices for this epoch
            indices = np.arange(len(images_list))
            perm = np.random.permutation(len(indices))
            shuffled_indices = indices[perm]

            # Create batches
            for batch_start in range(
                0,
                len(shuffled_indices),
                self.train_batch_size,
            ):
                batch_indices = shuffled_indices[
                    batch_start : batch_start + self.train_batch_size
                ]

                # Get batch images and preprocess to fixed size
                batch_images = []
                for idx in batch_indices:
                    img = images_list[int(idx)]
                    # Ensure image is the correct size
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

                # Stack batch
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

            # Log progress
            print(f"Epoch {epoch + 1}/{self.train_epochs}, Loss: {final_loss:.6f}")

        # Set model back to eval mode
        self.model.eval()
        return MetricsResultInterface(
            accuracy=final_accuracy,
            loss=final_loss,
        )

    def _preprocess_region(self, region: ImageArray) -> torch.Tensor:
        """
        Preprocess an image region for the ViT model.

        Args:
            region: Image region as numpy array.

        Returns:
            Preprocessed tensor of shape (1, C, image_size, image_size).

        """
        # Normalize to [0, 1]
        region_float = region.astype(np.float32) / 255.0

        # Handle grayscale vs RGB
        if region_float.ndim == 2:
            # Grayscale: (H, W) -> (1, 1, H, W)
            tensor = torch.from_numpy(region_float).unsqueeze(0).unsqueeze(0)
        else:
            # RGB: (H, W, C) -> (1, C, H, W)
            tensor = torch.from_numpy(region_float).permute(2, 0, 1).unsqueeze(0)

        # Handle channel mismatch
        if tensor.shape[1] != self.channels:
            if self.channels == 1 and tensor.shape[1] == 3:
                # Convert RGB to grayscale using luminosity method
                tensor = (
                    tensor[:, 0:1, :, :] * 0.299
                    + tensor[:, 1:2, :, :] * 0.587
                    + tensor[:, 2:3, :, :] * 0.114
                )
            elif self.channels == 3 and tensor.shape[1] == 1:
                # Convert grayscale to RGB by repeating channels
                tensor = tensor.repeat(1, 3, 1, 1)

        # Resize to expected image_size (ViT requires fixed input size)
        if tensor.shape[2] != self.image_size or tensor.shape[3] != self.image_size:
            tensor = F.interpolate(
                tensor,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        return tensor

    def load_weights(self, path: Path) -> None:
        """
        Load model weights from a file.

        Args:
            path: Path to the weights file.

        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def save_weights(self, path: Path) -> None:
        """
        Save model weights to a file.

        Args:
            path: Path to save the weights file.

        """
        torch.save(self.model.state_dict(), path)
