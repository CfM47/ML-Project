from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from auto_ml.interfaces import (
    ClassificationDatasetInterface,
    ClassificationModelInterface,
    ImageArray,
    MetricsResultInterface,
)
from auto_ml.models.cnn.model import CNNClassifier


class CNNModel(ClassificationModelInterface):
    """
    CNN Model implementation for AutoML.

    Wrap the CNNClassifier model from cnn.model to implement
    the ClassificationModelInterface for region-based classification.
    """

    def __init__(
        self,
        num_classes: int = 3,
        channels: int = 1,
        base_filters: int = 32,
        dropout: float = 0.5,
        device: str = "auto",
        train_epochs: int = 10,
        train_batch_size: int = 16,
        train_learning_rate: float = 0.01,
    ) -> None:
        """
        Initialize the CNN Model.

        Args:
            num_classes: Number of output classes.
            channels: Number of input channels (1 for grayscale, 3 for RGB).
            base_filters: Base number of filters (doubled at each block).
            dropout: Dropout rate before final classification layer.
            device: Device to run the model on ("auto", "cuda", "mps", "cpu").
            train_epochs: Number of training epochs.
            train_batch_size: Training batch size.
            train_learning_rate: Learning rate for training.

        """
        self.num_classes = num_classes
        self.channels = channels
        self.train_epochs = train_epochs
        self.train_batch_size = train_batch_size
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

        self.model = CNNClassifier(
            num_classes=num_classes,
            channels=channels,
            base_filters=base_filters,
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

        # Convert to tensor
        region_tensor = self._preprocess_region(region)
        region_tensor = region_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            probabilities = self.model(region_tensor, return_logits=False)

        # Get class label and confidence
        confidence, class_label = torch.max(probabilities, dim=1)

        return int(class_label.item()), float(confidence.item())

    def _preprocess_region(self, region: ImageArray) -> torch.Tensor:
        """
        Preprocess an image region for the CNN model.

        Args:
            region: Image region as numpy array.

        Returns:
            Preprocessed tensor of shape (1, C, H, W).

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

    def train(
        self,
        dataset: ClassificationDatasetInterface,
    ) -> MetricsResultInterface:
        """
        Train the CNN model on provided data.

        Train the CNN model on the provided dataset. Handles variable-sized images
        by processing them individually.

        Args:
            dataset: The training dataset.

        Returns:
            MetricsResultInterface containing training metrics.

        """
        # Set model to training mode
        self.model.train()

        # Convert dataset to tensors (returns list of tensors + labels)
        images_list, labels_tensor = dataset.to_tensors()

        # Group images by size for batching
        size_groups: Dict[Tuple[int, int], List[int]] = {}
        for idx, img in enumerate(images_list):
            size = (img.shape[1], img.shape[2])  # (H, W)
            if size not in size_groups:
                size_groups[size] = []
            size_groups[size].append(idx)

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_learning_rate,
        )

        final_loss = 0.0
        final_accuracy = 0.0

        # Train on grouped batches
        for epoch in range(self.train_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            # Process each size group in sorted order for determinism
            for size in sorted(size_groups.keys()):
                indices = size_groups[size]
                # Shuffle indices within each size group using numpy for determinism
                perm = np.random.permutation(len(indices))
                shuffled_indices = [indices[i] for i in perm]

                # Create batches
                for batch_start in range(
                    0,
                    len(shuffled_indices),
                    self.train_batch_size,
                ):
                    batch_indices = shuffled_indices[
                        batch_start : batch_start + self.train_batch_size
                    ]

                    # Stack images of same size into batch
                    batch_images = torch.stack(
                        [images_list[i] for i in batch_indices],
                    ).to(self.device)
                    batch_labels = torch.tensor(
                        [labels_tensor[i] for i in batch_indices],
                        dtype=torch.long,
                    ).to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(batch_images, return_logits=True)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item() * batch_images.size(0)
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
