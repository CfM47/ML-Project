from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from auto_ml.interfaces import ClassificationModelInterface, ImageArray
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

        """
        self.num_classes = num_classes
        self.channels = channels
        self.image_size = image_size

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
