"""Swin Transformer for classification."""

from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision.models.swin_transformer import SwinTransformer


class SwinClassifier(nn.Module):
    """
    Swin Transformer model for image classification.

    This model uses the Swin Transformer as a feature extractor and adds a
    classification head on top. It is designed to be a standalone
    classifier that can be wrapped by an interface class.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: Tuple[int, int] = (4, 4),
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: List[int] = [7, 7],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_classes: int = 3,
        channels: int = 1,
    ) -> None:
        """
        Initialize the Swin Classifier.

        Args:
            image_size: Input image size. The Swin Transformer from torchvision
                        expects a fixed input size.
            patch_size: Patch size for the transformer.
            embed_dim: Embedding dimension.
            depths: Number of layers in each stage.
            num_heads: Number of attention heads in each stage.
            window_size: Window size for self-attention.
            mlp_ratio: Ratio of MLP hidden dim to embedding dim.
            dropout: Dropout rate.
            num_classes: Number of output classes.
            channels: Number of input channels (1 for grayscale, 3 for RGB).

        """
        super().__init__()

        # Swin Transformer Backbone
        self.swin = SwinTransformer(
            patch_size=list(patch_size),
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_classes=num_classes,  # This will be replaced, but required
        )

        # Modify the first layer for the correct number of input channels
        self.swin.features[0][0] = nn.Conv2d(
            channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Replace the final classification head
        feature_dim = self.swin.head.in_features
        self.swin.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor, return_logits: bool = True) -> torch.Tensor:
        """
        Forward pass for classification.

        Args:
            x: Input tensor of shape (B, C, H, W).
            return_logits: If True, returns raw logits. If False, returns
                           probabilities after applying softmax.

        Returns:
            Tensor of shape (B, num_classes) containing logits or probabilities.

        """
        output = self.swin(x)

        # this is added so that mypy does not complain :/
        assert isinstance(output, torch.Tensor)

        if return_logits:
            return output

        return torch.softmax(output, dim=1)
