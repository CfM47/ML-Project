import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    """CNN Classifier for variable-size image classification."""

    def __init__(
        self,
        num_classes: int = 3,
        channels: int = 1,
        base_filters: int = 32,
        dropout: float = 0.5,
    ) -> None:
        """
        Initialize the CNN Classifier.

        Args:
            num_classes: Number of output classes.
            channels: Number of input channels (1 for grayscale, 3 for RGB).
            base_filters: Base number of filters (doubled at each block).
            dropout: Dropout rate before final classification layer.

        """
        super().__init__()

        self.num_classes = num_classes

        # Feature extraction blocks
        # Each block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> MaxPool
        # Receptive field grows while spatial dimensions shrink
        self.features = nn.Sequential(
            # Block 1: channels -> base_filters
            self._make_block(channels, base_filters),
            # Block 2: base_filters -> base_filters * 2
            self._make_block(base_filters, base_filters * 2),
            # Block 3: base_filters * 2 -> base_filters * 4
            self._make_block(base_filters * 2, base_filters * 4),
            # Block 4: base_filters * 4 -> base_filters * 8
            self._make_block(base_filters * 4, base_filters * 8),
            # Block 5: base_filters * 8 -> base_filters * 16
            self._make_block(base_filters * 8, base_filters * 16),
        )

        # Adaptive pooling: any spatial size -> 1x1
        # This enables variable input sizes
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_filters * 16, base_filters * 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(base_filters * 4, num_classes),
        )

        # Softmax for probability distribution output
        self.softmax = nn.Softmax(dim=1)

    def _make_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Create a convolutional block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.

        Returns:
            A sequential block with two conv layers and max pooling.

        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass of the CNN Classifier.

        Args:
            x: Input tensor of shape (B, C, H, W). H and W can be any size >= 32.
            return_logits: If True, return raw logits instead of probabilities.
                          Use True when training with CrossEntropyLoss.

        Returns:
            Tensor of shape (B, num_classes) with probabilities (or logits if
            return_logits=True). Probabilities sum to 1 along dim=1.

        """
        # Feature extraction
        x = self.features(x)

        # Global pooling: (B, C, H', W') -> (B, C, 1, 1)
        x = self.global_pool(x)

        # Classification: (B, C, 1, 1) -> (B, num_classes)
        x = self.classifier(x)

        # Return probabilities or logits
        if return_logits:
            return x
        return self.softmax(x)
