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
        num_blocks: int = 3,
    ) -> None:
        """
        Initialize the CNN Classifier.

        Args:
            num_classes: Number of output classes.
            channels: Number of input channels (1 for grayscale, 3 for RGB).
            base_filters: Base number of filters (doubled at each block).
            dropout: Dropout rate before final classification layer.
            num_blocks: Number of convolutional blocks. Each block halves the
                spatial dimensions via MaxPool2d, so the minimum supported
                input size is 2^num_blocks × 2^num_blocks pixels.
                Examples: num_blocks=3 → min 8×8, num_blocks=5 → min 32×32.

        """
        super().__init__()

        self.num_classes = num_classes
        self.num_blocks = num_blocks

        # Feature extraction blocks
        # Each block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> MaxPool
        # Receptive field grows while spatial dimensions shrink
        self.features = self._build_feature_blocks(channels, base_filters, num_blocks)

        # Calculate final channel count after num_blocks doublings:
        # Block 1: base_filters, Block 2: base_filters*2, ...
        # Block N: base_filters * 2^(N-1)
        final_channels = base_filters * (2 ** (num_blocks - 1))

        # Adaptive pooling: any spatial size -> 1x1
        # This enables variable input sizes
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(final_channels, max(final_channels // 4, num_classes)),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(max(final_channels // 4, num_classes), num_classes),
        )

        # Softmax for probability distribution output
        self.softmax = nn.Softmax(dim=1)

    def _build_feature_blocks(
        self,
        channels: int,
        base_filters: int,
        num_blocks: int,
    ) -> nn.Sequential:
        """
        Build feature extraction blocks dynamically.

        Args:
            channels: Number of input channels.
            base_filters: Base number of filters.
            num_blocks: Number of convolutional blocks to create.

        Returns:
            Sequential container with all feature extraction blocks.

        """
        blocks = []
        in_channels = channels

        for i in range(num_blocks):
            out_channels = base_filters * (2**i)
            blocks.append(self._make_block(in_channels, out_channels))
            in_channels = out_channels

        return nn.Sequential(*blocks)

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
        Perform forward pass of the CNN Classifier.

        Args:
            x: Input tensor of shape (B, C, H, W). H and W must be at least
                2^num_blocks pixels (e.g., 8×8 for num_blocks=3).
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

        sm: torch.Tensor = self.softmax(x)

        return sm
