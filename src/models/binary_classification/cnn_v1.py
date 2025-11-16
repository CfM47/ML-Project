import torch
import torch.nn as nn


class CNNBinaryClassifierV1(nn.Module):
    """
    A CNN for binary classification of 256x256 grayscale images.

    The model takes in a batch of grayscale images and outputs a single logit for each
    image, which can be passed through a sigmoid function to obtain the probability for
    the positive class.
    """

    def __init__(self) -> None:
        """Initialize the CNN model architecture."""
        super().__init__()
        self.conv_stack = nn.Sequential(
            # Input size: (N, 1, 256, 256)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (N, 16, 128, 128)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (N, 32, 64, 64)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (N, 64, 32, 32)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (N, 128, 16, 16)
        )
        self.flatten = nn.Flatten()
        self.fc_stack = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the model.

        Args:
            x: Input tensor of shape (N, 1, 256, 256), where N is the batch size.

        Returns:
            A tensor of shape (N, 1) containing the raw logits for binary
            classification.

        """
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits: torch.Tensor = self.fc_stack(x)
        return logits
