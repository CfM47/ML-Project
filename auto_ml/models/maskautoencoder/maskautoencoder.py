import torch
import torch.nn as nn
from typing import Tuple

class MaskAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for mask dimensionality reduction.

    Compresses 512x512 masks to a 3D latent representation while
    preserving structural information through reconstruction.
    """

    def __init__(self, latent_dim: int = 3) -> None:
        """
        Initialize the autoencoder.

        Args:
            latent_dim: Dimension of the latent space (default: 3).

        """
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 512x512 -> 3D
        # 512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 256x256
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # 4x4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Flatten(),  # 512 * 4 * 4 = 8192
            nn.Linear(512 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
        )

        # Decoder: 3D -> 512x512
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 256x256
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 512x512
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        x = self.decoder_fc(z)
        x = x.view(-1, 512, 4, 4)
        return self.decoder_conv(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: returns (reconstruction, latent_vector)."""
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z
