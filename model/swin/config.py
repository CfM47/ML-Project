"""Configuration for Swin model training and validation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class SwinTrainingConfig:
    """Configuration for Swin training and validation experiments."""

    # Training percentages for learning curve analysis
    train_percentages: List[int] = field(
        default_factory=lambda: [10, 20, 30, 40, 50, 60, 70, 80],
    )

    # K-fold settings (matches DataAugmentatorNode default)
    n_folds: int = 5

    # Swin model settings (from setup/models/setup.py)
    epochs: int = 40
    batch_size: int = 2
    learning_rate: float = 1e-4
    embed_dim: int = 128
    depths: List[int] = field(default_factory=lambda: [2, 2, 18, 2])
    num_heads: List[int] = field(default_factory=lambda: [8, 16, 32, 64])

    # Early stopping (None = disabled)
    patience: int | None = None

    # Augmentation copies (for 2Geo2Photo1SEM)
    augmentation_copies: int = 2

    # Visualization
    num_test_visualizations: int = 10

    # Output directory
    output_dir: Path = field(default_factory=lambda: Path("results/swin"))

    # Reproducibility
    seed: int = 42

    # Device: "auto", "cuda", "mps", or "cpu"
    device: str = "auto"
