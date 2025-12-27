"""CNN Classifier Validation Module.

Perform k-fold cross-validation with varying training data percentages
to evaluate the CNN classifier's learning curve.
"""

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from auto_ml.models.cnn.model import CNNClassifier


# ==============================================================================
# Configuration
# ==============================================================================

# Project root directory (relative to this file)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ValidationConfig:
    """Configuration for validation experiments."""

    # Data paths (relative to project root)
    data_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "data/sem_images/raw")
    labels_file: str = "Labels.csv"

    # Training percentages to evaluate
    train_percentages: List[int] = field(
        default_factory=lambda: [10, 20, 30, 40, 50, 60, 70, 80],
    )

    # K-fold settings
    n_folds: int = 5

    # Training settings
    max_epochs: Optional[int] = None  # If None, train until patience runs out
    patience: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3

    # Random crop settings
    min_crop_size: int = 32
    max_crop_size: int = 128  # Reduced default for memory efficiency
    fixed_crop_size: Optional[int] = None  # If set, use fixed size instead of random
    max_samples_per_image: Optional[int] = None  # If set, cap samples per image to this value

    # Model settings
    num_classes: int = 3
    channels: int = 1
    base_filters: int = 32
    dropout: float = 0.5

    # Reproducibility
    seed: int = 42

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# Data Classes
# ==============================================================================


@dataclass
class FoldMetrics:
    """Store metrics for a single fold."""

    fold: int
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_accuracies: List[float] = field(default_factory=list)
    val_accuracies: List[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    stopped_early: bool = False


@dataclass
class PercentageMetrics:
    """Store metrics for a training percentage."""

    percentage: int
    fold_metrics: List[FoldMetrics] = field(default_factory=list)

    @property
    def mean_best_val_loss(self) -> float:
        """Calculate mean best validation loss across folds."""
        return np.mean([fm.best_val_loss for fm in self.fold_metrics])

    @property
    def std_best_val_loss(self) -> float:
        """Calculate std of best validation loss across folds."""
        return np.std([fm.best_val_loss for fm in self.fold_metrics])

    @property
    def mean_final_val_accuracy(self) -> float:
        """Calculate mean final validation accuracy across folds."""
        accuracies = [
            fm.val_accuracies[fm.best_epoch] if fm.val_accuracies else 0.0
            for fm in self.fold_metrics
        ]
        return np.mean(accuracies)

    @property
    def std_final_val_accuracy(self) -> float:
        """Calculate std of final validation accuracy across folds."""
        accuracies = [
            fm.val_accuracies[fm.best_epoch] if fm.val_accuracies else 0.0
            for fm in self.fold_metrics
        ]
        return np.std(accuracies)

    @property
    def mean_best_train_loss(self) -> float:
        """Calculate mean training loss at best epoch across folds."""
        losses = [
            fm.train_losses[fm.best_epoch] if fm.train_losses else float("inf")
            for fm in self.fold_metrics
        ]
        return np.mean(losses)

    @property
    def std_best_train_loss(self) -> float:
        """Calculate std of training loss at best epoch across folds."""
        losses = [
            fm.train_losses[fm.best_epoch] if fm.train_losses else float("inf")
            for fm in self.fold_metrics
        ]
        return np.std(losses)

    @property
    def mean_final_train_accuracy(self) -> float:
        """Calculate mean training accuracy at best epoch across folds."""
        accuracies = [
            fm.train_accuracies[fm.best_epoch] if fm.train_accuracies else 0.0
            for fm in self.fold_metrics
        ]
        return np.mean(accuracies)

    @property
    def std_final_train_accuracy(self) -> float:
        """Calculate std of training accuracy at best epoch across folds."""
        accuracies = [
            fm.train_accuracies[fm.best_epoch] if fm.train_accuracies else 0.0
            for fm in self.fold_metrics
        ]
        return np.std(accuracies)


# ==============================================================================
# Dataset
# ==============================================================================


class SEMDataset(Dataset):
    """SEM Image Dataset with random square crop and rotation augmentation."""

    def __init__(
        self,
        image_paths: List[Path],
        labels: List[int],
        min_crop_size: int = 32,
        max_crop_size: int = 128,
        fixed_crop_size: Optional[int] = None,
        augment: bool = True,
        random_rotation: bool = True,
        rotation_angles: Tuple[int, ...] = (0, 90, 180, 270),
        expand_dataset: bool = True,
        max_samples_per_image: Optional[int] = None,
    ) -> None:
        """
        Initialize the SEM dataset.

        Args:
            image_paths: List of paths to image files.
            labels: List of integer labels corresponding to each image.
            min_crop_size: Minimum size of random square crop.
            max_crop_size: Maximum size of random square crop.
            fixed_crop_size: If set, use this fixed size for all crops.
            augment: Whether to apply random crop augmentation.
            random_rotation: Whether to apply random rotation augmentation.
            rotation_angles: Tuple of possible rotation angles in degrees.
            expand_dataset: Whether to expand dataset size based on possible crops.
            max_samples_per_image: If set, cap the samples per image to this value.

        """
        self.image_paths = image_paths
        self.labels = labels
        self.min_crop_size = min_crop_size
        self.max_crop_size = max_crop_size
        self.fixed_crop_size = fixed_crop_size
        self.augment = augment
        self.random_rotation = random_rotation
        self.rotation_angles = rotation_angles
        self.expand_dataset = expand_dataset
        self.max_samples_per_image = max_samples_per_image

        # Calculate samples per image based on possible non-overlapping crops
        self.samples_per_image = self._calculate_samples_per_image()

    def _calculate_samples_per_image(self) -> int:
        """
        Calculate the number of unique samples that can be extracted from each image.

        Estimate based on non-overlapping crops at expected crop size.

        Returns:
            Number of samples per image.

        """
        if not self.augment or not self.expand_dataset or not self.image_paths:
            return 1

        # Get image dimensions from first image (assume all images same size)
        img = Image.open(self.image_paths[0])
        h, w = img.height, img.width

        # Expected crop size
        if self.fixed_crop_size is not None:
            expected_crop = self.fixed_crop_size
        else:
            expected_crop = (self.min_crop_size + self.max_crop_size) // 2

        # Number of non-overlapping crops in each dimension
        crops_h = max(1, h // expected_crop)
        crops_w = max(1, w // expected_crop)
        num_crops = crops_h * crops_w

        # Multiply by number of rotations if rotation is enabled
        num_rotations = len(self.rotation_angles) if self.random_rotation else 1

        calculated = num_crops * num_rotations

        # Cap at max_samples_per_image if specified
        if self.max_samples_per_image is not None:
            return min(calculated, self.max_samples_per_image)

        return calculated

    def __len__(self) -> int:
        """Return the number of samples (expanded if expand_dataset is True)."""
        return len(self.image_paths) * self.samples_per_image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image tensor, label).

        """
        # Map expanded index to original image index
        original_idx = idx % len(self.image_paths)

        # Load image
        img = Image.open(self.image_paths[original_idx]).convert("L")  # Grayscale
        img_array = np.array(img, dtype=np.float32) / 255.0

        if self.augment:
            img_array = self._random_square_crop(img_array)
            if self.random_rotation:
                img_array = self._random_rotate(img_array)
        else:
            # Use deterministic random crop for validation (variable size, reproducible)
            img_array = self._deterministic_random_crop(img_array, idx)

        # Convert to tensor: (H, W) -> (1, H, W)
        tensor = torch.from_numpy(img_array).unsqueeze(0)

        return tensor, self.labels[original_idx]

    def _random_rotate(self, img: np.ndarray) -> np.ndarray:
        """
        Apply random rotation to the image.

        Args:
            img: Input image array of shape (H, W).

        Returns:
            Rotated image array.

        """
        angle = random.choice(self.rotation_angles)
        if angle == 0:
            return img
        # np.rot90 rotates counter-clockwise, so we adjust for 90-degree increments
        k = angle // 90
        return np.ascontiguousarray(np.rot90(img, k=k))

    def _random_square_crop(self, img: np.ndarray) -> np.ndarray:
        """
        Extract a random square crop from the image.

        Args:
            img: Input image array of shape (H, W).

        Returns:
            Cropped image array.

        """
        h, w = img.shape

        # Determine crop size (must fit within image)
        if self.fixed_crop_size is not None:
            crop_size = min(self.fixed_crop_size, h, w)
        else:
            max_possible_size = min(h, w, self.max_crop_size)
            min_size = min(self.min_crop_size, max_possible_size)
            crop_size = random.randint(min_size, max_possible_size)

        # Random top-left corner
        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)

        return img[top : top + crop_size, left : left + crop_size]

    def _deterministic_random_crop(
        self,
        img: np.ndarray,
        idx: int,
    ) -> np.ndarray:
        """
        Extract a deterministic random square crop from the image.

        Use a seed based on the image index for reproducibility while still
        providing variable crop sizes and positions across the validation set.

        Args:
            img: Input image array of shape (H, W).
            idx: Sample index used for deterministic seeding.

        Returns:
            Cropped image array.

        """
        h, w = img.shape

        # Create a local random generator seeded by the index for reproducibility
        rng = random.Random(idx)

        # Determine crop size (must fit within image)
        if self.fixed_crop_size is not None:
            crop_size = min(self.fixed_crop_size, h, w)
        else:
            max_possible_size = min(h, w, self.max_crop_size)
            min_size = min(self.min_crop_size, max_possible_size)
            crop_size = rng.randint(min_size, max_possible_size)

        # Deterministic random position based on index
        top = rng.randint(0, h - crop_size)
        left = rng.randint(0, w - crop_size)

        return img[top : top + crop_size, left : left + crop_size]


def collate_variable_size(
    batch: List[Tuple[torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for variable-size images.

    Pad all images in the batch to the size of the largest image.

    Args:
        batch: List of (image, label) tuples.

    Returns:
        Tuple of (batched images, batched labels).

    """
    images, labels = zip(*batch)

    # Find max dimensions
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    # Pad images to max size
    padded_images = []
    for img in images:
        _, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        # Pad on right and bottom
        padded = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), value=0)
        padded_images.append(padded)

    return torch.stack(padded_images), torch.tensor(labels, dtype=torch.long)


# ==============================================================================
# Data Loading
# ==============================================================================


# Valid image extensions
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif"}


def load_dataset(config: ValidationConfig) -> Tuple[List[Path], List[int], Dict[str, int]]:
    """
    Load the dataset from disk.

    Args:
        config: Validation configuration.

    Returns:
        Tuple of (image_paths, labels, label_to_idx mapping).

    """
    labels_path = config.data_dir / config.labels_file

    image_paths = []
    labels = []
    label_to_idx: Dict[str, int] = {}

    with open(labels_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            label_str = row["label"]

            # Skip non-image files
            if not any(filename.lower().endswith(ext) for ext in _IMAGE_EXTENSIONS):
                continue

            # Build label mapping
            if label_str not in label_to_idx:
                label_to_idx[label_str] = len(label_to_idx)

            # Find image in subdirectory
            img_path = config.data_dir / label_str / filename
            if img_path.exists():
                image_paths.append(img_path)
                labels.append(label_to_idx[label_str])

    return image_paths, labels, label_to_idx


def create_kfold_splits(
    n_samples: int,
    n_folds: int,
    seed: int,
) -> List[Tuple[List[int], List[int]]]:
    """
    Create k-fold cross-validation splits.

    Args:
        n_samples: Total number of samples.
        n_folds: Number of folds.
        seed: Random seed for reproducibility.

    Returns:
        List of (train_indices, val_indices) tuples for each fold.

    """
    rng = random.Random(seed)
    indices = list(range(n_samples))
    rng.shuffle(indices)

    fold_size = n_samples // n_folds
    splits = []

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n_samples

        val_indices = indices[val_start:val_end]
        train_indices = indices[:val_start] + indices[val_end:]
        splits.append((train_indices, val_indices))

    return splits


# ==============================================================================
# Training
# ==============================================================================


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Args:
        model: The model to train.
        dataloader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to use.

    Returns:
        Tuple of (average loss, accuracy).

    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, return_logits=True)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """
    Validate the model.

    Args:
        model: The model to validate.
        dataloader: Validation data loader.
        criterion: Loss function.
        device: Device to use.

    Returns:
        Tuple of (average loss, accuracy).

    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images, return_logits=True)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train_fold(
    train_paths: List[Path],
    train_labels: List[int],
    val_paths: List[Path],
    val_labels: List[int],
    fold: int,
    config: ValidationConfig,
) -> FoldMetrics:
    """
    Train a single fold.

    Args:
        train_paths: Training image paths.
        train_labels: Training labels.
        val_paths: Validation image paths.
        val_labels: Validation labels.
        fold: Fold number.
        config: Validation configuration.

    Returns:
        FoldMetrics containing training history.

    """
    # Create datasets
    train_dataset = SEMDataset(
        train_paths,
        train_labels,
        min_crop_size=config.min_crop_size,
        max_crop_size=config.max_crop_size,
        fixed_crop_size=config.fixed_crop_size,
        augment=True,
        expand_dataset=True,  # Expand training set based on possible crops
        max_samples_per_image=config.max_samples_per_image,
    )
    val_dataset = SEMDataset(
        val_paths,
        val_labels,
        min_crop_size=config.min_crop_size,
        max_crop_size=config.max_crop_size,
        fixed_crop_size=config.fixed_crop_size,
        augment=False,  # No augmentation for validation - deterministic evaluation
        expand_dataset=False,  # No expansion for validation
    )

    # Log dataset sizes
    print(
        f"    Dataset sizes - Train: {len(train_paths)} images -> "
        f"{len(train_dataset)} samples ({train_dataset.samples_per_image}x expansion), "
        f"Val: {len(val_paths)} images -> {len(val_dataset)} samples",
    )
    # Log image dimensions for first image
    from PIL import Image as _PILImage
    _first_img = _PILImage.open(train_paths[0])
    _expected = (train_dataset.min_crop_size + train_dataset.max_crop_size) // 2
    print(
        f"    Image dimensions: {_first_img.width}x{_first_img.height}px, "
        f"Crop range: {train_dataset.min_crop_size}-{train_dataset.max_crop_size}px, "
        f"Expected crop: {_expected}px",
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_variable_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_variable_size,
    )

    # Create model
    model = CNNClassifier(
        num_classes=config.num_classes,
        channels=config.channels,
        base_filters=config.base_filters,
        dropout=config.dropout,
    ).to(config.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop with early stopping
    metrics = FoldMetrics(fold=fold)
    patience_counter = 0
    epoch = 0

    while True:
        # Check max_epochs limit
        if config.max_epochs is not None and epoch >= config.max_epochs:
            break

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            config.device,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, config.device)

        metrics.train_losses.append(train_loss)
        metrics.val_losses.append(val_loss)
        metrics.train_accuracies.append(train_acc)
        metrics.val_accuracies.append(val_acc)

        # Check for improvement
        improved = val_loss < metrics.best_val_loss
        if improved:
            metrics.best_val_loss = val_loss
            metrics.best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        # Log epoch progress
        max_epochs_str = str(config.max_epochs) if config.max_epochs else "∞"
        status = "✓ improved" if improved else f"patience {patience_counter}/{config.patience}"
        print(
            f"      Epoch {epoch + 1:3d}/{max_epochs_str} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:5.1f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:5.1f}% | {status}"
        )

        # Early stopping
        if patience_counter >= config.patience:
            metrics.stopped_early = True
            break

        epoch += 1

    return metrics


# ==============================================================================
# Main Validation Loop
# ==============================================================================


def run_validation(
    config: Optional[ValidationConfig] = None,
) -> Tuple[List[PercentageMetrics], plt.Figure]:
    """
    Run the full validation experiment.

    Args:
        config: Validation configuration. Uses defaults if None.

    Returns:
        Tuple of (list of PercentageMetrics, matplotlib Figure).

    """
    if config is None:
        config = ValidationConfig()

    # Set seed for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Load data
    print("Loading dataset...")
    image_paths, labels, label_to_idx = load_dataset(config)
    print(f"Loaded {len(image_paths)} images with {len(label_to_idx)} classes")
    print(f"Classes: {label_to_idx}")

    # Create k-fold splits
    kfold_splits = create_kfold_splits(len(image_paths), config.n_folds, config.seed)

    all_metrics: List[PercentageMetrics] = []

    for percentage in config.train_percentages:
        print(f"\n{'='*60}")
        print(f"Training with {percentage}% of data")
        print(f"{'='*60}")

        pct_metrics = PercentageMetrics(percentage=percentage)

        for fold, (train_indices, val_indices) in enumerate(kfold_splits):
            print(f"\n  Fold {fold + 1}/{config.n_folds}")

            # Subsample training data by percentage
            n_train_samples = int(len(train_indices) * percentage / 100)
            rng = random.Random(config.seed + fold)
            sampled_train_indices = rng.sample(train_indices, n_train_samples)

            # Get paths and labels for this fold
            train_paths = [image_paths[i] for i in sampled_train_indices]
            train_labels_fold = [labels[i] for i in sampled_train_indices]
            val_paths = [image_paths[i] for i in val_indices]
            val_labels_fold = [labels[i] for i in val_indices]

            # Train this fold
            fold_metrics = train_fold(
                train_paths,
                train_labels_fold,
                val_paths,
                val_labels_fold,
                fold,
                config,
            )

            pct_metrics.fold_metrics.append(fold_metrics)

            print(f"    Best val loss: {fold_metrics.best_val_loss:.4f} at epoch {fold_metrics.best_epoch + 1}")
            if fold_metrics.stopped_early:
                print(f"    Early stopped after {len(fold_metrics.train_losses)} epochs")

        all_metrics.append(pct_metrics)

        print(f"\n  {percentage}% Summary:")
        print(f"    Mean best val loss: {pct_metrics.mean_best_val_loss:.4f} ± {pct_metrics.std_best_val_loss:.4f}")
        print(f"    Mean val accuracy: {pct_metrics.mean_final_val_accuracy:.4f} ± {pct_metrics.std_final_val_accuracy:.4f}")

    # Generate plots
    fig = plot_results(all_metrics, config)

    return all_metrics, fig


# ==============================================================================
# Plotting
# ==============================================================================


def plot_results(
    all_metrics: List[PercentageMetrics],
    config: ValidationConfig,
) -> plt.Figure:
    """
    Generate visualization of the validation results.

    Args:
        all_metrics: List of PercentageMetrics from the validation run.
        config: Validation configuration.

    Returns:
        Matplotlib Figure with the plots.

    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    percentages = [pm.percentage for pm in all_metrics]

    # Plot 1: Mean loss (train & val) vs training percentage
    ax1 = axes[0, 0]
    # Validation loss (blue)
    mean_val_losses = [pm.mean_best_val_loss for pm in all_metrics]
    std_val_losses = [pm.std_best_val_loss for pm in all_metrics]
    ax1.errorbar(
        percentages,
        mean_val_losses,
        yerr=std_val_losses,
        marker="o",
        color="blue",
        capsize=5,
        capthick=2,
        label="Validation",
    )
    # Training loss (red)
    mean_train_losses = [pm.mean_best_train_loss for pm in all_metrics]
    std_train_losses = [pm.std_best_train_loss for pm in all_metrics]
    ax1.errorbar(
        percentages,
        mean_train_losses,
        yerr=std_train_losses,
        marker="s",
        color="red",
        capsize=5,
        capthick=2,
        label="Training",
    )
    ax1.set_xlabel("Training Data Percentage (%)")
    ax1.set_ylabel("Loss (at best epoch)")
    ax1.set_title("Loss vs Training Data Size")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Mean accuracy (train & val) vs training percentage
    ax2 = axes[0, 1]
    # Validation accuracy (blue)
    mean_val_accs = [pm.mean_final_val_accuracy * 100 for pm in all_metrics]
    std_val_accs = [pm.std_final_val_accuracy * 100 for pm in all_metrics]
    ax2.errorbar(
        percentages,
        mean_val_accs,
        yerr=std_val_accs,
        marker="o",
        color="blue",
        capsize=5,
        capthick=2,
        label="Validation",
    )
    # Training accuracy (red)
    mean_train_accs = [pm.mean_final_train_accuracy * 100 for pm in all_metrics]
    std_train_accs = [pm.std_final_train_accuracy * 100 for pm in all_metrics]
    ax2.errorbar(
        percentages,
        mean_train_accs,
        yerr=std_train_accs,
        marker="s",
        color="red",
        capsize=5,
        capthick=2,
        label="Training",
    )
    ax2.set_xlabel("Training Data Percentage (%)")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy vs Training Data Size")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Loss curves for lowest percentage (train & val together)
    ax3 = axes[1, 0]
    pm_low = all_metrics[0]  # Lowest percentage
    max_epochs = max(len(fm.train_losses) for fm in pm_low.fold_metrics)
    avg_train_losses = []
    avg_val_losses = []
    for epoch in range(max_epochs):
        train_losses = [
            fm.train_losses[epoch]
            for fm in pm_low.fold_metrics
            if epoch < len(fm.train_losses)
        ]
        val_losses = [
            fm.val_losses[epoch]
            for fm in pm_low.fold_metrics
            if epoch < len(fm.val_losses)
        ]
        if train_losses:
            avg_train_losses.append(np.mean(train_losses))
        if val_losses:
            avg_val_losses.append(np.mean(val_losses))
    ax3.plot(
        range(1, len(avg_train_losses) + 1),
        avg_train_losses,
        color="red",
        label="Training",
    )
    ax3.plot(
        range(1, len(avg_val_losses) + 1),
        avg_val_losses,
        color="blue",
        label="Validation",
    )
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.set_title(f"Loss Curves ({pm_low.percentage}% Training Data)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Loss curves for highest percentage (train & val together)
    ax4 = axes[1, 1]
    pm_high = all_metrics[-1]  # Highest percentage
    max_epochs = max(len(fm.train_losses) for fm in pm_high.fold_metrics)
    avg_train_losses = []
    avg_val_losses = []
    for epoch in range(max_epochs):
        train_losses = [
            fm.train_losses[epoch]
            for fm in pm_high.fold_metrics
            if epoch < len(fm.train_losses)
        ]
        val_losses = [
            fm.val_losses[epoch]
            for fm in pm_high.fold_metrics
            if epoch < len(fm.val_losses)
        ]
        if train_losses:
            avg_train_losses.append(np.mean(train_losses))
        if val_losses:
            avg_val_losses.append(np.mean(val_losses))
    ax4.plot(
        range(1, len(avg_train_losses) + 1),
        avg_train_losses,
        color="red",
        label="Training",
    )
    ax4.plot(
        range(1, len(avg_val_losses) + 1),
        avg_val_losses,
        color="blue",
        label="Validation",
    )
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Loss")
    ax4.set_title(f"Loss Curves ({pm_high.percentage}% Training Data)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ==============================================================================
# Entry Point
# ==============================================================================


def main(
    patience: int = 5,
    n_folds: int = 5,
    max_epochs: Optional[int] = None,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    min_crop_size: int = 32,
    max_crop_size: int = 128,
    fixed_crop_size: Optional[int] = None,
    max_samples_per_image: Optional[int] = None,
    seed: int = 42,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Run validation and return the results graph.

    Args:
        patience: Number of epochs without improvement before early stopping.
        n_folds: Number of folds for cross-validation.
        max_epochs: Maximum number of epochs per fold. If None, train until patience runs out.
        batch_size: Batch size for training.
        learning_rate: Learning rate for optimizer.
        min_crop_size: Minimum crop size for random cropping.
        max_crop_size: Maximum crop size for random cropping.
        fixed_crop_size: If set, use fixed crop size (more memory efficient).
        max_samples_per_image: If set, cap the samples per image to this value.
        seed: Random seed for reproducibility.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure with the validation results.

    """
    config = ValidationConfig(
        patience=patience,
        n_folds=n_folds,
        max_epochs=max_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        min_crop_size=min_crop_size,
        max_crop_size=max_crop_size,
        fixed_crop_size=fixed_crop_size,
        max_samples_per_image=max_samples_per_image,
        seed=seed,
    )

    all_metrics, fig = run_validation(config)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to {save_path}")

    return fig


if __name__ == "__main__":
    figure = main()
    plt.show()
