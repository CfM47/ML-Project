"""Visualization utilities for training and validation results."""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from auto_ml.interfaces import MaskPair, SegmentationDatasetInterface
from model.swin.metrics import PercentageMetrics, TrainingHistory

# Default colors for 3-class segmentation (RGB, 0-1 range)
DEFAULT_MASK_COLORS: List[Tuple[float, float, float]] = [
    (1.0, 0.2, 0.2),  # Class 0: Red
    (0.2, 0.6, 1.0),  # Class 1: Blue
    (0.0, 0.0, 0.0),  # Class 2: Black (background)
]

DEFAULT_MASK_ALPHA = 0.6  # Overlay opacity


def _plot_mask(
    ax: Axes,
    mask: np.ndarray,
    underlay: np.ndarray,
    colors: List[Tuple[float, float, float]] | None = None,
    alpha: float = DEFAULT_MASK_ALPHA,
) -> None:
    """
    Plot a segmentation mask overlaid on an underlay image.

    Args:
        ax: Matplotlib axes to plot on.
        mask: Segmentation mask with integer class labels (0, 1, 2).
        underlay: Grayscale image to use as background.
        colors: List of RGB tuples (0-1 range) for each class.
            Defaults to DEFAULT_MASK_COLORS.
        alpha: Opacity of the mask overlay (0-1). Defaults to DEFAULT_MASK_ALPHA.

    """
    if colors is None:
        colors = DEFAULT_MASK_COLORS

    # Normalize underlay to 0-1 range
    if underlay.max() > 1:
        underlay_norm = underlay.astype(np.float32) / 255.0
    else:
        underlay_norm = underlay.astype(np.float32)

    # Convert grayscale to RGB if needed
    if underlay_norm.ndim == 2:
        underlay_rgb = np.stack([underlay_norm] * 3, axis=-1)
    else:
        underlay_rgb = underlay_norm

    # Create colored mask overlay
    h, w = mask.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.float32)
    for class_idx, color in enumerate(colors):
        class_mask = mask == class_idx
        for c in range(3):
            mask_rgb[:, :, c] += class_mask * color[c]

    # Blend underlay and mask
    blended = (1 - alpha) * underlay_rgb + alpha * mask_rgb
    blended = np.clip(blended, 0, 1)

    ax.imshow(blended)
    ax.axis("off")


def plot_training_loss_curves(
    history: TrainingHistory,
    output_path: Path | None = None,
) -> Figure:
    """
    Plot training and validation loss curves for a single training run.

    Args:
        history: Validated TrainingHistory dataclass.
        output_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure.

    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(history.epochs, history.train_losses, color="red", label="Training")
    ax.plot(history.epochs, history.val_losses, color="blue", label="Validation")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved training loss curves to {output_path}")

    return fig


def plot_results(
    all_metrics: List[PercentageMetrics],
    output_path: Path | None = None,
) -> Figure:
    """
    Generate 4-panel learning curve figure.

    Layout:
    - Top-left: Loss vs training percentage (train & val)
    - Top-right: F1 & Accuracy vs training percentage (val only)
    - Bottom-left: Loss curves for lowest percentage (10%)
    - Bottom-right: Loss curves for highest percentage (80%)

    Args:
        all_metrics: List of PercentageMetrics from validation run.
        output_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure.

    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    percentages = [pm.percentage for pm in all_metrics]

    # --- Top-left: Loss vs training percentage ---
    ax1 = axes[0, 0]
    _plot_loss_vs_percentage(ax1, all_metrics, percentages)

    # --- Top-right: F1 & Accuracy vs training percentage ---
    ax2 = axes[0, 1]
    _plot_metrics_vs_percentage(ax2, all_metrics, percentages)

    # --- Bottom-left: Loss curves for lowest percentage ---
    ax3 = axes[1, 0]
    pm_low = all_metrics[0]
    _plot_loss_curves(ax3, pm_low, f"Loss Curves ({pm_low.percentage}% Training Data)")

    # --- Bottom-right: Loss curves for highest percentage ---
    ax4 = axes[1, 1]
    pm_high = all_metrics[-1]
    _plot_loss_curves(
        ax4,
        pm_high,
        f"Loss Curves ({pm_high.percentage}% Training Data)",
    )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved learning curves to {output_path}")

    return fig


def visualize_predictions(
    test_dataset: SegmentationDatasetInterface,
    mask_pairs: List[MaskPair],
    num_samples: int,
    output_path: Path | None = None,
) -> Figure:
    """
    Create side-by-side visualization grid.

    Show Input | Ground Truth | Predicted for num_samples.

    Args:
        test_dataset: Test dataset with original images.
        mask_pairs: List of (predicted_mask, real_mask) tuples.
        num_samples: Number of samples to visualize.
        output_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure.

    """
    num_samples = min(num_samples, len(mask_pairs))

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    # Handle single sample case
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        image = test_dataset.images[i]
        predicted_mask, real_mask = mask_pairs[i]

        # Input image
        axes[i, 0].imshow(image, cmap="gray")
        axes[i, 0].set_title("Input" if i == 0 else "")
        axes[i, 0].axis("off")

        # Ground truth mask (with underlay)
        _plot_mask(axes[i, 1], real_mask, image)
        axes[i, 1].set_title("Ground Truth" if i == 0 else "")

        # Predicted mask (with underlay)
        _plot_mask(axes[i, 2], predicted_mask, image)
        axes[i, 2].set_title("Predicted" if i == 0 else "")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved predictions visualization to {output_path}")

    return fig


def plot_progression_grid(
    test_dataset: SegmentationDatasetInterface,
    predictions_by_percentage: Dict[int, List[np.ndarray]],
    sample_indices: List[int],
    output_path: Path | None = None,
) -> Figure:
    """
    Create progression grid showing predictions across training percentages.

    Layout:
    - Columns: [10%, 20%, ..., 80%, Ground Truth, Original]
    - Rows: One per sample

    Args:
        test_dataset: Test dataset with original images and masks.
        predictions_by_percentage: Dict mapping percentage to list of predicted masks.
        sample_indices: Indices of samples in test_dataset that were predicted.
        output_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure.

    """
    percentages = sorted(predictions_by_percentage.keys())
    num_samples = len(sample_indices)
    num_cols = len(percentages) + 2  # +2 for ground truth and original

    fig, axes = plt.subplots(
        num_samples,
        num_cols,
        figsize=(2 * num_cols, 2 * num_samples),
    )

    # Handle single sample case
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(sample_indices):
        image = test_dataset.images[idx]
        ground_truth = test_dataset.masks[idx]

        # Predictions for each percentage (with underlay)
        for col, pct in enumerate(percentages):
            pred = predictions_by_percentage[pct][row]
            _plot_mask(axes[row, col], pred, image)
            if row == 0:
                axes[row, col].set_title(f"{pct}%", fontsize=10)

        # Ground truth column (with underlay)
        gt_col = len(percentages)
        _plot_mask(axes[row, gt_col], ground_truth, image)
        if row == 0:
            axes[row, gt_col].set_title("Ground Truth", fontsize=10)

        # Original image column
        orig_col = len(percentages) + 1
        axes[row, orig_col].imshow(image, cmap="gray")
        if row == 0:
            axes[row, orig_col].set_title("Original", fontsize=10)
        axes[row, orig_col].axis("off")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved progression grid to {output_path}")

    return fig


# --- Helper Functions ---


def _plot_loss_vs_percentage(
    ax: Axes,
    all_metrics: List[PercentageMetrics],
    percentages: List[int],
) -> None:
    """Plot train and validation loss vs training percentage."""
    # Validation loss (blue)
    mean_val_losses = [pm.mean_final_val_loss for pm in all_metrics]
    std_val_losses = [pm.std_final_val_loss for pm in all_metrics]
    ax.errorbar(
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
    mean_train_losses = [pm.mean_final_train_loss for pm in all_metrics]
    std_train_losses = [pm.std_final_train_loss for pm in all_metrics]
    ax.errorbar(
        percentages,
        mean_train_losses,
        yerr=std_train_losses,
        marker="s",
        color="red",
        capsize=5,
        capthick=2,
        label="Training",
    )

    ax.set_xlabel("Training Data Percentage (%)")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Training Data Size")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_metrics_vs_percentage(
    ax: Axes,
    all_metrics: List[PercentageMetrics],
    percentages: List[int],
) -> None:
    """Plot F1 and Accuracy vs training percentage (validation only)."""
    # F1 (Dice Macro) - blue
    mean_f1 = [pm.mean_dice_macro * 100 for pm in all_metrics]
    std_f1 = [pm.std_dice_macro * 100 for pm in all_metrics]
    ax.errorbar(
        percentages,
        mean_f1,
        yerr=std_f1,
        marker="o",
        color="blue",
        capsize=5,
        capthick=2,
        label="F1 (Dice Macro)",
    )

    # Accuracy - green
    mean_acc = [pm.mean_accuracy * 100 for pm in all_metrics]
    std_acc = [pm.std_accuracy * 100 for pm in all_metrics]
    ax.errorbar(
        percentages,
        mean_acc,
        yerr=std_acc,
        marker="s",
        color="green",
        capsize=5,
        capthick=2,
        label="Accuracy",
    )

    ax.set_xlabel("Training Data Percentage (%)")
    ax.set_ylabel("Score (%)")
    ax.set_title("F1 & Accuracy vs Training Data Size")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_loss_curves(
    ax: Axes,
    pm: PercentageMetrics,
    title: str,
) -> None:
    """Plot averaged loss curves over epochs for a percentage."""
    if not pm.fold_metrics:
        return

    max_epochs = max(len(fm.train_history) for fm in pm.fold_metrics)

    avg_train_losses = []
    avg_val_losses = []

    for epoch in range(max_epochs):
        train_losses = [
            fm.train_history[epoch].get("train_loss", float("nan"))
            for fm in pm.fold_metrics
            if epoch < len(fm.train_history)
        ]
        val_losses = [
            fm.train_history[epoch].get("val_loss", float("nan"))
            for fm in pm.fold_metrics
            if epoch < len(fm.train_history)
        ]

        if train_losses:
            avg_train_losses.append(np.nanmean(train_losses))
        if val_losses:
            avg_val_losses.append(np.nanmean(val_losses))

    epochs = range(1, len(avg_train_losses) + 1)

    ax.plot(epochs, avg_train_losses, color="red", label="Training")
    ax.plot(
        range(1, len(avg_val_losses) + 1),
        avg_val_losses,
        color="blue",
        label="Validation",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_metric_histograms(
    subset_distributions: Dict[str, List[float]],
    full_metrics: Dict[str, float],
    output_dir: Path | None = None,
) -> Dict[str, Figure]:
    """
    Create individual histogram for each metric showing leave-two-out distribution.

    Each histogram shows the distribution of metric values across all subsets,
    with a vertical line indicating the full test set metric value.

    Args:
        subset_distributions: Dict mapping metric name to list of subset values.
        full_metrics: Dict mapping metric name to full test set value.
        output_dir: Optional directory to save histogram files.

    Returns:
        Dict mapping metric name to Figure.

    """
    histograms: Dict[str, Figure] = {}

    for metric_name, values in subset_distributions.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot histogram
        ax.hist(
            values,
            bins=20,
            color="steelblue",
            edgecolor="white",
            alpha=0.7,
        )

        # Add vertical line for full test set value
        if metric_name in full_metrics:
            full_value = full_metrics[metric_name]
            ax.axvline(
                full_value,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Full test set: {full_value:.4f}",
            )

        # Calculate and display statistics
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        ax.axvline(
            mean_val,
            color="green",
            linestyle="-",
            linewidth=2,
            label=f"Mean: {mean_val:.4f} (std: {std_val:.4f})",
        )

        ax.set_xlabel(metric_name)
        ax.set_ylabel("Frequency")
        ax.set_title(f"{metric_name} Distribution (Leave-Two-Out)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save if output directory provided
        if output_dir:
            output_path = output_dir / f"histogram_{metric_name}.png"
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved histogram to {output_path}")

        histograms[metric_name] = fig

    return histograms
