"""Visualization utilities for training and validation results."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from auto_ml.interfaces import MaskPair, SegmentationDatasetInterface
from model.swin.metrics import PercentageMetrics


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

    # Color map for masks (3 classes)
    cmap = plt.cm.get_cmap("viridis", 3)

    for i in range(num_samples):
        image = test_dataset.images[i]
        predicted_mask, real_mask = mask_pairs[i]

        # Input image
        axes[i, 0].imshow(image, cmap="gray")
        axes[i, 0].set_title("Input" if i == 0 else "")
        axes[i, 0].axis("off")

        # Ground truth mask
        axes[i, 1].imshow(real_mask, cmap=cmap, vmin=0, vmax=2)
        axes[i, 1].set_title("Ground Truth" if i == 0 else "")
        axes[i, 1].axis("off")

        # Predicted mask
        axes[i, 2].imshow(predicted_mask, cmap=cmap, vmin=0, vmax=2)
        axes[i, 2].set_title("Predicted" if i == 0 else "")
        axes[i, 2].axis("off")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved predictions visualization to {output_path}")

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
