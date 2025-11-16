from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import np
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Subset

from src.approaches.cnn_v1.configs.schemas import ApproachConfig
from src.approaches.cnn_v1.data.datasets import get_data, get_dataloader
from src.approaches.cnn_v1.schemas import CrossValidationMetrics
from src.approaches.cnn_v1.train import run_training_loop, setup_training
from src.utils.config import load_config


def validate(config_name: str = "base") -> None:
    """
    Run 4-fold cross-validation.

    Uses stratified subsets of the data of increasing size.
    """
    config_path = Path(__file__).parent / "configs" / f"{config_name}.yaml"
    config = load_config(config_path, ApproachConfig)
    results: List[CrossValidationMetrics] = []

    # Get the full dataset once
    full_train_dataset, full_val_dataset = get_data(config)
    all_indices = np.arange(len(full_train_dataset))
    all_targets = np.array(full_train_dataset.targets)

    start_percent = config.validation.start_train_percentage / 100
    end_percent = config.validation.end_train_percentage / 100
    percent_step = config.validation.train_percentage_step / 100

    # Percentages of the total dataset to use for each experiment run
    active_data_percentages = np.arange(
        start_percent,
        end_percent,
        percent_step,
    ).tolist()
    k_folds = config.validation.folds

    for percent in active_data_percentages:
        print("\n" + "=" * 60)
        print(f"Running {k_folds}-Fold CV on {percent:.0%} of the total data")
        print("=" * 60)

        # 1. Create a stratified subset of the data for this run
        active_indices, _ = train_test_split(
            all_indices,
            train_size=percent,
            stratify=all_targets,
            random_state=42,
        )
        active_targets = all_targets[active_indices]

        fold_val_losses, fold_train_losses = [], []
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

        # 2. Perform k-fold cross-validation on this subset
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(active_indices, active_targets),
        ):
            print(f"\n--- Fold {fold + 1}/{k_folds} ---")

            # Map the fold indices from the 'active' subset back to the original dataset
            # indices
            train_fold_original_indices = active_indices[train_idx]
            val_fold_original_indices = active_indices[val_idx]

            # Create subsets and dataloaders for the current fold
            train_subset: Subset[Any] = Subset(
                full_train_dataset,
                train_fold_original_indices,
            )
            val_subset: Subset[Any] = Subset(
                full_val_dataset,
                val_fold_original_indices,
            )
            train_loader = get_dataloader(config, train_subset, train=True)
            val_loader = get_dataloader(config, val_subset, train=False)

            # Setup a fresh model for each fold
            ctx = setup_training(config)

            # Run the training loop
            best_val_loss, final_train_loss = run_training_loop(
                ctx,
                config,
                train_loader,
                val_loader,
            )
            fold_val_losses.append(best_val_loss)
            fold_train_losses.append(final_train_loss)

        # 3. Average the results for the current data percentage
        avg_val_loss = float(np.mean(fold_val_losses))
        avg_train_loss = float(np.mean(fold_train_losses))
        results.append(
            CrossValidationMetrics(
                train_percent=percent,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
            ),
        )
        print(f"\nAverage Val Loss for {percent:.0%} data: {avg_val_loss:.4f}")

    plot_loss_evolution(results)


def plot_loss_evolution(results: List[CrossValidationMetrics]) -> None:
    """Plot the evolution of train/validation loss and save the figure."""
    if not results:
        print("No results to plot.")
        return

    # Convert Pydantic models to a list of dicts for DataFrame creation
    df = pd.DataFrame(results)

    plt.figure(figsize=(12, 8))

    # Plot both losses
    plt.plot(
        df["train_percent"] * 100,
        df["train_loss"],
        marker="o",
        linewidth=2,
        markersize=8,
        label="Training Loss",
        color="#2E86AB",
    )
    plt.plot(
        df["train_percent"] * 100,
        df["val_loss"],
        marker="s",
        linewidth=2,
        markersize=8,
        label="Validation Loss",
        color="#A23B72",
    )

    plt.xlabel("Training Data Size (%)", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(
        "Training and Validation Loss Evolution by Dataset Size",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(15, 85)

    # Add annotations for the best validation loss point
    if not df.empty:
        min_val_loss_idx = df["val_loss"].idxmin()
        min_val_loss_point = df.iloc[min_val_loss_idx]

        plt.annotate(
            f"Best Val Loss: {min_val_loss_point['val_loss']:.4f} "
            f"at {min_val_loss_point['train_percent'] * 100:.0f}%",
            xy=(
                min_val_loss_point["train_percent"] * 100,
                min_val_loss_point["val_loss"],
            ),
            xytext=(
                min_val_loss_point["train_percent"] * 100 + 10,
                min_val_loss_point["val_loss"] + 0.02,
            ),
            arrowprops=dict(arrowstyle="->", color="black", alpha=0.7),
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

    plt.tight_layout()

    # Save the figure instead of showing it
    output_dir = Path(".data") / "cnn_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "validation_loss_evolution.png"
    plt.savefig(save_path)
    plt.close()  # Close the figure to free up memory

    print(f"\nSaved loss evolution plot to {save_path}")
