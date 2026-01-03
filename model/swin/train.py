"""
Swin Segmentation Training and Validation Module.

Provide two entry points:
1. run_percentage_validation: K-fold cross-validation with varying training percentages
2. run_final_training: Train on 100% data and evaluate on test set
"""

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from matplotlib.figure import Figure

from auto_ml.implementations import SwinModel, load_dataset_from_directories
from auto_ml.interfaces import MaskPair, SegmentationDatasetInterface
from model.swin.config import SwinTrainingConfig
from model.swin.data import create_augmentator, create_kfold_splits, subsample_dataset
from model.swin.evaluation import create_evaluator, evaluate_model
from model.swin.metrics import FoldMetrics, PercentageMetrics
from model.swin.visualization import plot_results, visualize_predictions

# ==============================================================================
# Entry Points
# ==============================================================================


def run_percentage_validation(
    train_unlabeled_dir: str | Path,
    train_labeled_dir: str | Path,
    config: SwinTrainingConfig | None = None,
) -> Tuple[List[PercentageMetrics], Figure]:
    """
    Run learning curve analysis with k-fold cross-validation.

    For each training percentage, run k-fold CV and aggregate metrics.
    Save plots to output directory.

    Args:
        train_unlabeled_dir: Directory with unlabeled training images.
        train_labeled_dir: Directory with labeled training masks.
        config: Training configuration. Uses defaults if None.

    Returns:
        Tuple of (list of PercentageMetrics, learning curves Figure).

    """
    if config is None:
        config = SwinTrainingConfig()

    # Setup output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("Loading training dataset...")
    train_dataset = load_dataset_from_directories(
        Path(train_unlabeled_dir),
        Path(train_labeled_dir),
    )
    print(f"Loaded {len(train_dataset)} training samples")

    # Run validation for each percentage
    all_metrics = _run_validation_loop(train_dataset, config)

    # Generate and save plots
    plot_path = config.output_dir / "learning_curves.png"
    fig = plot_results(all_metrics, output_path=plot_path)

    return all_metrics, fig


def run_final_training(
    train_unlabeled_dir: str | Path,
    train_labeled_dir: str | Path,
    test_unlabeled_dir: str | Path,
    test_labeled_dir: str | Path,
    config: SwinTrainingConfig | None = None,
) -> Tuple[SwinModel, Dict[str, float], List[MaskPair], Figure]:
    """
    Train final model on 100% data and evaluate on test set.

    Save model weights and prediction visualizations to output directory.

    Args:
        train_unlabeled_dir: Directory with unlabeled training images.
        train_labeled_dir: Directory with labeled training masks.
        test_unlabeled_dir: Directory with unlabeled test images.
        test_labeled_dir: Directory with labeled test masks.
        config: Training configuration. Uses defaults if None.

    Returns:
        Tuple of (trained_model, test_metrics, test_mask_pairs, predictions Figure).

    """
    if config is None:
        config = SwinTrainingConfig()

    # Setup output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("Loading training dataset...")
    train_dataset = load_dataset_from_directories(
        Path(train_unlabeled_dir),
        Path(train_labeled_dir),
    )
    print(f"Loaded {len(train_dataset)} training samples")

    print("Loading test dataset...")
    test_dataset = load_dataset_from_directories(
        Path(test_unlabeled_dir),
        Path(test_labeled_dir),
    )
    print(f"Loaded {len(test_dataset)} test samples")

    # Train final model
    model = _train_final_model(train_dataset, config)

    # Evaluate on test set
    test_metrics, test_mask_pairs = _evaluate_on_test(model, test_dataset)

    # Save model
    model_path = config.output_dir / "model.pt"
    _save_model(model, model_path)

    # Visualize predictions
    viz_path = config.output_dir / "test_predictions.png"
    fig = visualize_predictions(
        test_dataset,
        test_mask_pairs,
        num_samples=config.num_test_visualizations,
        output_path=viz_path,
    )

    return model, test_metrics, test_mask_pairs, fig


# ==============================================================================
# Validation Loop
# ==============================================================================


def _run_validation_loop(
    train_dataset: SegmentationDatasetInterface,
    config: SwinTrainingConfig,
) -> List[PercentageMetrics]:
    """Run learning curve analysis for all training percentages."""
    all_metrics: List[PercentageMetrics] = []

    for percentage in config.train_percentages:
        print(f"\n{'=' * 60}")
        print(f"Training with {percentage}% of data")
        print(f"{'=' * 60}")

        pct_metrics = _run_percentage_experiment(train_dataset, percentage, config)
        all_metrics.append(pct_metrics)

        print(f"\n  {percentage}% Summary:")
        print(
            f"    Mean F1 (Dice): {pct_metrics.mean_dice_macro:.4f} "
            f"± {pct_metrics.std_dice_macro:.4f}",
        )
        print(
            f"    Mean Accuracy: {pct_metrics.mean_accuracy:.4f} "
            f"± {pct_metrics.std_accuracy:.4f}",
        )

    return all_metrics


def _run_percentage_experiment(
    dataset: SegmentationDatasetInterface,
    percentage: int,
    config: SwinTrainingConfig,
) -> PercentageMetrics:
    """Run k-fold cross-validation for a single training percentage."""
    # Subsample dataset
    subsampled = subsample_dataset(dataset, percentage, config.seed)
    print(f"  Subsampled to {len(subsampled)} samples ({percentage}%)")

    # Create k-fold splits
    splits = create_kfold_splits(subsampled, config.n_folds, config.seed)

    pct_metrics = PercentageMetrics(percentage=percentage)

    for fold, (train_split, val_split) in enumerate(splits):
        print(f"\n  Fold {fold + 1}/{config.n_folds}")
        fold_metrics = _train_fold(train_split, val_split, fold, config)
        pct_metrics.fold_metrics.append(fold_metrics)

        print(
            f"    F1 (Dice): {fold_metrics.dice_macro:.4f}, "
            f"Accuracy: {fold_metrics.accuracy:.4f}",
        )

    return pct_metrics


def _train_fold(
    train_dataset: SegmentationDatasetInterface,
    val_dataset: SegmentationDatasetInterface,
    fold: int,
    config: SwinTrainingConfig,
) -> FoldMetrics:
    """
    Train and evaluate a single fold.

    1. Apply augmentation to training data (not validation)
    2. Create and train SwinModel
    3. Evaluate on validation set
    4. Return metrics

    """
    # Apply augmentation to training data only
    augmentator = create_augmentator(num_copies=config.augmentation_copies)
    aug_train = augmentator.augment(train_dataset)
    print(f"    Training: {len(train_dataset)} -> {len(aug_train)} samples (augmented)")
    print(f"    Validation: {len(val_dataset)} samples (no augmentation)")

    # Create model
    model = _create_swin_model(config)

    # Train model
    train_result = model.train(aug_train, validation_dataset=val_dataset)

    # Evaluate on validation set
    evaluator = create_evaluator()
    metrics, _ = evaluate_model(model, val_dataset, evaluator)

    return FoldMetrics(
        fold=fold,
        train_history=train_result.history,
        dice_macro=metrics.get("Dice_Macro", 0.0),
        accuracy=metrics.get("Accuracy", 0.0),
    )


# ==============================================================================
# Final Training
# ==============================================================================


def _train_final_model(
    train_dataset: SegmentationDatasetInterface,
    config: SwinTrainingConfig,
) -> SwinModel:
    """Train final model with 80/20 train/val split and augmentation."""
    print("\n" + "=" * 60)
    print("Training final model")
    print("=" * 60)

    # Split into train/val (80/20)
    train_split, val_split = train_dataset.split(
        ratio=0.8,
        shuffle=True,
        random_seed=config.seed,
    )
    print(f"Split: {len(train_split)} train, {len(val_split)} validation")

    # Apply augmentation to training data only
    augmentator = create_augmentator(num_copies=config.augmentation_copies)
    aug_train = augmentator.augment(train_split)
    print(f"Training: {len(train_split)} -> {len(aug_train)} samples (augmented)")

    # Create and train model with validation for early stopping
    model = _create_swin_model(config)
    model.train(aug_train, validation_dataset=val_split)

    return model


def _evaluate_on_test(
    model: SwinModel,
    test_dataset: SegmentationDatasetInterface,
) -> Tuple[Dict[str, float], List[MaskPair]]:
    """Evaluate trained model on test dataset (no augmentation)."""
    print("\n" + "=" * 60)
    print("Evaluating on test set")
    print("=" * 60)

    evaluator = create_evaluator()
    metrics, mask_pairs = evaluate_model(model, test_dataset, evaluator)

    print(f"Test F1 (Dice): {metrics.get('Dice_Macro', 0.0):.4f}")
    print(f"Test Accuracy: {metrics.get('Accuracy', 0.0):.4f}")

    return metrics, mask_pairs


# ==============================================================================
# Model Utilities
# ==============================================================================


def _create_swin_model(config: SwinTrainingConfig) -> SwinModel:
    """Create a fresh SwinModel instance from config."""
    return SwinModel(
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.learning_rate,
        embed_dim=config.embed_dim,
        depths=config.depths,
        num_heads=config.num_heads,
        patience=config.patience,
        device=config.device,
    )


def _save_model(model: SwinModel, path: Path) -> None:
    """Save model state dict to disk."""
    torch.save(model.model.state_dict(), path)
    print(f"Saved model to {path}")


# ==============================================================================
# CLI Entry Point
# ==============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Swin model training and validation")
    parser.add_argument(
        "mode",
        choices=["validate", "train"],
        help="Mode: 'validate' for percentage validation, 'train' for final training",
    )
    parser.add_argument(
        "--train-unlabeled",
        required=True,
        help="Path to unlabeled training images",
    )
    parser.add_argument(
        "--train-labeled",
        required=True,
        help="Path to labeled training masks",
    )
    parser.add_argument(
        "--test-unlabeled",
        help="Path to unlabeled test images (required for 'train' mode)",
    )
    parser.add_argument(
        "--test-labeled",
        help="Path to labeled test masks (required for 'train' mode)",
    )
    parser.add_argument(
        "--output-dir",
        default="model/swin/results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    config = SwinTrainingConfig(output_dir=Path(args.output_dir))

    if args.mode == "validate":
        run_percentage_validation(
            args.train_unlabeled,
            args.train_labeled,
            config=config,
        )
    elif args.mode == "train":
        if not args.test_unlabeled or not args.test_labeled:
            parser.error("'train' mode requires --test-unlabeled and --test-labeled")

        run_final_training(
            args.train_unlabeled,
            args.train_labeled,
            args.test_unlabeled,
            args.test_labeled,
            config=config,
        )
