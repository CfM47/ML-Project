"""
Swin Segmentation Training and Validation Module.

Provide three entry points:
1. run_percentage_validation: K-fold cross-validation with varying training percentages
2. run_final_training: Train on 100% data and evaluate on test set
3. run_evaluation_only: Load pretrained model and evaluate on test set
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from matplotlib.figure import Figure

from auto_ml.implementations import SwinModel, load_dataset_from_directories
from auto_ml.interfaces import MaskPair, SegmentationDatasetInterface
from model.swin.config import SwinTrainingConfig
from model.swin.data import create_augmentator, create_kfold_splits, subsample_dataset
from model.swin.evaluation import (
    create_evaluator,
    evaluate_leave_two_out,
    evaluate_model,
)
from model.swin.metrics import (
    FoldMetrics,
    PercentageMetrics,
    SubsetMetrics,
    TrainingHistory,
    TrainingResult,
)
from model.swin.visualization import (
    plot_metric_histograms,
    plot_progression_grid,
    plot_results,
    plot_training_loss_curves,
    visualize_predictions,
)

# ==============================================================================
# Entry Points
# ==============================================================================


def run_percentage_validation(
    train_unlabeled_dir: str | Path,
    train_labeled_dir: str | Path,
    test_unlabeled_dir: str | Path | None = None,
    test_labeled_dir: str | Path | None = None,
    config: SwinTrainingConfig | None = None,
) -> Tuple[List[PercentageMetrics], Figure, Figure | None]:
    """
    Run learning curve analysis with k-fold cross-validation.

    For each training percentage, run k-fold CV and aggregate metrics.
    Save plots to output directory. Optionally generate progression visualization
    if test set is provided.

    Args:
        train_unlabeled_dir: Directory with unlabeled training images.
        train_labeled_dir: Directory with labeled training masks.
        test_unlabeled_dir: Optional directory with unlabeled test images.
        test_labeled_dir: Optional directory with labeled test masks.
        config: Training configuration. Uses defaults if None.

    Returns:
        Tuple of (list of PercentageMetrics, learning curves Figure,
        progression Figure or None if no test set).

    """
    if config is None:
        config = SwinTrainingConfig()

    # Setup output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load training dataset
    print("Loading training dataset...")
    train_dataset = load_dataset_from_directories(
        Path(train_unlabeled_dir),
        Path(train_labeled_dir),
    )
    print(f"Loaded {len(train_dataset)} training samples")

    # Load test dataset if provided
    test_dataset: SegmentationDatasetInterface | None = None
    sample_indices: List[int] = []
    if test_unlabeled_dir and test_labeled_dir:
        print("Loading test dataset...")
        test_dataset = load_dataset_from_directories(
            Path(test_unlabeled_dir),
            Path(test_labeled_dir),
        )
        print(f"Loaded {len(test_dataset)} test samples")

        # Select random sample indices (fixed across percentages)
        rng = np.random.default_rng(config.seed)
        num_samples = min(config.num_progression_samples, len(test_dataset))
        sample_indices = rng.choice(
            len(test_dataset),
            size=num_samples,
            replace=False,
        ).tolist()
        print(f"Selected {num_samples} test samples for progression visualization")

    # Run validation for each percentage
    all_metrics, best_models = _run_validation_loop(train_dataset, config)

    # Generate and save learning curves
    plot_path = config.output_dir / "learning_curves.png"
    learning_curves_fig = plot_results(all_metrics, output_path=plot_path)

    # Generate progression visualization if test set is provided
    progression_fig: Figure | None = None
    if test_dataset and sample_indices:
        predictions_by_percentage = _collect_progression_predictions(
            best_models,
            test_dataset,
            sample_indices,
        )
        progression_path = config.output_dir / "progression.png"
        progression_fig = plot_progression_grid(
            test_dataset,
            predictions_by_percentage,
            sample_indices,
            output_path=progression_path,
        )

    return all_metrics, learning_curves_fig, progression_fig


def run_final_training(
    train_unlabeled_dir: str | Path,
    train_labeled_dir: str | Path,
    test_unlabeled_dir: str | Path,
    test_labeled_dir: str | Path,
    config: SwinTrainingConfig | None = None,
) -> TrainingResult:
    """
    Train final model on 100% data and evaluate on test set.

    Save model weights, prediction visualizations, metric histograms, and
    results JSON to output directory.

    Args:
        train_unlabeled_dir: Directory with unlabeled training images.
        train_labeled_dir: Directory with labeled training masks.
        test_unlabeled_dir: Directory with unlabeled test images.
        test_labeled_dir: Directory with labeled test masks.
        config: Training configuration. Uses defaults if None.

    Returns:
        TrainingResult containing model, metrics, mask_pairs, and figures.

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
    model, training_history = _train_final_model(train_dataset, config)

    # Save model
    model_path = config.output_dir / "model.pt"
    _save_model(model, model_path)

    # Evaluate model and save results
    return _evaluate_model(
        model,
        train_dataset,
        test_dataset,
        config,
        training_history,
    )


def run_evaluation_only(
    train_unlabeled_dir: str | Path,
    train_labeled_dir: str | Path,
    test_unlabeled_dir: str | Path,
    test_labeled_dir: str | Path,
    pretrained_model_path: str | Path,
    config: SwinTrainingConfig | None = None,
) -> TrainingResult:
    """
    Load a pretrained model and evaluate on test set.

    Entry point for evaluating without training. Perform leave-two-out analysis
    and save prediction visualizations, metric histograms, and results JSON.

    Args:
        train_unlabeled_dir: Directory with unlabeled training images.
        train_labeled_dir: Directory with labeled training masks.
        test_unlabeled_dir: Directory with unlabeled test images.
        test_labeled_dir: Directory with labeled test masks.
        pretrained_model_path: Path to pretrained model weights (.pt file).
        config: Training configuration. Uses defaults if None.

    Returns:
        TrainingResult containing model, metrics, mask_pairs, and figures.

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

    # Load pretrained model
    model = _load_model(Path(pretrained_model_path), config)

    # Evaluate model (no training history since we loaded a pretrained model)
    return _evaluate_model(
        model,
        train_dataset,
        test_dataset,
        config,
        training_history=None,
    )


# ==============================================================================
# Validation Loop
# ==============================================================================


def _run_validation_loop(
    train_dataset: SegmentationDatasetInterface,
    config: SwinTrainingConfig,
) -> Tuple[List[PercentageMetrics], Dict[int, SwinModel]]:
    """Run learning curve analysis for all training percentages."""
    all_metrics: List[PercentageMetrics] = []
    best_models_by_percentage: Dict[int, SwinModel] = {}

    for percentage in config.train_percentages:
        print(f"\n{'=' * 60}")
        print(f"Training with {percentage}% of data")
        print(f"{'=' * 60}")

        pct_metrics, best_model = _run_percentage_experiment(
            train_dataset,
            percentage,
            config,
        )
        all_metrics.append(pct_metrics)
        best_models_by_percentage[percentage] = best_model

        print(f"\n  {percentage}% Summary:")
        for metric_name in pct_metrics.get_all_metric_names():
            mean_val = pct_metrics.get_metric_mean(metric_name)
            std_val = pct_metrics.get_metric_std(metric_name)
            print(f"    {metric_name}: {mean_val:.4f} ± {std_val:.4f}")

    return all_metrics, best_models_by_percentage


def _run_percentage_experiment(
    dataset: SegmentationDatasetInterface,
    percentage: int,
    config: SwinTrainingConfig,
) -> Tuple[PercentageMetrics, SwinModel]:
    """Run k-fold cross-validation for a single training percentage."""
    # Subsample dataset
    subsampled = subsample_dataset(dataset, percentage, config.seed)
    print(f"  Subsampled to {len(subsampled)} samples ({percentage}%)")

    # Create k-fold splits
    splits = create_kfold_splits(subsampled, config.n_folds, config.seed)

    pct_metrics = PercentageMetrics(percentage=percentage)

    # Track best fold model by dice score
    best_model: SwinModel | None = None
    best_dice: float = -1.0

    for fold, (train_split, val_split) in enumerate(splits):
        print(f"\n  Fold {fold + 1}/{config.n_folds}")
        fold_metrics, model = _train_fold(train_split, val_split, fold, config)
        pct_metrics.fold_metrics.append(fold_metrics)

        # Track best model
        if fold_metrics.dice_macro > best_dice:
            best_dice = fold_metrics.dice_macro
            best_model = model

        print(f"    Fold {fold + 1} Metrics:")
        for metric_name, value in sorted(fold_metrics.metrics.items()):
            print(f"      {metric_name}: {value:.4f}")

    # best_model is guaranteed to be set since we have at least one fold
    assert best_model is not None
    return pct_metrics, best_model


def _train_fold(
    train_dataset: SegmentationDatasetInterface,
    val_dataset: SegmentationDatasetInterface,
    fold: int,
    config: SwinTrainingConfig,
) -> Tuple[FoldMetrics, SwinModel]:
    """
    Train and evaluate a single fold.

    1. Apply augmentation to training data (not validation)
    2. Create and train SwinModel
    3. Evaluate on validation set
    4. Return metrics and trained model

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

    # Evaluate on validation set (autoencoder trained on training set)
    evaluator = create_evaluator(train_dataset)
    metrics, _ = evaluate_model(model, val_dataset, evaluator)

    fold_metrics = FoldMetrics(
        fold=fold,
        train_history=train_result.history,
        metrics=metrics,
    )

    return fold_metrics, model


# ==============================================================================
# Final Training
# ==============================================================================


def _train_final_model(
    train_dataset: SegmentationDatasetInterface,
    config: SwinTrainingConfig,
) -> Tuple[SwinModel, TrainingHistory | None]:
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
    train_result = model.train(aug_train, validation_dataset=val_split)

    # Create validated training history
    training_history = TrainingHistory.from_history_dicts(train_result.history)

    return model, training_history


def _evaluate_on_test(
    model: SwinModel,
    test_dataset: SegmentationDatasetInterface,
    train_dataset: SegmentationDatasetInterface,
) -> Tuple[SubsetMetrics, List[MaskPair]]:
    """
    Evaluate trained model on test dataset with leave-two-out analysis.

    Args:
        model: Trained SwinModel.
        test_dataset: Test dataset to evaluate on.
        train_dataset: Training dataset for Mask_Cohesion autoencoder reference.

    Returns:
        Tuple of (SubsetMetrics, mask_pairs).

    """
    print("\n" + "=" * 60)
    print("Evaluating on test set")
    print("=" * 60)

    # Create evaluator (trained on training set for Mask_Cohesion)
    evaluator = create_evaluator(train_dataset)

    # Evaluate full test set
    full_metrics, mask_pairs = evaluate_model(model, test_dataset, evaluator)

    print("\nFull Test Set Metrics:")
    for metric_name, value in sorted(full_metrics.items()):
        print(f"  {metric_name}: {value:.4f}")

    # Perform leave-two-out evaluation
    subset_means, subset_stds, subset_distributions = evaluate_leave_two_out(
        mask_pairs,
        evaluator,
    )

    # Build SubsetMetrics
    n = len(mask_pairs)
    subset_metrics = SubsetMetrics(
        subset_size=n - 2,
        num_subsets=n * (n - 1) // 2,
        full_metrics=full_metrics,
        subset_means=subset_means,
        subset_stds=subset_stds,
        subset_distributions=subset_distributions,
    )

    print("\nLeave-Two-Out Summary:")
    for metric_name in sorted(subset_means.keys()):
        mean_val = subset_means[metric_name]
        std_val = subset_stds[metric_name]
        print(f"  {metric_name}: {mean_val:.4f} ± {std_val:.4f}")

    return subset_metrics, mask_pairs


def _evaluate_model(
    model: SwinModel,
    train_dataset: SegmentationDatasetInterface,
    test_dataset: SegmentationDatasetInterface,
    config: SwinTrainingConfig,
    training_history: TrainingHistory | None = None,
) -> TrainingResult:
    """
    Evaluate a model on test set with leave-two-out analysis.

    Save predictions, histograms, and results JSON to output directory.

    Args:
        model: Trained or loaded SwinModel.
        train_dataset: Training dataset for Mask_Cohesion autoencoder reference.
        test_dataset: Test dataset to evaluate on.
        config: Training configuration with output directory.
        training_history: Optional training history for loss curves plot.

    Returns:
        TrainingResult containing model, metrics, mask_pairs, and figures.

    """
    # Evaluate on test set with leave-two-out analysis
    test_metrics, test_mask_pairs = _evaluate_on_test(
        model,
        test_dataset,
        train_dataset,
    )

    # Visualize predictions
    viz_path = config.output_dir / "test_predictions.png"
    predictions_fig = visualize_predictions(
        test_dataset,
        test_mask_pairs,
        num_samples=config.num_test_visualizations,
        output_path=viz_path,
    )

    # Plot training loss curves if history is valid
    loss_curves_fig: Figure | None = None
    if training_history is not None:
        loss_curves_path = config.output_dir / "training_loss_curves.png"
        loss_curves_fig = plot_training_loss_curves(
            training_history,
            output_path=loss_curves_path,
        )

    # Plot metric histograms
    histograms = plot_metric_histograms(
        test_metrics.subset_distributions,
        test_metrics.full_metrics,
        output_dir=config.output_dir,
    )

    # Build result object
    result = TrainingResult(
        model=model,
        test_metrics=test_metrics,
        mask_pairs=test_mask_pairs,
        predictions_figure=predictions_fig,
        loss_curves_figure=loss_curves_fig,
        histograms=histograms,
    )

    # Save results JSON
    _save_results_json(result, config.output_dir / "results.json")

    return result


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


def _save_results_json(result: TrainingResult, path: Path) -> None:
    """Save training results to JSON file."""
    results_dict = result.to_dict()
    with open(path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"Saved results to {path}")


def _load_model(path: Path, config: SwinTrainingConfig) -> SwinModel:
    """Load model weights from disk."""
    model = _create_swin_model(config)
    model.model.load_state_dict(torch.load(path, weights_only=True))
    print(f"Loaded model from {path}")
    return model


def _collect_progression_predictions(
    best_models: Dict[int, SwinModel],
    test_dataset: SegmentationDatasetInterface,
    sample_indices: List[int],
) -> Dict[int, List[np.ndarray]]:
    """
    Collect predictions from best models at each percentage for selected samples.

    Args:
        best_models: Dictionary mapping percentage to best fold's model.
        test_dataset: Test dataset to predict on.
        sample_indices: Indices of samples to predict.

    Returns:
        Dictionary mapping percentage to list of predicted masks.

    """
    predictions_by_percentage: Dict[int, List[np.ndarray]] = {}

    for percentage, model in sorted(best_models.items()):
        print(f"  Collecting predictions for {percentage}%...")
        predictions = _predict_samples(model, test_dataset, sample_indices)
        predictions_by_percentage[percentage] = predictions

    return predictions_by_percentage


def _predict_samples(
    model: SwinModel,
    dataset: SegmentationDatasetInterface,
    indices: List[int],
) -> List[np.ndarray]:
    """
    Predict masks for specific sample indices.

    Args:
        model: Trained SwinModel.
        dataset: Dataset containing samples.
        indices: Indices of samples to predict.

    Returns:
        List of predicted masks as numpy arrays.

    """
    # Create subset dataset with only the selected samples
    subset_samples = [dataset.samples[idx] for idx in indices]
    subset_dataset = SegmentationDatasetInterface(samples=subset_samples)

    # Predict only on the subset
    mask_pairs = model.evaluate(subset_dataset)

    # Extract predicted masks (first element of each pair)
    predictions = [pair[0] for pair in mask_pairs]

    return predictions


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
