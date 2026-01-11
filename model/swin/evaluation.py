"""Evaluation utilities for training and validation."""

from typing import Any, Dict, List, Tuple

from auto_ml.implementations.evaluators import (
    AccuracyEvaluator,
    AutoencoderMaskEvaluator,
    DiceClass0Evaluator,
    DiceClass1Evaluator,
    IoUClass0Evaluator,
    IoUClass1Evaluator,
    PrecisionClass0Evaluator,
    PrecisionClass1Evaluator,
    RecallClass0Evaluator,
    RecallClass1Evaluator,
)
from auto_ml.implementations.nodes import EvaluatorNode
from auto_ml.implementations.segmentators.swin import SwinModel
from auto_ml.interfaces import MaskPair, SegmentationDatasetInterface


def create_evaluator(dataset: SegmentationDatasetInterface) -> EvaluatorNode:
    """
    Create evaluator with all available metrics for binary segmentation.

    Args:
        dataset: Dataset used for training the Mask_Cohesion autoencoder evaluator.

    Returns:
        Configured EvaluatorNode with all metrics.

    """
    evaluators = {
        # General
        "Accuracy": AccuracyEvaluator(),
        # Autoencoder (requires training on reference masks)
        "Mask_Cohesion": AutoencoderMaskEvaluator(
            reference_masks=dataset.masks,
            latent_dim=8,
            epochs=40,
            nu=0.5,
            device="auto",
        ),
        # IoU Metrics (binary: Class0 and Class1 only)
        "IoU_Class0": IoUClass0Evaluator(),
        "IoU_Class1": IoUClass1Evaluator(),
        # Dice Metrics (binary: Class0 and Class1 only)
        "Dice_Class0": DiceClass0Evaluator(),
        "Dice_Class1": DiceClass1Evaluator(),
        # Precision Metrics (binary: Class0 and Class1 only)
        "Precision_Class0": PrecisionClass0Evaluator(),
        "Precision_Class1": PrecisionClass1Evaluator(),
        # Recall Metrics (binary: Class0 and Class1 only)
        "Recall_Class0": RecallClass0Evaluator(),
        "Recall_Class1": RecallClass1Evaluator(),
    }
    return EvaluatorNode(evaluators=evaluators, name="SwinValidationEvaluator")


def evaluate_model(
    model: SwinModel,
    dataset: SegmentationDatasetInterface,
    evaluator: EvaluatorNode,
) -> Tuple[Dict[str, float], List[MaskPair]]:
    """
    Evaluate a trained model on a dataset.

    Args:
        model: Trained SwinModel.
        dataset: Dataset to evaluate on.
        evaluator: EvaluatorNode for computing metrics.

    Returns:
        Tuple of (metrics_dict, mask_pairs).

    """
    # Get mask pairs from model evaluation
    mask_pairs = model.evaluate(dataset)

    # Evaluator expects List[List[MaskPair]] (one list per fold)
    # We wrap in a single list since this is a single evaluation
    evaluation_results = evaluator.evaluate([mask_pairs])

    # Extract scalar metrics (evaluator returns lists per fold, we take first)
    metrics: Dict[str, float] = {}
    for metric_name, values in evaluation_results.items():
        if isinstance(values, list) and len(values) > 0:
            metrics[metric_name] = float(values[0])
        else:
            metrics[metric_name] = float(values)  # type: ignore[arg-type]

    # Compute F1 scores from Precision and Recall
    _add_f1_metrics(metrics)

    return metrics, mask_pairs


def _add_f1_metrics(metrics: Dict[str, float]) -> None:
    """
    Compute F1 scores from Precision and Recall and add them to metrics dict.

    Formula: F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Args:
        metrics: Metrics dictionary to update in-place.

    """
    for class_id in [0, 1]:
        precision_key = f"Precision_Class{class_id}"
        recall_key = f"Recall_Class{class_id}"
        f1_key = f"F1_Class{class_id}"

        if precision_key in metrics and recall_key in metrics:
            precision = metrics[precision_key]
            recall = metrics[recall_key]
            denominator = precision + recall
            if denominator > 0:
                metrics[f1_key] = 2 * (precision * recall) / denominator
            else:
                metrics[f1_key] = 0.0


def extract_metrics_from_evaluation(
    evaluation_results: Dict[str, Any],
) -> Dict[str, float]:
    """
    Extract scalar metrics from evaluator results.

    Args:
        evaluation_results: Results from EvaluatorNode.evaluate().

    Returns:
        Dictionary of metric name to scalar value.

    """
    metrics: Dict[str, float] = {}
    for metric_name, values in evaluation_results.items():
        if isinstance(values, list) and len(values) > 0:
            metrics[metric_name] = float(values[0])
        else:
            metrics[metric_name] = float(values)  # type: ignore[arg-type]
    return metrics


def evaluate_leave_two_out(
    mask_pairs: List[MaskPair],
    evaluator: EvaluatorNode,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, List[float]]]:
    """
    Evaluate all leave-two-out subsets for robustness analysis.

    For n samples, evaluate all C(n,2) subsets of size n-2.

    Args:
        mask_pairs: List of (predicted_mask, real_mask) tuples from full test set.
        evaluator: Configured EvaluatorNode for computing metrics.

    Returns:
        Tuple of (subset_means, subset_stds, subset_distributions).

    """
    import itertools

    import numpy as np

    n = len(mask_pairs)
    subset_size = n - 2
    num_subsets = n * (n - 1) // 2  # C(n, 2)

    print(
        f"Leave-two-out evaluation: {num_subsets} subsets of size "
        f"{subset_size} from {n} samples",
    )

    # Generate all combinations of 2 indices to exclude
    exclude_pairs = list(itertools.combinations(range(n), 2))

    # Initialize distributions dict
    distributions: Dict[str, List[float]] = {}

    for exclude_idx_pair in exclude_pairs:
        # Create subset by excluding two samples
        subset = [mask_pairs[i] for i in range(n) if i not in exclude_idx_pair]

        # Evaluate subset
        evaluation_results = evaluator.evaluate([subset])

        # Extract metrics
        subset_metrics = extract_metrics_from_evaluation(evaluation_results)
        _add_f1_metrics(subset_metrics)

        # Add to distributions
        for metric_name, value in subset_metrics.items():
            if metric_name not in distributions:
                distributions[metric_name] = []
            distributions[metric_name].append(value)

    # Compute means and stds
    subset_means: Dict[str, float] = {}
    subset_stds: Dict[str, float] = {}

    for metric_name, values in distributions.items():
        subset_means[metric_name] = float(np.mean(values))
        subset_stds[metric_name] = float(np.std(values))

    return subset_means, subset_stds, distributions
