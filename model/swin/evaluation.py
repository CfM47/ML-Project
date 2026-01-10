"""Evaluation utilities for training and validation."""

from typing import Any, Dict, List, Tuple

from auto_ml.implementations.evaluators import (
    AccuracyEvaluator,
    DiceMacroAverageEvaluator,
)
from auto_ml.implementations.nodes import EvaluatorNode
from auto_ml.implementations.segmentators.swin import SwinModel
from auto_ml.interfaces import MaskPair, SegmentationDatasetInterface


def create_evaluator() -> EvaluatorNode:
    """
    Create evaluator with Dice (F1) and Accuracy metrics only.

    Returns:
        Configured EvaluatorNode.

    """
    return EvaluatorNode(
        evaluators={
            "Dice_Macro": DiceMacroAverageEvaluator(),
            "Accuracy": AccuracyEvaluator(),
        },
        name="SwinValidationEvaluator",
    )


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
            metrics[metric_name] = values[0]
        else:
            metrics[metric_name] = float(values)

    return metrics, mask_pairs


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
            metrics[metric_name] = values[0]
        else:
            metrics[metric_name] = float(values)
    return metrics
