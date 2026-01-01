"""Precision evaluators for multi-class segmentation."""

from typing import List, Tuple

import numpy as np

from auto_ml.interfaces import EvaluatorInterface, MaskArray, MaskPair


class PrecisionClass0Evaluator(EvaluatorInterface):
    """
    Precision for class 0.

    Measure the accuracy of positive predictions for class 0.
    Precision answers: "Of all pixels predicted as class 0, how many were correct?"

    High precision means few false positives (low false detection rate).
    Important for SEM analysis to understand false detection of material phases.

    Formula: Precision = TP / (TP + FP)
    Range: [0.0, 1.0] where 1.0 is perfect precision

    Edge cases:
        - Return 0.0 if no pixels are predicted as class 0

    """

    def evaluate(self, mask_pairs: List[List[MaskPair]]) -> float:
        """
        Evaluate precision for class 0 across all mask pairs.

        Args:
            mask_pairs: List of mask pair lists from ModelNode,
                       where each pair is (predicted_mask, real_mask).

        Returns:
            Precision for class 0 as float in [0.0, 1.0].

        """
        tp, fp, fn, tn = aggregate_confusion_matrices(mask_pairs, class_id=0)

        denominator = tp + fp
        if denominator == 0:
            return 0.0

        precision = tp / denominator
        return float(precision)


class PrecisionClass1Evaluator(EvaluatorInterface):
    """
    Precision for class 1.

    Measure the accuracy of positive predictions for class 1.
    Precision answers: "Of all pixels predicted as class 1, how many were correct?"

    High precision means few false positives (low false detection rate).
    Important for SEM analysis to understand false detection of material phases.

    Formula: Precision = TP / (TP + FP)
    Range: [0.0, 1.0] where 1.0 is perfect precision

    Edge cases:
        - Return 0.0 if no pixels are predicted as class 1

    """

    def evaluate(self, mask_pairs: List[List[MaskPair]]) -> float:
        """
        Evaluate precision for class 1 across all mask pairs.

        Args:
            mask_pairs: List of mask pair lists from ModelNode,
                       where each pair is (predicted_mask, real_mask).

        Returns:
            Precision for class 1 as float in [0.0, 1.0].

        """
        tp, fp, fn, tn = aggregate_confusion_matrices(mask_pairs, class_id=1)

        denominator = tp + fp
        if denominator == 0:
            return 0.0

        precision = tp / denominator
        return float(precision)


class PrecisionClass2Evaluator(EvaluatorInterface):
    """
    Precision for class 2.

    Measure the accuracy of positive predictions for class 2.
    Precision answers: "Of all pixels predicted as class 2, how many were correct?"

    High precision means few false positives (low false detection rate).
    Important for SEM analysis to understand false detection of material phases.

    Formula: Precision = TP / (TP + FP)
    Range: [0.0, 1.0] where 1.0 is perfect precision

    Edge cases:
        - Return 0.0 if no pixels are predicted as class 2

    """

    def evaluate(self, mask_pairs: List[List[MaskPair]]) -> float:
        """
        Evaluate precision for class 2 across all mask pairs.

        Args:
            mask_pairs: List of mask pair lists from ModelNode,
                       where each pair is (predicted_mask, real_mask).

        Returns:
            Precision for class 2 as float in [0.0, 1.0].

        """
        tp, fp, fn, tn = aggregate_confusion_matrices(mask_pairs, class_id=2)

        denominator = tp + fp
        if denominator == 0:
            return 0.0

        precision = tp / denominator
        return float(precision)


class PrecisionMacroAverageEvaluator(EvaluatorInterface):
    """
    Precision Macro Average across all classes (unweighted mean).

    Compute precision for each class independently and return the unweighted mean.
    Treat all classes equally regardless of their frequency in the dataset.

    This metric is useful when all classes are equally important, such as
    in SEM segmentation where all material phases matter.

    Formula: Precision_macro = (Precision_0 + Precision_1 + Precision_2) / 3
    Range: [0.0, 1.0] where 1.0 is perfect precision for all classes

    Edge cases:
        - Classes with no predictions contribute 0.0 to the average
        - Always divide by 3 (all classes) for consistency

    """

    def evaluate(self, mask_pairs: List[List[MaskPair]]) -> float:
        """
        Evaluate macro-averaged precision across all classes.

        Args:
            mask_pairs: List of mask pair lists from ModelNode,
                       where each pair is (predicted_mask, real_mask).

        Returns:
            Macro-averaged precision as float in [0.0, 1.0].

        """
        precisions = []

        for class_id in [0, 1, 2]:
            tp, fp, fn, tn = aggregate_confusion_matrices(mask_pairs, class_id)
            denominator = tp + fp

            if denominator == 0:
                precision = 0.0
            else:
                precision = tp / denominator

            precisions.append(precision)

        macro_precision = sum(precisions) / 3.0
        return float(macro_precision)


# --- Helper Functions ---


def compute_confusion_matrix_per_class(
    predicted_mask: MaskArray,
    real_mask: MaskArray,
    class_id: int,
) -> Tuple[int, int, int, int]:
    """
    Compute confusion matrix components for a specific class.

    Args:
        predicted_mask: Predicted segmentation mask (512x512 uint8).
        real_mask: Ground truth segmentation mask (512x512 uint8).
        class_id: Class ID to compute metrics for (0, 1, or 2).

    Returns:
        Tuple of (tp, fp, fn, tn) as integers where:
            - tp: True Positives (predicted class c AND real class c)
            - fp: False Positives (predicted class c BUT real is not c)
            - fn: False Negatives (real class c BUT predicted is not c)
            - tn: True Negatives (predicted not c AND real not c)

    """
    mask_c = real_mask == class_id
    pred_c = predicted_mask == class_id

    tp = int(np.sum(mask_c & pred_c))
    fp = int(np.sum(pred_c & ~mask_c))
    fn = int(np.sum(mask_c & ~pred_c))
    tn = int(np.sum(~mask_c & ~pred_c))

    return tp, fp, fn, tn


def aggregate_confusion_matrices(
    mask_pairs: List[List[MaskPair]],
    class_id: int,
) -> Tuple[int, int, int, int]:
    """
    Aggregate confusion matrix values across all folds and samples.

    Iterate through all folds and all samples within each fold,
    summing up the confusion matrix components for the specified class.

    Args:
        mask_pairs: List of mask pair lists from ModelNode,
                   where each pair is (predicted_mask, real_mask).
        class_id: Class ID to compute metrics for (0, 1, or 2).

    Returns:
        Aggregated tuple of (tp, fp, fn, tn) across all folds.

    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    for fold_pairs in mask_pairs:
        for predicted_mask, real_mask in fold_pairs:
            tp, fp, fn, tn = compute_confusion_matrix_per_class(
                predicted_mask,
                real_mask,
                class_id,
            )
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

    return total_tp, total_fp, total_fn, total_tn
