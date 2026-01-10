"""Recall evaluators for multi-class segmentation."""

from typing import List, Tuple

import numpy as np

from auto_ml.interfaces import EvaluatorInterface, MaskArray, MaskPair


class RecallClass0Evaluator(EvaluatorInterface):
    """
    Recall (Sensitivity) for class 0.

    Measure the completeness of detection for class 0.
    Recall answers: "Of all pixels that should be class 0, how many did we find?"

    High recall means few false negatives (low miss rate).
    Critical for SEM to ensure all material phases are properly identified.

    Formula: Recall = TP / (TP + FN)
    Range: [0.0, 1.0] where 1.0 is perfect recall

    Edge cases:
        - Return 0.0 if no pixels in ground truth are class 0

    """

    def evaluate(self, mask_pairs: List[List[MaskPair]]) -> float:
        """
        Evaluate recall for class 0 across all mask pairs.

        Args:
            mask_pairs: List of mask pair lists from ModelNode,
                       where each pair is (predicted_mask, real_mask).

        Returns:
            Recall for class 0 as float in [0.0, 1.0].

        """
        tp, fp, fn, tn = aggregate_confusion_matrices(mask_pairs, class_id=0)

        denominator = tp + fn
        if denominator == 0:
            return 0.0

        recall = tp / denominator
        return float(recall)


class RecallClass1Evaluator(EvaluatorInterface):
    """
    Recall (Sensitivity) for class 1.

    Measure the completeness of detection for class 1.
    Recall answers: "Of all pixels that should be class 1, how many did we find?"

    High recall means few false negatives (low miss rate).
    Critical for SEM to ensure all material phases are properly identified.

    Formula: Recall = TP / (TP + FN)
    Range: [0.0, 1.0] where 1.0 is perfect recall

    Edge cases:
        - Return 0.0 if no pixels in ground truth are class 1

    """

    def evaluate(self, mask_pairs: List[List[MaskPair]]) -> float:
        """
        Evaluate recall for class 1 across all mask pairs.

        Args:
            mask_pairs: List of mask pair lists from ModelNode,
                       where each pair is (predicted_mask, real_mask).

        Returns:
            Recall for class 1 as float in [0.0, 1.0].

        """
        tp, fp, fn, tn = aggregate_confusion_matrices(mask_pairs, class_id=1)

        denominator = tp + fn
        if denominator == 0:
            return 0.0

        recall = tp / denominator
        return float(recall)


class RecallClass2Evaluator(EvaluatorInterface):
    """
    Recall (Sensitivity) for class 2.

    Measure the completeness of detection for class 2.
    Recall answers: "Of all pixels that should be class 2, how many did we find?"

    High recall means few false negatives (low miss rate).
    Critical for SEM to ensure all material phases are properly identified.

    Formula: Recall = TP / (TP + FN)
    Range: [0.0, 1.0] where 1.0 is perfect recall

    Edge cases:
        - Return 0.0 if no pixels in ground truth are class 2

    """

    def evaluate(self, mask_pairs: List[List[MaskPair]]) -> float:
        """
        Evaluate recall for class 2 across all mask pairs.

        Args:
            mask_pairs: List of mask pair lists from ModelNode,
                       where each pair is (predicted_mask, real_mask).

        Returns:
            Recall for class 2 as float in [0.0, 1.0].

        """
        tp, fp, fn, tn = aggregate_confusion_matrices(mask_pairs, class_id=2)

        denominator = tp + fn
        if denominator == 0:
            return 0.0

        recall = tp / denominator
        return float(recall)


class RecallMacroAverageEvaluator(EvaluatorInterface):
    """
    Recall Macro Average across all classes (unweighted mean).

    Compute recall for each class independently and return the unweighted mean.
    Treat all classes equally regardless of their frequency in the dataset.

    This metric is useful when all classes are equally important, particularly
    for SEM segmentation where all material phases need proper detection.

    Formula: Recall_macro = (Recall_0 + Recall_1 + Recall_2) / 3
    Range: [0.0, 1.0] where 1.0 is perfect recall for all classes

    Edge cases:
        - Classes with no ground truth pixels contribute 0.0 to the average
        - Always divide by 3 (all classes) for consistency

    """

    def evaluate(self, mask_pairs: List[List[MaskPair]]) -> float:
        """
        Evaluate macro-averaged recall across all classes.

        Args:
            mask_pairs: List of mask pair lists from ModelNode,
                       where each pair is (predicted_mask, real_mask).

        Returns:
            Macro-averaged recall as float in [0.0, 1.0].

        """
        recalls = []

        for class_id in [0, 1, 2]:
            tp, fp, fn, tn = aggregate_confusion_matrices(mask_pairs, class_id)
            denominator = tp + fn

            if denominator == 0:
                recall = 0.0
            else:
                recall = tp / denominator

            recalls.append(recall)

        macro_recall = sum(recalls) / 3.0
        return float(macro_recall)


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
