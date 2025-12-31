"""Dice coefficient evaluators for multi-class segmentation."""

from typing import List, Tuple

import numpy as np

from auto_ml.interfaces import EvaluatorInterface, MaskArray, MaskPair


class DiceClass0Evaluator(EvaluatorInterface):
    """
    Dice Coefficient (F1 Score) for class 0.

    Measure the overlap between predicted and ground truth regions
    for class 0. More sensitive to small regions than IoU.

    The Dice coefficient is the harmonic mean of precision and recall,
    and is equivalent to the F1 score. It emphasizes the intersection
    more heavily than IoU.

    Formula: Dice = 2*TP / (2*TP + FP + FN)
    Range: [0.0, 1.0] where 1.0 is perfect overlap

    Edge cases:
        - Return 0.0 if class 0 never appears in predictions or ground truth

    """

    def evaluate(self, mask_pairs: List[List[MaskPair]]) -> float:
        """
        Evaluate Dice coefficient for class 0 across all mask pairs.

        Args:
            mask_pairs: List of mask pair lists from ModelNode,
                       where each pair is (predicted_mask, real_mask).

        Returns:
            Dice coefficient for class 0 as float in [0.0, 1.0].

        """
        tp, fp, fn, tn = aggregate_confusion_matrices(mask_pairs, class_id=0)

        denominator = 2 * tp + fp + fn
        if denominator == 0:
            return 0.0

        dice = (2 * tp) / denominator
        return float(dice)


class DiceClass1Evaluator(EvaluatorInterface):
    """
    Dice Coefficient (F1 Score) for class 1.

    Measure the overlap between predicted and ground truth regions
    for class 1. More sensitive to small regions than IoU.

    The Dice coefficient is the harmonic mean of precision and recall,
    and is equivalent to the F1 score. It emphasizes the intersection
    more heavily than IoU.

    Formula: Dice = 2*TP / (2*TP + FP + FN)
    Range: [0.0, 1.0] where 1.0 is perfect overlap

    Edge cases:
        - Return 0.0 if class 1 never appears in predictions or ground truth

    """

    def evaluate(self, mask_pairs: List[List[MaskPair]]) -> float:
        """
        Evaluate Dice coefficient for class 1 across all mask pairs.

        Args:
            mask_pairs: List of mask pair lists from ModelNode,
                       where each pair is (predicted_mask, real_mask).

        Returns:
            Dice coefficient for class 1 as float in [0.0, 1.0].

        """
        tp, fp, fn, tn = aggregate_confusion_matrices(mask_pairs, class_id=1)

        denominator = 2 * tp + fp + fn
        if denominator == 0:
            return 0.0

        dice = (2 * tp) / denominator
        return float(dice)


class DiceClass2Evaluator(EvaluatorInterface):
    """
    Dice Coefficient (F1 Score) for class 2.

    Measure the overlap between predicted and ground truth regions
    for class 2. More sensitive to small regions than IoU.

    The Dice coefficient is the harmonic mean of precision and recall,
    and is equivalent to the F1 score. It emphasizes the intersection
    more heavily than IoU.

    Formula: Dice = 2*TP / (2*TP + FP + FN)
    Range: [0.0, 1.0] where 1.0 is perfect overlap

    Edge cases:
        - Return 0.0 if class 2 never appears in predictions or ground truth

    """

    def evaluate(self, mask_pairs: List[List[MaskPair]]) -> float:
        """
        Evaluate Dice coefficient for class 2 across all mask pairs.

        Args:
            mask_pairs: List of mask pair lists from ModelNode,
                       where each pair is (predicted_mask, real_mask).

        Returns:
            Dice coefficient for class 2 as float in [0.0, 1.0].

        """
        tp, fp, fn, tn = aggregate_confusion_matrices(mask_pairs, class_id=2)

        denominator = 2 * tp + fp + fn
        if denominator == 0:
            return 0.0

        dice = (2 * tp) / denominator
        return float(dice)


class DiceMacroAverageEvaluator(EvaluatorInterface):
    """
    Dice Coefficient Macro Average across all classes (unweighted mean).

    Compute Dice for each class independently and return the unweighted mean.
    Treat all classes equally regardless of their frequency in the dataset.

    This metric is useful when all classes are equally important, particularly
    for SEM segmentation where minority material phases should not be ignored.

    Formula: Dice_macro = (Dice_0 + Dice_1 + Dice_2) / 3
    Range: [0.0, 1.0] where 1.0 is perfect overlap for all classes

    Edge cases:
        - Classes that never appear contribute 0.0 to the average
        - Always divide by 3 (all classes) for consistency

    """

    def evaluate(self, mask_pairs: List[List[MaskPair]]) -> float:
        """
        Evaluate macro-averaged Dice across all classes.

        Args:
            mask_pairs: List of mask pair lists from ModelNode,
                       where each pair is (predicted_mask, real_mask).

        Returns:
            Macro-averaged Dice as float in [0.0, 1.0].

        """
        dices = []

        for class_id in [0, 1, 2]:
            tp, fp, fn, tn = aggregate_confusion_matrices(mask_pairs, class_id)
            denominator = 2 * tp + fp + fn

            if denominator == 0:
                dice = 0.0
            else:
                dice = (2 * tp) / denominator

            dices.append(dice)

        macro_dice = sum(dices) / 3.0
        return float(macro_dice)


class DiceWeightedAverageEvaluator(EvaluatorInterface):
    """
    Dice Coefficient Weighted Average by class frequency.

    Weight each class Dice by the number of ground truth pixels for that class.
    Better reflect overall accuracy when classes are imbalanced.

    This metric is useful for understanding overall performance in datasets
    with class imbalance, as it weights classes by their actual prevalence.

    Formula: Dice_weighted = sum(weight_c * Dice_c) / sum(weight_c)
             where weight_c = TP_c + FN_c (ground truth pixels for class c)
    Range: [0.0, 1.0] where 1.0 is perfect overlap

    Edge cases:
        - Return 0.0 if all classes have zero ground truth pixels
        - Classes with more ground truth pixels contribute more to the average

    """

    def evaluate(self, mask_pairs: List[List[MaskPair]]) -> float:
        """
        Evaluate weighted-averaged Dice across all classes.

        Args:
            mask_pairs: List of mask pair lists from ModelNode,
                       where each pair is (predicted_mask, real_mask).

        Returns:
            Weighted-averaged Dice as float in [0.0, 1.0].

        """
        weighted_sum = 0.0
        total_weight = 0

        for class_id in [0, 1, 2]:
            tp, fp, fn, tn = aggregate_confusion_matrices(mask_pairs, class_id)

            # Weight by ground truth pixel count
            weight = tp + fn

            if weight > 0:
                denominator = 2 * tp + fp + fn
                if denominator > 0:
                    dice = (2 * tp) / denominator
                    weighted_sum += dice * weight
                    total_weight += weight

        if total_weight == 0:
            return 0.0

        weighted_dice = weighted_sum / total_weight
        return float(weighted_dice)


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
