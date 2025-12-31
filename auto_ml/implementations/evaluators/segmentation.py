"""Segmentation evaluator implementations for multi-class segmentation."""

from typing import List, Tuple

import numpy as np

from auto_ml.interfaces import EvaluatorInterface, MaskArray, MaskPair


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
                predicted_mask, real_mask, class_id
            )
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

    return total_tp, total_fp, total_fn, total_tn


# ============================================================================
# IoU (Intersection over Union) Evaluators
# ============================================================================


class IoUClass0Evaluator(EvaluatorInterface):
    """
    IoU (Intersection over Union) for class 0.

    Measure the overlap between predicted and ground truth regions
    for class 0. Also known as the Jaccard Index.

    Formula: IoU = TP / (TP + FP + FN)
    Range: [0.0, 1.0] where 1.0 is perfect overlap

    Edge cases:
        - Return 0.0 if class 0 never appears in predictions or ground truth

    """

    def evaluate(self, mask_pairs: List[List[MaskPair]]) -> float:
        """
        Evaluate IoU for class 0 across all mask pairs.

        Args:
            mask_pairs: List of mask pair lists from ModelNode,
                       where each pair is (predicted_mask, real_mask).

        Returns:
            IoU score for class 0 as float in [0.0, 1.0].

        """
        tp, fp, fn, tn = aggregate_confusion_matrices(mask_pairs, class_id=0)

        denominator = tp + fp + fn
        if denominator == 0:
            return 0.0

        iou = tp / denominator
        return float(iou)


class IoUClass1Evaluator(EvaluatorInterface):
    """
    IoU (Intersection over Union) for class 1.

    Measure the overlap between predicted and ground truth regions
    for class 1. Also known as the Jaccard Index.

    Formula: IoU = TP / (TP + FP + FN)
    Range: [0.0, 1.0] where 1.0 is perfect overlap

    Edge cases:
        - Return 0.0 if class 1 never appears in predictions or ground truth

    """

    def evaluate(self, mask_pairs: List[List[MaskPair]]) -> float:
        """
        Evaluate IoU for class 1 across all mask pairs.

        Args:
            mask_pairs: List of mask pair lists from ModelNode,
                       where each pair is (predicted_mask, real_mask).

        Returns:
            IoU score for class 1 as float in [0.0, 1.0].

        """
        tp, fp, fn, tn = aggregate_confusion_matrices(mask_pairs, class_id=1)

        denominator = tp + fp + fn
        if denominator == 0:
            return 0.0

        iou = tp / denominator
        return float(iou)


class IoUClass2Evaluator(EvaluatorInterface):
    """
    IoU (Intersection over Union) for class 2.

    Measure the overlap between predicted and ground truth regions
    for class 2. Also known as the Jaccard Index.

    Formula: IoU = TP / (TP + FP + FN)
    Range: [0.0, 1.0] where 1.0 is perfect overlap

    Edge cases:
        - Return 0.0 if class 2 never appears in predictions or ground truth

    """

    def evaluate(self, mask_pairs: List[List[MaskPair]]) -> float:
        """
        Evaluate IoU for class 2 across all mask pairs.

        Args:
            mask_pairs: List of mask pair lists from ModelNode,
                       where each pair is (predicted_mask, real_mask).

        Returns:
            IoU score for class 2 as float in [0.0, 1.0].

        """
        tp, fp, fn, tn = aggregate_confusion_matrices(mask_pairs, class_id=2)

        denominator = tp + fp + fn
        if denominator == 0:
            return 0.0

        iou = tp / denominator
        return float(iou)


class IoUMacroAverageEvaluator(EvaluatorInterface):
    """
    IoU Macro Average across all classes (unweighted mean).

    Compute IoU for each class independently and return the unweighted mean.
    Treat all classes equally regardless of their frequency in the dataset.

    This metric is useful when all classes are equally important,
    such as in SEM segmentation where minority phases should not be ignored.

    Formula: IoU_macro = (IoU_0 + IoU_1 + IoU_2) / 3
    Range: [0.0, 1.0] where 1.0 is perfect overlap for all classes

    Edge cases:
        - Classes that never appear contribute 0.0 to the average
        - Always divide by 3 (all classes) for consistency

    """

    def evaluate(self, mask_pairs: List[List[MaskPair]]) -> float:
        """
        Evaluate macro-averaged IoU across all classes.

        Args:
            mask_pairs: List of mask pair lists from ModelNode,
                       where each pair is (predicted_mask, real_mask).

        Returns:
            Macro-averaged IoU as float in [0.0, 1.0].

        """
        ious = []

        for class_id in [0, 1, 2]:
            tp, fp, fn, tn = aggregate_confusion_matrices(mask_pairs, class_id)
            denominator = tp + fp + fn

            if denominator == 0:
                iou = 0.0
            else:
                iou = tp / denominator

            ious.append(iou)

        macro_iou = sum(ious) / 3.0
        return float(macro_iou)


class IoUWeightedAverageEvaluator(EvaluatorInterface):
    """
    IoU Weighted Average by class frequency.

    Weight each class IoU by the number of ground truth pixels for that class.
    Better reflect overall accuracy when classes are imbalanced.

    This metric is useful for understanding overall performance in datasets
    with class imbalance, as it weights classes by their actual prevalence.

    Formula: IoU_weighted = sum(weight_c * IoU_c) / sum(weight_c)
             where weight_c = TP_c + FN_c (ground truth pixels for class c)
    Range: [0.0, 1.0] where 1.0 is perfect overlap

    Edge cases:
        - Return 0.0 if all classes have zero ground truth pixels
        - Classes with more ground truth pixels contribute more to the average

    """

    def evaluate(self, mask_pairs: List[List[MaskPair]]) -> float:
        """
        Evaluate weighted-averaged IoU across all classes.

        Args:
            mask_pairs: List of mask pair lists from ModelNode,
                       where each pair is (predicted_mask, real_mask).

        Returns:
            Weighted-averaged IoU as float in [0.0, 1.0].

        """
        weighted_sum = 0.0
        total_weight = 0

        for class_id in [0, 1, 2]:
            tp, fp, fn, tn = aggregate_confusion_matrices(mask_pairs, class_id)

            # Weight is total ground truth pixels for this class
            weight = tp + fn

            if weight > 0:
                denominator = tp + fp + fn
                if denominator > 0:
                    iou = tp / denominator
                    weighted_sum += iou * weight
                    total_weight += weight

        if total_weight == 0:
            return 0.0

        weighted_iou = weighted_sum / total_weight
        return float(weighted_iou)


# ============================================================================
# Dice Coefficient (F1 Score) Evaluators
# ============================================================================


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

            # Weight is total ground truth pixels for this class
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


# ============================================================================
# Precision Evaluators
# ============================================================================


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


# ============================================================================
# Recall Evaluators
# ============================================================================


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
