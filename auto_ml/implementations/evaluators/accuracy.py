"""Accuracy evaluator implementation."""

from typing import List

from auto_ml.interfaces import EvaluatorInterface, MaskPair


class AccuracyEvaluator(EvaluatorInterface):
    """Accuracy Evaluator: Calculates overall pixel-wise accuracy."""

    def evaluate(self, mask_pairs: List[List[MaskPair]]) -> float:
        """Evaluate overall accuracy across all mask pairs."""
        total_correct = 0
        total_pixels = 0

        for fold_pairs in mask_pairs:
            for predicted_mask, real_mask in fold_pairs:
                correct = (predicted_mask == real_mask).sum()
                total_correct += correct
                total_pixels += real_mask.size

        accuracy = total_correct / total_pixels if total_pixels > 0 else 0.0
        return float(accuracy)
