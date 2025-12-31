"""Evaluator implementations subpackage."""

from auto_ml.implementations.evaluators.accuracy import AccuracyEvaluator
from auto_ml.implementations.evaluators.autoencoder import AutoencoderMaskEvaluator
from auto_ml.implementations.evaluators.segmentation import (
    DiceClass0Evaluator,
    DiceClass1Evaluator,
    DiceClass2Evaluator,
    DiceMacroAverageEvaluator,
    DiceWeightedAverageEvaluator,
    IoUClass0Evaluator,
    IoUClass1Evaluator,
    IoUClass2Evaluator,
    IoUMacroAverageEvaluator,
    IoUWeightedAverageEvaluator,
    PrecisionClass0Evaluator,
    PrecisionClass1Evaluator,
    PrecisionClass2Evaluator,
    PrecisionMacroAverageEvaluator,
    RecallClass0Evaluator,
    RecallClass1Evaluator,
    RecallClass2Evaluator,
    RecallMacroAverageEvaluator,
)

__all__ = [
    "AccuracyEvaluator",
    "AutoencoderMaskEvaluator",
    # Dice Metrics
    "DiceClass0Evaluator",
    "DiceClass1Evaluator",
    "DiceClass2Evaluator",
    "DiceMacroAverageEvaluator",
    "DiceWeightedAverageEvaluator",
    # IoU Metrics
    "IoUClass0Evaluator",
    "IoUClass1Evaluator",
    "IoUClass2Evaluator",
    "IoUMacroAverageEvaluator",
    "IoUWeightedAverageEvaluator",
    # Precision Metrics
    "PrecisionClass0Evaluator",
    "PrecisionClass1Evaluator",
    "PrecisionClass2Evaluator",
    "PrecisionMacroAverageEvaluator",
    # Recall Metrics
    "RecallClass0Evaluator",
    "RecallClass1Evaluator",
    "RecallClass2Evaluator",
    "RecallMacroAverageEvaluator",
]
