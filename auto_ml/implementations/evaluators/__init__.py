"""Evaluator implementations subpackage."""

from auto_ml.implementations.evaluators.accuracy import AccuracyEvaluator
from auto_ml.implementations.evaluators.autoencoder import AutoencoderMaskEvaluator

# Dice metrics
from auto_ml.implementations.evaluators.dice import (
    DiceClass0Evaluator,
    DiceClass1Evaluator,
    DiceClass2Evaluator,
    DiceMacroAverageEvaluator,
    DiceWeightedAverageEvaluator,
)

# IoU metrics
from auto_ml.implementations.evaluators.iou import (
    IoUClass0Evaluator,
    IoUClass1Evaluator,
    IoUClass2Evaluator,
    IoUMacroAverageEvaluator,
    IoUWeightedAverageEvaluator,
)

# Precision metrics
from auto_ml.implementations.evaluators.precision import (
    PrecisionClass0Evaluator,
    PrecisionClass1Evaluator,
    PrecisionClass2Evaluator,
    PrecisionMacroAverageEvaluator,
)

# Recall metrics
from auto_ml.implementations.evaluators.recall import (
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
