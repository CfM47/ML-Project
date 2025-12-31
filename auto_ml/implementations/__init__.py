"""Concrete implementations of the AutoML interfaces."""

# Augmentators
from auto_ml.implementations.augmentators import (
    AdaptiveHistogramEqualizationAugmentator,
    BrightnessAugmentator,
    ChargingArtifactAugmentator,
    ContrastAugmentator,
    ElasticDeformationAugmentator,
    GammaAugmentator,
    GaussianBlurAugmentator,
    GaussianNoiseAugmentator,
    HorizontalFlipAugmentator,
    IdentityAugmentator,
    MultiplyDatasetAugmentator,
    OneOfAugmentator,
    RandomApplyAugmentator,
    RandomChoiceAugmentator,
    RandomCropAugmentator,
    RotationAugmentator,
    ScaleAugmentator,
    ScanLineNoiseAugmentator,
    SequentialAugmentator,
    TranslationAugmentator,
    VerticalFlipAugmentator,
)

# Datasets
from auto_ml.implementations.datasets import load_dataset_from_directories

# Evaluators
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

# Nodes
from auto_ml.implementations.nodes import (
    DataAugmentatorNode,
    EvaluatorNode,
    ModelNode,
)
from auto_ml.implementations.segmentators import (
    QuadtreeSegmentationModel,
    SwinModel,
    ViTModel,
)

# Models
from auto_ml.implementations.segmentators.base import InMemoryPyTorchDataset

__all__ = [
    # Augmentators
    "IdentityAugmentator",
    "RotationAugmentator",
    "HorizontalFlipAugmentator",
    "VerticalFlipAugmentator",
    "ScaleAugmentator",
    "TranslationAugmentator",
    "RandomCropAugmentator",
    "BrightnessAugmentator",
    "ContrastAugmentator",
    "GaussianNoiseAugmentator",
    "GaussianBlurAugmentator",
    "GammaAugmentator",
    "SequentialAugmentator",
    "RandomChoiceAugmentator",
    "RandomApplyAugmentator",
    "OneOfAugmentator",
    "MultiplyDatasetAugmentator",
    "ElasticDeformationAugmentator",
    "AdaptiveHistogramEqualizationAugmentator",
    "ChargingArtifactAugmentator",
    "ScanLineNoiseAugmentator",
    # Datasets
    "load_dataset_from_directories",
    # Nodes
    "DataAugmentatorNode",
    "ModelNode",
    "EvaluatorNode",
    # Models
    "InMemoryPyTorchDataset",
    "ViTModel",
    "SwinModel",
    "QuadtreeSegmentationModel",
    # Evaluators
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
