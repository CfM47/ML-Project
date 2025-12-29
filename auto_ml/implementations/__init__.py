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

# Nodes
from auto_ml.implementations.nodes import (
    DataAugmentatorNode,
    EvaluatorNode,
    ModelNode,
)

# Models
from auto_ml.implementations.segmentators.base import InMemoryPyTorchDataset
from auto_ml.implementations.segmentators.quadtree_model import (
    QuadtreeSegmentationModel,
)
from auto_ml.implementations.segmentators.swin_model import SwinModel
from auto_ml.implementations.segmentators.vit_model import ViTModel

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
]
