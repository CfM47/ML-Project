"""Concrete implementations of the AutoML interfaces."""

# Augmentators
from auto_ml.implementations.augmentators import IdentityAugmentator

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
