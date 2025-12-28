"""Concrete implementations of the AutoML interfaces."""

# Augmentators
from auto_ml.implementations.augmentators import IdentityAugmentator

# Datasets
from auto_ml.implementations.datasets import load_dataset_from_directories

# Evaluators
from auto_ml.implementations.evaluators.accuracy import AccuracyEvaluator
from auto_ml.implementations.evaluators.autoencoder import AutoencoderMaskEvaluator

# Models
from auto_ml.implementations.models.base import InMemoryPyTorchDataset
from auto_ml.implementations.models.quadtree_model import QuadtreeSegmentationModel
from auto_ml.implementations.models.swin_model import SwinModel
from auto_ml.implementations.models.vit_model import ViTModel

# Nodes
from auto_ml.implementations.nodes import (
    DataAugmentatorNode,
    EvaluatorNode,
    ModelNode,
)

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
