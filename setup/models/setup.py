from pathlib import Path
from typing import List, Optional

from auto_ml.implementations.classifiers.cnn import CNNModel
from auto_ml.implementations.classifiers.vit import ViTModel as ViTClassificationModel
from auto_ml.implementations.nodes import ModelNode
from auto_ml.implementations.segmentators.quadtree import QuadtreeSegmentationModel
from auto_ml.implementations.segmentators.sliding_window import (
    SlidingWindowSegmentationModel,
)
from auto_ml.implementations.segmentators.swin import SwinModel
from auto_ml.implementations.segmentators.vit import ViTModel as ViTSegmentationModel
from auto_ml.interfaces import ClassificationModelInterface


def create_vit_model_node() -> ModelNode:
    """Create a ModelNode with a ViT segmentation model."""
    vit_model = ViTSegmentationModel(
        epochs=40,
        batch_size=2,
        dim=512,
        depth=12,
        heads=16,
        mlp_dim=2048,
        device="auto",
    )
    return ModelNode(model=vit_model, name="ViT_Big_Model_Node")


def create_swin_model_node() -> ModelNode:
    """Create a ModelNode with a Swin segmentation model."""
    swin_model = SwinModel(
        epochs=40,
        batch_size=2,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[8, 16, 32, 64],
        device="auto",
    )
    return ModelNode(model=swin_model, name="Swin_Big_Model_Node")


def _create_quadtree_model_node(
    classifier: ClassificationModelInterface,
    classifier_dataset_dir: Path,
    optimize_metric: Optional[str] = None,
) -> ModelNode:
    """Create a QuadTree segmentation model node given a classifier."""
    quadtree_model = QuadtreeSegmentationModel(
        classifier,
        classifier_dataset_dir,
        min_region_size=8,
        threshold=0.5,
        optimize_metric=optimize_metric,
    )
    return ModelNode(
        model=quadtree_model,
        name=f"Quadtree-{classifier.__class__.__name__}_Model_Node",
    )


def create_quadtree_model_nodes(classifier_dataset_dir: Path) -> List[ModelNode]:
    """Create a list of QuadTree segmentation model nodes with different classifiers."""
    classifiers = [
        CNNModel(train_epochs=50),
        ViTClassificationModel(
            train_epochs=40,
            dim=256,
            depth=6,
            heads=8,
            mlp_dim=512,
            device="auto",
        ),
    ]

    # change this parameters at taste
    optimize_metric = "f1_score"

    return [
        _create_quadtree_model_node(classifier, classifier_dataset_dir, optimize_metric)
        for classifier in classifiers
    ]


def _create_sliding_window_model_node(
    classifier: ClassificationModelInterface,
    classifier_dataset_dir: Path,
    window_size: int,
    stride: int,
) -> ModelNode:
    """Create a SlidingWindow model node with specified parameters."""
    model = SlidingWindowSegmentationModel(
        classifier=classifier,
        classifier_dataset_dir=classifier_dataset_dir,
        window_size=window_size,
        stride=stride,
        aggregation_method="majority_vote",
    )

    classifier_name = classifier.__class__.__name__.replace("Model", "")
    node_name = f"SlidingWindow-{classifier_name}_W{window_size}_S{stride}"

    return ModelNode(model=model, name=node_name)


def create_sliding_window_model_nodes(classifier_dataset_dir: Path) -> List[ModelNode]:
    """Create SlidingWindow model nodes with different configurations."""
    # Two classifiers
    classifiers = [
        CNNModel(train_epochs=50, num_blocks=5),  # Supports min 32x32
        ViTClassificationModel(
            train_epochs=40,
            dim=256,
            depth=6,
            heads=8,
            mlp_dim=512,
            device="auto",
        ),
    ]

    # Four window/stride configurations
    configs = [
        {"window_size": 64, "stride": 32},  # Fine-grained
        {"window_size": 128, "stride": 64},  # Balanced
        {"window_size": 128, "stride": 128},  # Fast (non-overlapping)
        {"window_size": 256, "stride": 128},  # Coarse
    ]

    nodes = []
    for classifier in classifiers:
        for config in configs:
            nodes.append(
                _create_sliding_window_model_node(
                    classifier=classifier,
                    classifier_dataset_dir=classifier_dataset_dir,
                    window_size=config["window_size"],
                    stride=config["stride"],
                ),
            )

    return nodes


def get_model_nodes(classifier_dataset_dir: Path) -> List[ModelNode]:
    """Return a list of model nodes for use in Auto-ML."""
    return [
        create_vit_model_node(),
        create_swin_model_node(),
        *create_quadtree_model_nodes(classifier_dataset_dir),
        *create_sliding_window_model_nodes(classifier_dataset_dir),
    ]
