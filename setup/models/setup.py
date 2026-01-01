from pathlib import Path
from typing import List, Optional

from auto_ml.implementations.classifiers.cnn import CNNModel
from auto_ml.implementations.classifiers.vit import ViTModel as ViTClassificationModel
from auto_ml.implementations.nodes import ModelNode
from auto_ml.implementations.segmentators.quadtree import QuadtreeSegmentationModel
from auto_ml.implementations.segmentators.swin import SwinModel
from auto_ml.implementations.segmentators.vit import ViTModel as ViTSegmentationModel
from auto_ml.interfaces import ClassificationModelInterface


def create_vit_model_node() -> ModelNode:
    """Create a ModelNode with a ViT segmentation model."""
    vit_model = ViTSegmentationModel(epochs=40, batch_size=2, device="auto")
    return ModelNode(model=vit_model, name="ViT_Model_Node")


def create_swin_model_node() -> ModelNode:
    """Create a ModelNode with a Swin segmentation model."""
    swin_model = SwinModel(epochs=40, batch_size=2, device="auto")
    return ModelNode(model=swin_model, name="Swin_Model_Node")


def _create_quadtree_model_node(
    classifier: ClassificationModelInterface,
    classifier_dataset_dir: Path,
    optimize_metric: Optional[str] = None,
) -> ModelNode:
    """Create a QuadTree segmentation model node given a classifier."""
    quadtree_model = QuadtreeSegmentationModel(
        classifier,
        classifier_dataset_dir,
        threshold=0.5,
        optimize_metric=optimize_metric,
    )
    return ModelNode(
        model=quadtree_model,
        name=f"Quadtree-{classifier.__class__.__name__}_Model_Node",
    )


def create_quadtree_model_nodes() -> List[ModelNode]:
    """Create a list of QuadTree segmentation model nodes with different classifiers."""
    classifiers = [
        # here we initialize the classifiers
        CNNModel(train_epochs=50),
        ViTClassificationModel(),
    ]

    # change this parameters at taste
    optimize_metric = "f1_score"
    classifier_dataset_dir = Path("path/to/dataset")

    return [
        _create_quadtree_model_node(classifier, classifier_dataset_dir, optimize_metric)
        for classifier in classifiers
    ]


def get_model_nodes() -> List[ModelNode]:
    """Return a list of model nodes for use in Auto-ML."""
    return [
        create_vit_model_node(),
        create_swin_model_node(),
        *create_quadtree_model_nodes(),
    ]
