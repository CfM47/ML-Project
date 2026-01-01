from typing import List

from auto_ml.implementations.nodes import ModelNode
from auto_ml.implementations.segmentators.swin import SwinModel
from auto_ml.implementations.segmentators.vit import ViTModel


def create_vit_model_node() -> ModelNode:
    """Create a ModelNode with a ViT segmentation model."""
    vit_model = ViTModel(epochs=20, batch_size=2, device="auto")
    return ModelNode(model=vit_model, name="ViT_Model_Node")


def create_swin_model_node() -> ModelNode:
    """Create a ModelNode with a Swin segmentation model."""
    swin_model = SwinModel(epochs=20, batch_size=2, device="auto")
    return ModelNode(model=swin_model, name="Swin_Model_Node")


def get_model_nodes() -> List[ModelNode]:
    """Return a list of model nodes for use in Auto-ML."""
    return [create_vit_model_node(), create_swin_model_node()]
