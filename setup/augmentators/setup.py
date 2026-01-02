from typing import List

from auto_ml.implementations.nodes import DataAugmentatorNode
from setup.augmentators.identity import get_identity_augmentator_node


def get_augmentator_nodes() -> List[DataAugmentatorNode]:
    """Return augmentator nodes to use in Auto-ML."""
    return [
        get_identity_augmentator_node(),
    ]
