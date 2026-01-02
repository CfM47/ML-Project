"""Baseline augmentation node with no augmentation."""

from auto_ml.implementations.augmentators.composite import MultiplyDatasetAugmentator
from auto_ml.implementations.augmentators.identity import IdentityAugmentator
from auto_ml.implementations.nodes import DataAugmentatorNode


def get_baseline_node(num_copies: int = 1) -> DataAugmentatorNode:
    """Create a baseline node with no augmentation for comparison.

    Args:
        num_copies: Number of copies to create (default: 1).

    """
    return DataAugmentatorNode(
        augmentator=MultiplyDatasetAugmentator(
            augmentators=[IdentityAugmentator()],
            num_copies=num_copies,
            include_original=False,
        ),
        name=f"Baseline_NoAugmentation_x{num_copies}",
    )
