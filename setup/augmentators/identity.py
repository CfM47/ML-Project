from auto_ml.implementations.augmentators.identity import IdentityAugmentator
from auto_ml.implementations.nodes import DataAugmentatorNode


def get_identity_augmentator_node() -> DataAugmentatorNode:
    """Return an identity augmentator node."""
    return DataAugmentatorNode(
        augmentator=IdentityAugmentator(),
        name="Aug_Identity_K5",
        k_folds=5,
        random_seed=42,
    )
