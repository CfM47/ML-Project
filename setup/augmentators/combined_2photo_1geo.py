"""Combined augmentation node: 2 photometric, 1 geometric (no SEM)."""

from auto_ml.implementations.augmentators.composite import (
    MultiplyDatasetAugmentator,
    RandomApplyAugmentator,
    SequentialAugmentator,
)
from auto_ml.implementations.augmentators.geometric import (
    RotationAugmentator,
)
from auto_ml.implementations.augmentators.photometric import (
    GaussianBlurAugmentator,
    GaussianNoiseAugmentator,
)
from auto_ml.implementations.nodes import DataAugmentatorNode


def get_combined_2photo_1geo_node(num_copies: int = 1) -> DataAugmentatorNode:
    """Create a node with 2 photometric and 1 geometric augmentation (no SEM).

    Args:
        num_copies: Number of augmented copies to create (default: 1).

    """
    return DataAugmentatorNode(
        augmentator=MultiplyDatasetAugmentator(
            augmentators=[
                SequentialAugmentator(
                    augmentators=[
                        # 2 Photometric augmentations
                        RandomApplyAugmentator(
                            augmentator=GaussianNoiseAugmentator(noise_std_range=(0.0, 8.0)),
                            probability=0.4,
                        ),
                        RandomApplyAugmentator(
                            augmentator=GaussianBlurAugmentator(sigma_range=(0.0, 1.5)),
                            probability=0.4,
                        ),
                        # 1 Geometric augmentation
                        RandomApplyAugmentator(
                            augmentator=RotationAugmentator(angle_range=(-10.0, 10.0)),
                            probability=0.5,
                        ),
                    ],
                ),
            ],
            num_copies=num_copies,
            include_original=True,
        ),
        name=f"Combined_2Photo_1Geo_x{num_copies}",
    )
