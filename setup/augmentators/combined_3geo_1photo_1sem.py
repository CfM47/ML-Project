"""Combined augmentation node: 3 geometric, 1 photometric, 1 SEM."""

from auto_ml.implementations.augmentators.composite import (
    MultiplyDatasetAugmentator,
    RandomApplyAugmentator,
    SequentialAugmentator,
)
from auto_ml.implementations.augmentators.geometric import (
    HorizontalFlipAugmentator,
    RotationAugmentator,
    VerticalFlipAugmentator,
)
from auto_ml.implementations.augmentators.photometric import (
    ContrastAugmentator,
)
from auto_ml.implementations.augmentators.sem_specific import (
    AdaptiveHistogramEqualizationAugmentator,
)
from auto_ml.implementations.nodes import DataAugmentatorNode


def get_combined_3geo_1photo_1sem_node(num_copies: int = 1) -> DataAugmentatorNode:
    """
    Create a node with 3 geometric, 1 photometric, and 1 SEM augmentation.

    Args:
        num_copies: Number of augmented copies to create (default: 1).

    """
    return DataAugmentatorNode(
        augmentator=MultiplyDatasetAugmentator(
            augmentators=[
                SequentialAugmentator(
                    augmentators=[
                        # Geometric augmentations (independent)
                        RandomApplyAugmentator(
                            augmentator=HorizontalFlipAugmentator(),
                            probability=0.5,
                        ),
                        RandomApplyAugmentator(
                            augmentator=VerticalFlipAugmentator(),
                            probability=0.5,
                        ),
                        RandomApplyAugmentator(
                            augmentator=RotationAugmentator(angle_range=(-15.0, 15.0)),
                            probability=0.4,
                        ),
                        # 1 Photometric augmentation
                        RandomApplyAugmentator(
                            augmentator=ContrastAugmentator(contrast_range=(0.8, 1.2)),
                            probability=0.4,
                        ),
                        # 1 SEM augmentation
                        RandomApplyAugmentator(
                            augmentator=AdaptiveHistogramEqualizationAugmentator(
                                clip_limit=2.0,
                                tile_grid_size=(8, 8),
                            ),
                            probability=0.6,
                        ),
                    ],
                ),
            ],
            num_copies=num_copies,
            include_original=True,
        ),
        name=f"Combined_3Geo_1Photo_1SEM_x{num_copies}",
    )
