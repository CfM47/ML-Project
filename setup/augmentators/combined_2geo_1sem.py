"""Combined augmentation node: 2 geometric, 1 SEM (no photometric)."""

from auto_ml.implementations.augmentators.composite import (
    MultiplyDatasetAugmentator,
    RandomApplyAugmentator,
    SequentialAugmentator,
)
from auto_ml.implementations.augmentators.geometric import (
    ScaleAugmentator,
    VerticalFlipAugmentator,
)
from auto_ml.implementations.augmentators.sem_specific import (
    AdaptiveHistogramEqualizationAugmentator,
)
from auto_ml.implementations.nodes import DataAugmentatorNode


def get_combined_2geo_1sem_node(num_copies: int = 1) -> DataAugmentatorNode:
    """Create a node with 2 geometric and 1 SEM augmentation (no photometric).

    Args:
        num_copies: Number of augmented copies to create (default: 1).

    """
    return DataAugmentatorNode(
        augmentator=MultiplyDatasetAugmentator(
            augmentators=[
                SequentialAugmentator(
                    augmentators=[
                        # 2 Geometric augmentations
                        RandomApplyAugmentator(
                            augmentator=VerticalFlipAugmentator(),
                            probability=0.5,
                        ),
                        RandomApplyAugmentator(
                            augmentator=ScaleAugmentator(scale_range=(0.9, 1.1)),
                            probability=0.5,
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
        name=f"Combined_2Geo_1SEM_x{num_copies}",
    )
