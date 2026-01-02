"""Combined augmentation node: 1 geometric, 1 photometric, 1 SEM."""

from auto_ml.implementations.augmentators.composite import (
    MultiplyDatasetAugmentator,
    RandomApplyAugmentator,
    SequentialAugmentator,
)
from auto_ml.implementations.augmentators.geometric import (
    HorizontalFlipAugmentator,
)
from auto_ml.implementations.augmentators.photometric import (
    ContrastAugmentator,
)
from auto_ml.implementations.augmentators.sem_specific import (
    ElasticDeformationAugmentator,
)
from auto_ml.implementations.nodes import DataAugmentatorNode


def get_combined_1geo_1photo_1sem_node(num_copies: int = 1) -> DataAugmentatorNode:
    """Create a node with 1 geometric, 1 photometric, and 1 SEM augmentation.

    Args:
        num_copies: Number of augmented copies to create (default: 1).

    """
    return DataAugmentatorNode(
        augmentator=MultiplyDatasetAugmentator(
            augmentators=[
                SequentialAugmentator(
                    augmentators=[
                        # 1 Geometric augmentation
                        RandomApplyAugmentator(
                            augmentator=HorizontalFlipAugmentator(),
                            probability=0.5,
                        ),
                        # 1 Photometric augmentation
                        RandomApplyAugmentator(
                            augmentator=ContrastAugmentator(contrast_range=(0.8, 1.2)),
                            probability=0.3,
                        ),
                        # 1 SEM augmentation
                        RandomApplyAugmentator(
                            augmentator=ElasticDeformationAugmentator(
                                alpha=25.0,
                                sigma=3.5,
                            ),
                            probability=0.6,
                        ),
                    ],
                ),
            ],
            num_copies=num_copies,
            include_original=True,
        ),
        name=f"Combined_1Geo_1Photo_1SEM_x{num_copies}",
    )
