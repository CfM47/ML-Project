"""Combined augmentation node: 2 geometric, 1 photometric, 1 SEM."""

from auto_ml.implementations.augmentators.composite import (
    MultiplyDatasetAugmentator,
    RandomApplyAugmentator,
    SequentialAugmentator,
)
from auto_ml.implementations.augmentators.geometric import (
    HorizontalFlipAugmentator,
    RotationAugmentator,
)
from auto_ml.implementations.augmentators.photometric import (
    BrightnessAugmentator,
    ContrastAugmentator,
)
from auto_ml.implementations.augmentators.sem_specific import (
    ElasticDeformationAugmentator,
)
from auto_ml.implementations.nodes import DataAugmentatorNode


def get_combined_2geo_1photo_1sem_node(num_copies: int = 1) -> DataAugmentatorNode:
    """Create a node with 2 geometric, 1 photometric, and 1 SEM augmentation.

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
                            augmentator=HorizontalFlipAugmentator(),
                            probability=0.5,
                        ),
                        RandomApplyAugmentator(
                            augmentator=RotationAugmentator(angle_range=(-15.0, 15.0)),
                            probability=0.5,
                        ),
                        # 1 Photometric augmentation
                        RandomApplyAugmentator(
                            augmentator=SequentialAugmentator(
                                augmentators=[
                                    BrightnessAugmentator(brightness_range=(0.85, 1.15)),
                                    ContrastAugmentator(contrast_range=(0.85, 1.15)),
                                ],
                            ),
                            probability=0.3,
                        ),
                        # 1 SEM augmentation
                        RandomApplyAugmentator(
                            augmentator=ElasticDeformationAugmentator(
                                alpha=30.0,
                                sigma=4.0,
                            ),
                            probability=0.6,
                        ),
                    ],
                ),
            ],
            num_copies=num_copies,
            include_original=True,
        ),
        name=f"Combined_2Geo_1Photo_1SEM_x{num_copies}",
    )
