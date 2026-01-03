"""Dataset utilities for training and validation."""

from typing import List, Tuple

import numpy as np

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
from auto_ml.interfaces import SegmentationDatasetInterface


def subsample_dataset(
    dataset: SegmentationDatasetInterface,
    percentage: int,
    seed: int,
) -> SegmentationDatasetInterface:
    """
    Randomly subsample dataset to a given percentage.

    Args:
        dataset: The dataset to subsample.
        percentage: Percentage of samples to keep (1-100).
        seed: Random seed for reproducibility.

    Returns:
        Subsampled dataset.

    """
    if percentage >= 100:
        return dataset

    n_samples = len(dataset)
    n_keep = max(1, int(n_samples * percentage / 100))

    rng = np.random.default_rng(seed)
    indices = rng.choice(n_samples, size=n_keep, replace=False)

    return SegmentationDatasetInterface.from_pairs(
        [dataset.samples[i] for i in indices],
        metadata={**dataset.metadata, "subsampled_percentage": percentage},
    )


def create_kfold_splits(
    dataset: SegmentationDatasetInterface,
    n_folds: int,
    seed: int,
) -> List[Tuple[SegmentationDatasetInterface, SegmentationDatasetInterface]]:
    """
    Create k-fold train/val splits.

    Replicate the logic from DataAugmentatorNode.process() but without
    applying augmentation (augmentation is handled separately).

    Args:
        dataset: The dataset to split.
        n_folds: Number of folds.
        seed: Random seed for reproducibility.

    Returns:
        List of (train_dataset, val_dataset) tuples.

    """
    n_samples = len(dataset)
    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    fold_sizes = np.full(n_folds, n_samples // n_folds, dtype=int)
    fold_sizes[: n_samples % n_folds] += 1

    splits = []
    current = 0

    for i in range(n_folds):
        start, stop = current, current + fold_sizes[i]
        val_mask = np.zeros(n_samples, dtype=bool)
        val_mask[start:stop] = True

        val_indices = indices[val_mask]
        train_indices = indices[~val_mask]

        train_dataset = SegmentationDatasetInterface.from_pairs(
            [dataset.samples[j] for j in train_indices],
            metadata={**dataset.metadata, "split": "train", "fold": i},
        )
        val_dataset = SegmentationDatasetInterface.from_pairs(
            [dataset.samples[j] for j in val_indices],
            metadata={**dataset.metadata, "split": "val", "fold": i},
        )

        splits.append((train_dataset, val_dataset))
        current = stop

    return splits


def create_augmentator(num_copies: int = 2) -> MultiplyDatasetAugmentator:
    """
    Create the 2Geo2Photo1SEM augmentator.

    Same pipeline as setup/augmentators/combined_2geo_2photo_1sem.py.

    Args:
        num_copies: Number of augmented copies to create.

    Returns:
        Configured augmentator.

    """
    return MultiplyDatasetAugmentator(
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
                    # 2 Photometric augmentations
                    RandomApplyAugmentator(
                        augmentator=BrightnessAugmentator(
                            brightness_range=(0.85, 1.15),
                        ),
                        probability=0.3,
                    ),
                    RandomApplyAugmentator(
                        augmentator=ContrastAugmentator(
                            contrast_range=(0.85, 1.15),
                        ),
                        probability=0.4,
                    ),
                    # 1 SEM augmentation
                    RandomApplyAugmentator(
                        augmentator=ElasticDeformationAugmentator(
                            alpha=25.0,
                            sigma=3.5,
                        ),
                        probability=0.5,
                    ),
                ],
            ),
        ],
        num_copies=num_copies,
        include_original=True,
    )
