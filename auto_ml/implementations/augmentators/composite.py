"""Composite augmentation implementations for combining multiple augmentations."""

import random
from typing import List

from auto_ml.interfaces import DataAugmentatorInterface, SegmentationDatasetInterface


class SequentialAugmentator(DataAugmentatorInterface):
    """
    Apply multiple augmentations sequentially.

    Each augmentation is applied to the output of the previous one.
    """

    def __init__(self, augmentators: List[DataAugmentatorInterface]) -> None:
        """
        Initialize the sequential augmentator.

        Args:
            augmentators: List of augmentators to apply in order.

        """
        self.augmentators = augmentators

    def augment(
        self, dataset: SegmentationDatasetInterface,
    ) -> SegmentationDatasetInterface:
        """Apply all augmentations sequentially."""
        result = dataset

        for augmentator in self.augmentators:
            result = augmentator.augment(result)

        # Update metadata to reflect all augmentations
        aug_names = [
            aug.__class__.__name__.replace("Augmentator", "").lower()
            for aug in self.augmentators
        ]
        result.metadata["augmentation"] = f"sequential_{'+'.join(aug_names)}"

        return result


class RandomChoiceAugmentator(DataAugmentatorInterface):
    """
    Randomly choose one augmentation from a list to apply.

    Each sample gets one randomly selected augmentation.
    """

    def __init__(
        self,
        augmentators: List[DataAugmentatorInterface],
        random_seed: int | None = None,
    ) -> None:
        """
        Initialize the random choice augmentator.

        Args:
            augmentators: List of augmentators to choose from.
            random_seed: Random seed for reproducibility.

        """
        self.augmentators = augmentators
        self.random_seed = random_seed

    def augment(
        self, dataset: SegmentationDatasetInterface,
    ) -> SegmentationDatasetInterface:
        """Apply one randomly selected augmentation to each sample."""
        rng = random.Random(self.random_seed)
        augmented_samples = []

        for image, mask in dataset.samples:
            # Choose random augmentator
            augmentator = rng.choice(self.augmentators)

            # Create single-sample dataset
            temp_dataset = SegmentationDatasetInterface.from_pairs(
                [(image, mask)],
                metadata=dataset.metadata,
            )

            # Apply augmentation
            aug_dataset = augmentator.augment(temp_dataset)

            # Extract augmented sample
            aug_image, aug_mask = aug_dataset.samples[0]
            augmented_samples.append((aug_image, aug_mask))

        return SegmentationDatasetInterface.from_pairs(
            augmented_samples,
            metadata={**dataset.metadata, "augmentation": "random_choice"},
        )


class RandomApplyAugmentator(DataAugmentatorInterface):
    """
    Apply an augmentation with a given probability.

    Each sample has a chance of being augmented or left unchanged.
    """

    def __init__(
        self,
        augmentator: DataAugmentatorInterface,
        probability: float = 0.5,
        random_seed: int | None = None,
    ) -> None:
        """
        Initialize the random apply augmentator.

        Args:
            augmentator: The augmentator to apply.
            probability: Probability of applying the augmentation (0.0-1.0).
            random_seed: Random seed for reproducibility.

        """
        self.augmentator = augmentator
        self.probability = probability
        self.random_seed = random_seed

    def augment(
        self, dataset: SegmentationDatasetInterface,
    ) -> SegmentationDatasetInterface:
        """Apply augmentation to samples with given probability."""
        rng = random.Random(self.random_seed)
        augmented_samples = []

        for image, mask in dataset.samples:
            if rng.random() < self.probability:
                # Apply augmentation
                temp_dataset = SegmentationDatasetInterface.from_pairs(
                    [(image, mask)],
                    metadata=dataset.metadata,
                )
                aug_dataset = self.augmentator.augment(temp_dataset)
                aug_image, aug_mask = aug_dataset.samples[0]
                augmented_samples.append((aug_image, aug_mask))
            else:
                # Keep original
                augmented_samples.append((image, mask))

        return SegmentationDatasetInterface.from_pairs(
            augmented_samples,
            metadata={**dataset.metadata, "augmentation": "random_apply"},
        )


class OneOfAugmentator(DataAugmentatorInterface):
    """
    Apply one augmentation from a list with specified probabilities.

    Similar to RandomChoiceAugmentator but allows weighted selection.
    """

    def __init__(
        self,
        augmentators: List[DataAugmentatorInterface],
        probabilities: List[float] | None = None,
        random_seed: int | None = None,
    ) -> None:
        """
        Initialize the one-of augmentator.

        Args:
            augmentators: List of augmentators to choose from.
            probabilities: Probability for each augmentator.
                          If None, uses uniform distribution.
            random_seed: Random seed for reproducibility.

        """
        self.augmentators = augmentators

        if probabilities is None:
            self.probabilities = [1.0 / len(augmentators)] * len(augmentators)
        else:
            # Normalize probabilities
            total = sum(probabilities)
            self.probabilities = [p / total for p in probabilities]

        self.random_seed = random_seed

    def augment(
        self, dataset: SegmentationDatasetInterface,
    ) -> SegmentationDatasetInterface:
        """Apply one weighted-random augmentation to each sample."""
        rng = random.Random(self.random_seed)
        augmented_samples = []

        for image, mask in dataset.samples:
            # Choose random augmentator with weights
            augmentator = rng.choices(
                self.augmentators,
                weights=self.probabilities,
                k=1,
            )[0]

            # Create single-sample dataset
            temp_dataset = SegmentationDatasetInterface.from_pairs(
                [(image, mask)],
                metadata=dataset.metadata,
            )

            # Apply augmentation
            aug_dataset = augmentator.augment(temp_dataset)

            # Extract augmented sample
            aug_image, aug_mask = aug_dataset.samples[0]
            augmented_samples.append((aug_image, aug_mask))

        return SegmentationDatasetInterface.from_pairs(
            augmented_samples,
            metadata={**dataset.metadata, "augmentation": "one_of"},
        )


class MultiplyDatasetAugmentator(DataAugmentatorInterface):
    """
    Multiply dataset size by applying different augmentations.

    Creates N copies of the dataset with different augmentations applied.
    Useful for significant data expansion.
    """

    def __init__(
        self,
        augmentators: List[DataAugmentatorInterface],
        include_original: bool = True,
        num_copies: int = 1,
    ) -> None:
        """
        Initialize the multiply dataset augmentator.

        Args:
            augmentators: List of augmentators to apply.
            include_original: Whether to include original samples.
            num_copies: Number of times to repeat the augmentators list.

        """
        self.augmentators = augmentators * num_copies
        self.include_original = include_original

    def augment(
        self, dataset: SegmentationDatasetInterface,
    ) -> SegmentationDatasetInterface:
        """Create multiple augmented copies of the dataset."""
        all_samples = []

        # Include original if requested
        if self.include_original:
            all_samples.extend(dataset.samples)

        # Apply each augmentator
        for augmentator in self.augmentators:
            aug_dataset = augmentator.augment(dataset)
            all_samples.extend(aug_dataset.samples)

        total_multiplier = len(self.augmentators) + (1 if self.include_original else 0)

        return SegmentationDatasetInterface.from_pairs(
            all_samples,
            metadata={
                **dataset.metadata,
                "augmentation": "multiply_dataset",
                "multiplier": total_multiplier,
            },
        )
