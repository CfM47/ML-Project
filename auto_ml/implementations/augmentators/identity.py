"""Data augmentation implementations."""

from auto_ml.interfaces import DataAugmentatorInterface, DatasetInterface


class IdentityAugmentator(DataAugmentatorInterface):
    """
    Identity augmentator that returns the dataset unchanged.

    Useful as a baseline or when no augmentation is desired.
    """

    def augment(self, dataset: DatasetInterface) -> DatasetInterface:
        """Return the dataset unchanged."""
        return DatasetInterface(
            samples=list(dataset.samples),
            metadata={**dataset.metadata, "augmentation": "identity"},
        )
