"""Data augmentation implementations."""

from auto_ml.interfaces import DataAugmentatorInterface, SegmentationDatasetInterface


class IdentityAugmentator(DataAugmentatorInterface):
    """
    Identity augmentator that returns the dataset unchanged.

    Useful as a baseline or when no augmentation is desired.
    """

    def augment(
        self,
        dataset: SegmentationDatasetInterface,
    ) -> SegmentationDatasetInterface:
        """Return the dataset unchanged."""
        return SegmentationDatasetInterface(
            samples=list(dataset.samples),
            metadata={**dataset.metadata, "augmentation": "identity"},
        )
