"""Defines the interface for a data augmenter."""

from abc import ABC, abstractmethod
from typing import List

from src.domain.image import Image


class DataAugmenter(ABC):
    """
    An abstract base class for data augmenter.

    This interface abstracts the implementation of data augmentation algorithms.
    """

    @abstractmethod
    def augment(self, images: List[Image]) -> List[Image]:
        """
        Perform data augmentation to a list of images.

        Args:
            images: A list of images to be augmented.

        Returns:
            A new list of augmented images.

        """
        raise NotImplementedError
