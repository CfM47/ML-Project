"""Defines the interface for a segmenter."""

from abc import ABC, abstractmethod

from src.domain.image import Image


class Segmenter(ABC):
    """
    An abstract base class for an image segmenter.

    The purpose of a segmenter is to extract features from an image.
    """

    @abstractmethod
    def segment(self, image: Image) -> list:
        """
        Segments an image to extract features.

        Args:
            image: The image to be segmented.

        Returns:
            A list of features extracted from the image.

        """
        raise NotImplementedError
