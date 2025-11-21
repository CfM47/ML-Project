"""Defines the interface for a classifier."""

from abc import ABC, abstractmethod
from typing import Any

from src.domain.image import Image


class Classifier(ABC):
    """
    An abstract base class for classifiers.

    This interface abstracts the implementation of a classification model.
    """

    @abstractmethod
    def classify(self, features: list, image: Image) -> Any:
        """
        Classifies an image based on extracted features.

        Args:
            features: A list of features extracted from the image.
            image: The image to be classified.

        Returns:
            The classification result. The type of the result is
            left to the implementation (e.g., a string, a custom
            data class).

        """
        raise NotImplementedError
