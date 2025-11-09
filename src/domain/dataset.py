"""Defines the interface for a dataset."""

from abc import ABC, abstractmethod
from typing import Iterator, Tuple


class DataSet(ABC):
    """
    An abstract base class for datasets.

    This interface abstracts data loading and provides methods for
    common dataset operations like cross-validation splitting.
    """

    @abstractmethod
    def k_fold_split(self, n_splits: int) -> Iterator[Tuple[list, list]]:
        """
        Provide k-fold cross-validation splits.

        Args:
            n_splits: The number of folds.

        Yields:
            An iterator of (train_indices, test_indices) tuples.

        """
        raise NotImplementedError

    @abstractmethod
    def stratified_split(self, test_size: float) -> Iterator[Tuple[list, list]]:
        """
        Provide a stratified split of the dataset.

        Args:
            test_size: The proportion of the dataset to include in the test split.

        Yields:
            An iterator of (train_indices, test_indices) tuples.

        """
        raise NotImplementedError
