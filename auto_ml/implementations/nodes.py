"""Pipeline node implementations."""

from typing import Any, Dict, List, Tuple

import numpy as np

from auto_ml.interfaces import (
    DataAugmentatorInterface,
    DataAugmentatorNodeInterface,
    EvaluatorInterface,
    EvaluatorNodeInterface,
    MaskPair,
    ModelNodeInterface,
    SegmentationDatasetInterface,
    SegmentationModelInterface,
)


class DataAugmentatorNode(DataAugmentatorNodeInterface):
    """
    Data Augmentator Node implementation.

    Splits the dataset into K folds (or single split) and applies augmentation
    to the training set of each fold.
    """

    def __init__(
        self,
        augmentator: DataAugmentatorInterface,
        name: str = "DataAugmentatorNode",
        k_folds: int = 5,
        test_size: float = 0.2,
        random_seed: int = 42,
    ) -> None:
        """
        Initialize the node.

        Args:
            augmentator: The augmentator to apply to training sets.
            name: Name of this node instance.
            k_folds: Number of folds for Cross Validation.
                     If 1, performs a single random split based on test_size.
            test_size: Fraction of data for validation if k_folds=1.
            random_seed: Seed for reproducibility.

        """
        self.augmentator = augmentator
        self.name = name
        self.k_folds = k_folds
        self.test_size = test_size
        self.random_seed = random_seed

    def process(
        self,
        dataset: SegmentationDatasetInterface,
    ) -> List[Tuple[SegmentationDatasetInterface, SegmentationDatasetInterface]]:
        """
        Process the dataset.

        Returns:
            List of (augmented_train_dataset, val_dataset) tuples.

        """
        n_samples = len(dataset)
        indices = np.arange(n_samples)
        rng = np.random.default_rng(self.random_seed)

        # Shuffle indices once
        rng.shuffle(indices)

        results = []

        if self.k_folds <= 1:
            # Single Split
            split_idx = int(n_samples * (1 - self.test_size))
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]

            # Create Datasets
            train_dataset = SegmentationDatasetInterface.from_pairs(
                [dataset.samples[i] for i in train_indices],
                metadata={**dataset.metadata, "split": "train", "fold": 0},
            )
            val_dataset = SegmentationDatasetInterface.from_pairs(
                [dataset.samples[i] for i in val_indices],
                metadata={**dataset.metadata, "split": "val", "fold": 0},
            )

            # Augment Train
            aug_train = self.augmentator.augment(train_dataset)

            results.append((aug_train, val_dataset))

        else:
            # K-Fold Split
            fold_sizes = np.full(self.k_folds, n_samples // self.k_folds, dtype=int)
            fold_sizes[: n_samples % self.k_folds] += 1
            current = 0

            for i in range(self.k_folds):
                start, stop = current, current + fold_sizes[i]
                val_mask = np.zeros(n_samples, dtype=bool)
                val_mask[start:stop] = True

                val_indices_fold = indices[val_mask]
                train_indices_fold = indices[~val_mask]

                # Create Datasets
                train_dataset = SegmentationDatasetInterface.from_pairs(
                    [dataset.samples[j] for j in train_indices_fold],
                    metadata={**dataset.metadata, "split": "train", "fold": i},
                )
                val_dataset = SegmentationDatasetInterface.from_pairs(
                    [dataset.samples[j] for j in val_indices_fold],
                    metadata={**dataset.metadata, "split": "val", "fold": i},
                )

                # Augment Train
                aug_train = self.augmentator.augment(train_dataset)

                results.append((aug_train, val_dataset))

                current = stop

        return results


class ModelNode(ModelNodeInterface):
    """
    Generic Model Node implementation.

    Manages the training and evaluation of a model across multiple dataset pairs
    (e.g., cross-validation folds).
    """

    def __init__(
        self,
        model: SegmentationModelInterface,
        name: str = "ModelNode",
    ) -> None:
        """
        Initialize the Model Node.

        Args:
            model: The model to train and evaluate.
            name: Name of this node instance.

        """
        self.model = model
        self.name = name

    def train(
        self,
        dataset_pairs: List[
            Tuple[SegmentationDatasetInterface, SegmentationDatasetInterface]
        ],
    ) -> List[List[MaskPair]]:
        """
        Train the model on the provided dataset pairs.

        Args:
            dataset_pairs: List of (train_dataset, val_dataset) tuples.

        Returns:
            List of mask pair lists, one per dataset pair/fold.

        """
        all_mask_pairs: List[List[MaskPair]] = []

        for i, (train_dataset, val_dataset) in enumerate(dataset_pairs):
            print(f"Processing split {i + 1}/{len(dataset_pairs)}...")

            # Train model
            print(f"  Training on {len(train_dataset)} samples...")
            _ = self.model.train(train_dataset)

            # Evaluate on validation set
            print(f"  Evaluating on {len(val_dataset)} samples...")
            mask_pairs = self.model.evaluate(val_dataset)

            all_mask_pairs.append(mask_pairs)
            print(f"  Split {i + 1}: Collected {len(mask_pairs)} mask pairs")

        return all_mask_pairs


class EvaluatorNode(EvaluatorNodeInterface):
    """
    Evaluator Node implementation.

    Receives named evaluators and runs each on the mask pairs,
    returning a dictionary of results.
    """

    def __init__(
        self,
        evaluators: Dict[str, EvaluatorInterface],
        name: str = "EvaluatorNode",
    ) -> None:
        """
        Initialize the Evaluator Node.

        Args:
            evaluators: Dictionary mapping evaluator names to
                EvaluatorInterface instances.
            name: Name of this node instance.

        """
        self.evaluators = evaluators
        self.name = name

    def evaluate(self, mask_pairs: List[List[MaskPair]]) -> Dict[str, Any]:
        """
        Run all evaluators on the mask pairs.

        Args:
            mask_pairs: List of mask pair lists from ModelNode.

        Returns:
            Dictionary mapping evaluator names to their results.

        """
        print(f"\n{self.name}: Running evaluators...")

        results: Dict[str, Any] = {}

        for eval_name, evaluator in self.evaluators.items():
            print(f"  Running evaluator: {eval_name}")
            result = evaluator.evaluate(mask_pairs)
            results[eval_name] = result
            print(f"    Result: {result}")

        print(f"{self.name}: Evaluation complete.")
        return results
