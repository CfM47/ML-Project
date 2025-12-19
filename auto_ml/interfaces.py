"""
AutoML Training Interfaces Module.

This module defines all the interfaces for the AutoML training pipeline:
- ModelInputInterface: 512x512 image input
- ModelOutputInterface: 512x512 matrix with values 0, 1, or 2
- DatasetInterface: Set of original images and corresponding masks
- DataAugmentatorInterface: Dataset transformation (Dataset -> Dataset)
- DataAugmentatorNodeInterface: Dataset -> List of (Dataset, Dataset) tuples
- ModelNodeInterface: List of (Dataset, Dataset) tuples -> Dictionary
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray


# ==============================================================================
# Type Aliases for clarity
# ==============================================================================

# 512x512 image as numpy array (height, width, channels)
ImageArray = NDArray[np.uint8]  # Shape: (512, 512, 3) for RGB or (512, 512) for grayscale

# 512x512 segmentation mask with values 0, 1, or 2
MaskArray = NDArray[np.uint8]  # Shape: (512, 512), values in {0, 1, 2}


# ==============================================================================
# Model Input Interface
# ==============================================================================

@dataclass
class ModelInputInterface:
    """
    Model Input Interface: Represents a 512x512 image input.
    
    Attributes:
        image: A numpy array of shape (512, 512, 3) for RGB images
               or (512, 512) for grayscale images.
    """
    
    image: ImageArray
    
    def __post_init__(self) -> None:
        """Validate the image dimensions."""
        if self.image.ndim == 2:
            if self.image.shape != (512, 512):
                raise ValueError(
                    f"Grayscale image must be 512x512, got {self.image.shape}"
                )
        elif self.image.ndim == 3:
            if self.image.shape[:2] != (512, 512):
                raise ValueError(
                    f"Image height and width must be 512x512, got {self.image.shape[:2]}"
                )
        else:
            raise ValueError(
                f"Image must be 2D (grayscale) or 3D (color), got {self.image.ndim}D"
            )
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the image."""
        return self.image.shape
    
    @property
    def is_grayscale(self) -> bool:
        """Check if the image is grayscale."""
        return self.image.ndim == 2
    
    def to_tensor(self) -> NDArray[np.float32]:
        """
        Convert the image to a normalized float tensor.
        
        Returns:
            Normalized image array with values in [0, 1].
        """
        return self.image.astype(np.float32) / 255.0


# ==============================================================================
# Model Output Interface
# ==============================================================================

@dataclass
class ModelOutputInterface:
    """
    Model Output Interface: Represents a 512x512 segmentation mask.
    
    The mask contains integer values 0, 1, or 2 representing different
    segmentation classes.
    
    Attributes:
        mask: A numpy array of shape (512, 512) with values in {0, 1, 2}.
    """
    
    mask: MaskArray
    
    def __post_init__(self) -> None:
        """Validate the mask dimensions and values."""
        if self.mask.shape != (512, 512):
            raise ValueError(
                f"Mask must be 512x512, got {self.mask.shape}"
            )
        
        unique_values = np.unique(self.mask)
        valid_values = {0, 1, 2}
        if not set(unique_values).issubset(valid_values):
            invalid = set(unique_values) - valid_values
            raise ValueError(
                f"Mask values must be 0, 1, or 2. Found invalid values: {invalid}"
            )
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the mask."""
        return self.mask.shape
    
    def get_class_counts(self) -> Dict[int, int]:
        """
        Count pixels for each class.
        
        Returns:
            Dictionary mapping class index to pixel count.
        """
        unique, counts = np.unique(self.mask, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}
    
    def to_one_hot(self, num_classes: int = 3) -> NDArray[np.float32]:
        """
        Convert to one-hot encoded representation.
        
        Args:
            num_classes: Number of classes (default: 3).
        
        Returns:
            One-hot encoded array of shape (512, 512, num_classes).
        """
        one_hot = np.zeros((512, 512, num_classes), dtype=np.float32)
        for c in range(num_classes):
            one_hot[:, :, c] = (self.mask == c).astype(np.float32)
        return one_hot


# ==============================================================================
# Type Alias for Dataset Sample
# ==============================================================================

# A dataset sample is a pair of (image, mask)
DatasetSample = Tuple[ImageArray, MaskArray]


# ==============================================================================
# Dataset Interface
# ==============================================================================

@dataclass
class DatasetInterface:
    """
    Dataset Interface: A collection of 512x512 original images and their masks.
    
    The dataset is stored as a list of pairs, where each pair contains
    an original image and its corresponding mask.
    
    Attributes:
        samples: List of (image, mask) pairs.
        metadata: Optional dictionary containing dataset metadata.
    """
    
    samples: List[DatasetSample] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[ModelInputInterface, ModelOutputInterface]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample.
        
        Returns:
            Tuple of (ModelInputInterface, ModelOutputInterface).
        """
        image, mask = self.samples[idx]
        return (
            ModelInputInterface(image=image),
            ModelOutputInterface(mask=mask)
        )
    
    def __iter__(self):
        """Iterate over all samples in the dataset."""
        for sample in self.samples:
            yield sample
    
    def get_raw_sample(self, idx: int) -> DatasetSample:
        """
        Get a raw sample (image, mask) pair without wrapping in interfaces.
        
        Args:
            idx: Index of the sample.
        
        Returns:
            Tuple of (image array, mask array).
        """
        return self.samples[idx]
    
    @property
    def images(self) -> List[ImageArray]:
        """Get all images from the dataset."""
        return [sample[0] for sample in self.samples]
    
    @property
    def masks(self) -> List[MaskArray]:
        """Get all masks from the dataset."""
        return [sample[1] for sample in self.samples]
    
    def add_sample(self, image: ImageArray, mask: MaskArray) -> None:
        """
        Add a new sample to the dataset.
        
        Args:
            image: 512x512 image array.
            mask: 512x512 mask array with values in {0, 1, 2}.
        """
        # Validate by creating the interface objects
        _ = ModelInputInterface(image=image)
        _ = ModelOutputInterface(mask=mask)
        
        self.samples.append((image, mask))
    
    def add_pair(self, pair: DatasetSample) -> None:
        """
        Add a new (image, mask) pair to the dataset.
        
        Args:
            pair: Tuple of (image array, mask array).
        """
        image, mask = pair
        self.add_sample(image, mask)
    
    @classmethod
    def from_pairs(
        cls, 
        pairs: List[DatasetSample], 
        metadata: Dict[str, Any] | None = None
    ) -> "DatasetInterface":
        """
        Create a dataset from a list of (image, mask) pairs.
        
        Args:
            pairs: List of (image, mask) tuples.
            metadata: Optional metadata dictionary.
        
        Returns:
            DatasetInterface instance.
        """
        dataset = cls(samples=[], metadata=metadata or {})
        for image, mask in pairs:
            dataset.add_sample(image, mask)
        return dataset
    
    def split(
        self, 
        ratio: float = 0.8, 
        shuffle: bool = True,
        random_seed: int | None = None
    ) -> Tuple["DatasetInterface", "DatasetInterface"]:
        """
        Split the dataset into two parts.
        
        Args:
            ratio: Ratio for the first split (default: 0.8 for 80/20 split).
            shuffle: Whether to shuffle before splitting (default: True).
            random_seed: Random seed for reproducibility.
        
        Returns:
            Tuple of two DatasetInterface objects.
        """
        n = len(self)
        indices = np.arange(n)
        
        if shuffle:
            rng = np.random.default_rng(random_seed)
            rng.shuffle(indices)
        
        split_idx = int(n * ratio)
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_dataset = DatasetInterface(
            samples=[self.samples[i] for i in train_indices],
            metadata={**self.metadata, "split": "train"}
        )
        
        val_dataset = DatasetInterface(
            samples=[self.samples[i] for i in val_indices],
            metadata={**self.metadata, "split": "val"}
        )
        
        return train_dataset, val_dataset


# ==============================================================================
# Data Augmentator Interface (Base Class)
# ==============================================================================

class DataAugmentatorInterface(ABC):
    """
    Data Augmentator Interface: Transforms a Dataset into another Dataset.
    
    This is the base class for all data augmentation operations.
    Input: DatasetInterface
    Output: DatasetInterface
    """
    
    @abstractmethod
    def augment(self, dataset: DatasetInterface) -> DatasetInterface:
        """
        Apply augmentation to the dataset.
        
        Args:
            dataset: Input dataset to augment.
        
        Returns:
            Augmented dataset.
        """
        pass
    
    def __call__(self, dataset: DatasetInterface) -> DatasetInterface:
        """Allow calling the augmentator as a function."""
        return self.augment(dataset)


# ==============================================================================
# Data Augmentator Node Interface
# ==============================================================================

class DataAugmentatorNodeInterface(ABC):
    """
    Data Augmentator Node Interface: Processes a Dataset and returns 
    a list of tuples, where each tuple contains two Datasets.
    
    This can be used for creating train/validation splits, cross-validation
    folds, or other dataset partitioning schemes.
    
    Input: DatasetInterface
    Output: List[Tuple[DatasetInterface, DatasetInterface]]
    """
    
    @abstractmethod
    def process(
        self, dataset: DatasetInterface
    ) -> List[Tuple[DatasetInterface, DatasetInterface]]:
        """
        Process the dataset and return a list of dataset pairs.
        
        Args:
            dataset: Input dataset to process.
        
        Returns:
            List of tuples, each containing two datasets
            (e.g., train/validation pairs for k-fold cross-validation).
        """
        pass
    
    def __call__(
        self, dataset: DatasetInterface
    ) -> List[Tuple[DatasetInterface, DatasetInterface]]:
        """Allow calling the node as a function."""
        return self.process(dataset)


# ==============================================================================
# Model Node Interface
# ==============================================================================

class ModelNodeInterface(ABC):
    """
    Model Node Interface: Receives a list of dataset tuples and returns 
    a dictionary containing training results.
    
    This interface represents the training/evaluation node in the pipeline.
    
    Input: List[Tuple[DatasetInterface, DatasetInterface]]
           (e.g., list of (train_dataset, val_dataset) pairs)
    Output: Dict[str, Any] containing training metrics, model info, etc.
    """
    
    @abstractmethod
    def train(
        self, 
        dataset_pairs: List[Tuple[DatasetInterface, DatasetInterface]]
    ) -> Dict[str, Any]:
        """
        Train the model on the provided dataset pairs.
        
        Args:
            dataset_pairs: List of (train_dataset, val_dataset) tuples.
        
        Returns:
            Dictionary containing:
                - metrics: Training and validation metrics
                - model_path: Path to saved model (if applicable)
                - training_history: History of training progress
                - Additional implementation-specific data
        """
        pass
    
    def __call__(
        self, 
        dataset_pairs: List[Tuple[DatasetInterface, DatasetInterface]]
    ) -> Dict[str, Any]:
        """Allow calling the node as a function."""
        return self.train(dataset_pairs)
