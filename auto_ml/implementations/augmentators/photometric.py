"""Photometric (pixel-level) augmentation implementations."""

import random

import numpy as np
from scipy import ndimage

from auto_ml.interfaces import DataAugmentatorInterface, DatasetInterface


class BrightnessAugmentator(DataAugmentatorInterface):
    """
    Adjust image brightness.

    Only affects the image, mask remains unchanged.
    """

    def __init__(
        self,
        brightness_range: tuple[float, float] = (0.8, 1.2),
        random_seed: int | None = None,
    ) -> None:
        """
        Initialize the brightness augmentator.

        Args:
            brightness_range: Range of brightness multipliers (min, max).
                             1.0 = no change, < 1.0 = darker, > 1.0 = brighter.
            random_seed: Random seed for reproducibility.

        """
        self.brightness_range = brightness_range
        self.random_seed = random_seed

    def augment(self, dataset: DatasetInterface) -> DatasetInterface:
        """Apply random brightness adjustment to all images."""
        rng = random.Random(self.random_seed)
        augmented_samples = []

        for image, mask in dataset.samples:
            # Random brightness factor
            factor = rng.uniform(*self.brightness_range)

            # Apply brightness
            aug_image = np.clip(image * factor, 0, 255).astype(image.dtype)

            augmented_samples.append((aug_image, mask))

        return DatasetInterface.from_pairs(
            augmented_samples,
            metadata={**dataset.metadata, "augmentation": "brightness"},
        )


class ContrastAugmentator(DataAugmentatorInterface):
    """
    Adjust image contrast.

    Only affects the image, mask remains unchanged.
    """

    def __init__(
        self,
        contrast_range: tuple[float, float] = (0.8, 1.2),
        random_seed: int | None = None,
    ) -> None:
        """
        Initialize the contrast augmentator.

        Args:
            contrast_range: Range of contrast multipliers (min, max).
                           1.0 = no change, < 1.0 = less contrast,
                           > 1.0 = more contrast.
            random_seed: Random seed for reproducibility.

        """
        self.contrast_range = contrast_range
        self.random_seed = random_seed

    def augment(self, dataset: DatasetInterface) -> DatasetInterface:
        """Apply random contrast adjustment to all images."""
        rng = random.Random(self.random_seed)
        augmented_samples = []

        for image, mask in dataset.samples:
            # Random contrast factor
            factor = rng.uniform(*self.contrast_range)

            # Calculate mean for contrast adjustment
            mean = np.mean(image)

            # Apply contrast
            aug_image = np.clip(
                (image - mean) * factor + mean,
                0,
                255,
            ).astype(image.dtype)

            augmented_samples.append((aug_image, mask))

        return DatasetInterface.from_pairs(
            augmented_samples,
            metadata={**dataset.metadata, "augmentation": "contrast"},
        )


class GaussianNoiseAugmentator(DataAugmentatorInterface):
    """
    Add Gaussian noise to images.

    Only affects the image, mask remains unchanged.
    """

    def __init__(
        self,
        noise_std_range: tuple[float, float] = (0.0, 10.0),
        random_seed: int | None = None,
    ) -> None:
        """
        Initialize the Gaussian noise augmentator.

        Args:
            noise_std_range: Range of noise standard deviation (min, max).
            random_seed: Random seed for reproducibility.

        """
        self.noise_std_range = noise_std_range
        self.random_seed = random_seed

    def augment(self, dataset: DatasetInterface) -> DatasetInterface:
        """Add random Gaussian noise to all images."""
        rng = np.random.default_rng(self.random_seed)
        augmented_samples = []

        for image, mask in dataset.samples:
            # Random noise std
            noise_std = rng.uniform(*self.noise_std_range)

            # Generate noise
            noise = rng.normal(0, noise_std, image.shape)

            # Add noise
            aug_image = np.clip(
                image.astype(np.float32) + noise,
                0,
                255,
            ).astype(image.dtype)

            augmented_samples.append((aug_image, mask))

        return DatasetInterface.from_pairs(
            augmented_samples,
            metadata={**dataset.metadata, "augmentation": "gaussian_noise"},
        )


class GaussianBlurAugmentator(DataAugmentatorInterface):
    """
    Apply Gaussian blur to images.

    Only affects the image, mask remains unchanged.
    """

    def __init__(
        self,
        sigma_range: tuple[float, float] = (0.0, 2.0),
        random_seed: int | None = None,
    ) -> None:
        """
        Initialize the Gaussian blur augmentator.

        Args:
            sigma_range: Range of blur sigma values (min, max).
                        Larger values = more blur.
            random_seed: Random seed for reproducibility.

        """
        self.sigma_range = sigma_range
        self.random_seed = random_seed

    def augment(self, dataset: DatasetInterface) -> DatasetInterface:
        """Apply random Gaussian blur to all images."""
        rng = random.Random(self.random_seed)
        augmented_samples = []

        for image, mask in dataset.samples:
            # Random sigma
            sigma = rng.uniform(*self.sigma_range)

            # Apply blur
            if image.ndim == 2:
                aug_image = ndimage.gaussian_filter(image, sigma=sigma)
            else:
                # Apply to each channel
                channels = []
                for c in range(image.shape[2]):
                    blurred = ndimage.gaussian_filter(image[:, :, c], sigma=sigma)
                    channels.append(blurred)
                aug_image = np.stack(channels, axis=2)

            aug_image = aug_image.astype(image.dtype)

            augmented_samples.append((aug_image, mask))

        return DatasetInterface.from_pairs(
            augmented_samples,
            metadata={**dataset.metadata, "augmentation": "gaussian_blur"},
        )


class GammaAugmentator(DataAugmentatorInterface):
    """
    Apply gamma correction to images.

    Only affects the image, mask remains unchanged.
    """

    def __init__(
        self,
        gamma_range: tuple[float, float] = (0.8, 1.2),
        random_seed: int | None = None,
    ) -> None:
        """
        Initialize the gamma augmentator.

        Args:
            gamma_range: Range of gamma values (min, max).
                        1.0 = no change, < 1.0 = brighter,
                        > 1.0 = darker.
            random_seed: Random seed for reproducibility.

        """
        self.gamma_range = gamma_range
        self.random_seed = random_seed

    def augment(self, dataset: DatasetInterface) -> DatasetInterface:
        """Apply random gamma correction to all images."""
        rng = random.Random(self.random_seed)
        augmented_samples = []

        for image, mask in dataset.samples:
            # Random gamma
            gamma = rng.uniform(*self.gamma_range)

            # Normalize to [0, 1]
            normalized = image.astype(np.float32) / 255.0

            # Apply gamma
            corrected = np.power(normalized, gamma)

            # Scale back to [0, 255]
            aug_image = (corrected * 255).astype(image.dtype)

            augmented_samples.append((aug_image, mask))

        return DatasetInterface.from_pairs(
            augmented_samples,
            metadata={**dataset.metadata, "augmentation": "gamma"},
        )
