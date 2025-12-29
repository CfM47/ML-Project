"""SEM-specific augmentation implementations for microscopy images."""

import random
from typing import Tuple

import cv2
import numpy as np
from scipy import ndimage

from auto_ml.interfaces import DataAugmentatorInterface, DatasetInterface


class ElasticDeformationAugmentator(DataAugmentatorInterface):
    """
    Apply elastic deformation to simulate natural rock texture variations.

    Useful for SEM images where structures can appear with slight warping
    due to sample preparation or natural material heterogeneity.
    """

    def __init__(
        self,
        alpha: float = 50.0,
        sigma: float = 5.0,
        random_seed: int | None = None,
    ) -> None:
        """
        Initialize the elastic deformation augmentator.

        Args:
            alpha: Strength of deformation (higher = more distortion).
            sigma: Smoothness of deformation field (higher = smoother).
            random_seed: Random seed for reproducibility.

        """
        self.alpha = alpha
        self.sigma = sigma
        self.random_seed = random_seed

    def augment(self, dataset: DatasetInterface) -> DatasetInterface:
        """Apply elastic deformation to all samples."""
        rng = np.random.default_rng(self.random_seed)
        augmented_samples = []

        for image, mask in dataset.samples:
            h, w = image.shape[:2]

            # Generate random displacement fields
            dx = rng.uniform(-1, 1, (h, w)) * self.alpha
            dy = rng.uniform(-1, 1, (h, w)) * self.alpha

            # Smooth the displacement fields
            dx = ndimage.gaussian_filter(dx, self.sigma, mode='constant', cval=0)
            dy = ndimage.gaussian_filter(dy, self.sigma, mode='constant', cval=0)

            # Create meshgrid for remapping
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (x + dx).astype(np.float32)
            map_y = (y + dy).astype(np.float32)

            # Apply deformation to image
            if image.ndim == 2:
                aug_image = cv2.remap(
                    image,
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )
            else:
                aug_image = cv2.remap(
                    image,
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )

            # Apply same deformation to mask (nearest neighbor)
            aug_mask = cv2.remap(
                mask.astype(np.float32),
                map_x,
                map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REFLECT,
            ).astype(mask.dtype)

            augmented_samples.append((aug_image, aug_mask))

        return DatasetInterface.from_pairs(
            augmented_samples,
            metadata={**dataset.metadata, "augmentation": "elastic_deformation"},
        )


class AdaptiveHistogramEqualizationAugmentator(DataAugmentatorInterface):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Enhances local contrast, revealing subtle features in SEM images
    that may be important for segmentation. Only affects the image.
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> None:
        """
        Initialize the CLAHE augmentator.

        Args:
            clip_limit: Threshold for contrast limiting (higher = more contrast).
            tile_grid_size: Size of grid for local histogram equalization.

        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def augment(self, dataset: DatasetInterface) -> DatasetInterface:
        """Apply CLAHE to all images."""
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )

        augmented_samples = []

        for image, mask in dataset.samples:
            # Apply CLAHE
            if image.ndim == 2:
                aug_image = clahe.apply(image)
            else:
                # Apply to each channel separately
                channels = []
                for c in range(image.shape[2]):
                    channel = clahe.apply(image[:, :, c])
                    channels.append(channel)
                aug_image = np.stack(channels, axis=2)

            augmented_samples.append((aug_image, mask))

        return DatasetInterface.from_pairs(
            augmented_samples,
            metadata={**dataset.metadata, "augmentation": "clahe"},
        )


class ChargingArtifactAugmentator(DataAugmentatorInterface):
    """
    Simulate charging artifacts in SEM imaging of non-conductive samples.

    Adds localized brightness variations that mimic electron charging
    effects. Only affects the image.
    """

    def __init__(
        self,
        intensity_range: Tuple[float, float] = (0.7, 1.3),
        num_spots: Tuple[int, int] = (1, 3),
        spot_size_range: Tuple[int, int] = (30, 80),
        random_seed: int | None = None,
    ) -> None:
        """
        Initialize the charging artifact augmentator.

        Args:
            intensity_range: Range of brightness multipliers for charged spots.
            num_spots: Range of number of charging spots to add.
            spot_size_range: Range of spot radius in pixels.
            random_seed: Random seed for reproducibility.

        """
        self.intensity_range = intensity_range
        self.num_spots = num_spots
        self.spot_size_range = spot_size_range
        self.random_seed = random_seed

    def augment(self, dataset: DatasetInterface) -> DatasetInterface:
        """Add charging artifacts to all images."""
        rng = random.Random(self.random_seed)

        augmented_samples = []

        for image, mask in dataset.samples:
            h, w = image.shape[:2]
            aug_image = image.astype(np.float32).copy()

            # Random number of spots
            n_spots = rng.randint(*self.num_spots)

            for _ in range(n_spots):
                # Random spot location
                cx = rng.randint(0, w - 1)
                cy = rng.randint(0, h - 1)

                # Random spot size
                radius = rng.randint(*self.spot_size_range)

                # Random intensity
                intensity = rng.uniform(*self.intensity_range)

                # Create Gaussian spot mask
                y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
                distance = np.sqrt(x*x + y*y)
                spot_mask = np.exp(-(distance**2) / (2 * (radius/2)**2))

                # Apply charging effect
                if image.ndim == 2:
                    aug_image = aug_image * (1 + (intensity - 1) * spot_mask)
                else:
                    for c in range(image.shape[2]):
                        aug_image[:, :, c] = (
                            aug_image[:, :, c] * (1 + (intensity - 1) * spot_mask)
                        )

            # Clip and convert back
            aug_image = np.clip(aug_image, 0, 255).astype(image.dtype)

            augmented_samples.append((aug_image, mask))

        return DatasetInterface.from_pairs(
            augmented_samples,
            metadata={**dataset.metadata, "augmentation": "charging_artifact"},
        )


class ScanLineNoiseAugmentator(DataAugmentatorInterface):
    """
    Add scan line artifacts typical of SEM imaging.

    Simulates beam drift or electromagnetic interference during scanning.
    Only affects the image.
    """

    def __init__(
        self,
        probability: float = 0.3,
        intensity_range: Tuple[float, float] = (0.02, 0.08),
        direction: str = "horizontal",
        random_seed: int | None = None,
    ) -> None:
        """
        Initialize the scan line noise augmentator.

        Args:
            probability: Probability of adding noise to each line.
            intensity_range: Range of noise intensity (fraction of max value).
            direction: Direction of scan lines ("horizontal" or "vertical").
            random_seed: Random seed for reproducibility.

        """
        self.probability = probability
        self.intensity_range = intensity_range
        self.direction = direction
        self.random_seed = random_seed

    def augment(self, dataset: DatasetInterface) -> DatasetInterface:
        """Add scan line noise to all images."""
        rng = random.Random(self.random_seed)
        np_rng = np.random.default_rng(self.random_seed)

        augmented_samples = []

        for image, mask in dataset.samples:
            aug_image = image.astype(np.float32).copy()

            # Random intensity for this image
            intensity = rng.uniform(*self.intensity_range) * 255

            if self.direction == "horizontal":
                # Add horizontal scan lines
                for i in range(aug_image.shape[0]):
                    if rng.random() < self.probability:
                        noise = np_rng.normal(0, intensity, aug_image.shape[1])
                        if image.ndim == 2:
                            aug_image[i, :] += noise
                        else:
                            for c in range(image.shape[2]):
                                aug_image[i, :, c] += noise
            else:
                # Add vertical scan lines
                for j in range(aug_image.shape[1]):
                    if rng.random() < self.probability:
                        noise = np_rng.normal(0, intensity, aug_image.shape[0])
                        if image.ndim == 2:
                            aug_image[:, j] += noise
                        else:
                            for c in range(image.shape[2]):
                                aug_image[:, j, c] += noise

            # Clip and convert back
            aug_image = np.clip(aug_image, 0, 255).astype(image.dtype)

            augmented_samples.append((aug_image, mask))

        return DatasetInterface.from_pairs(
            augmented_samples,
            metadata={**dataset.metadata, "augmentation": "scan_line_noise"},
        )
