"""Geometric augmentation implementations."""

import random
from typing import Tuple

import numpy as np

from auto_ml.implementations.augmentators.base import (
    apply_affine_to_image,
    apply_affine_to_mask,
    rotation_matrix,
    scale_matrix,
    translation_matrix,
)
from auto_ml.interfaces import DataAugmentatorInterface, DatasetInterface


class RotationAugmentator(DataAugmentatorInterface):
    """
    Rotate images and masks by a specified angle.

    Applies the same rotation to both image and mask, preserving alignment.
    """

    def __init__(
        self,
        angle_range: Tuple[float, float] = (-15.0, 15.0),
        random_seed: int | None = None,
    ) -> None:
        """
        Initialize the rotation augmentator.

        Args:
            angle_range: Range of rotation angles in degrees (min, max).
                        Positive = counter-clockwise.
            random_seed: Random seed for reproducibility.

        """
        self.angle_range = angle_range
        self.random_seed = random_seed

    def augment(self, dataset: DatasetInterface) -> DatasetInterface:
        """Apply random rotation to all samples in the dataset."""
        rng = random.Random(self.random_seed)
        augmented_samples = []

        for image, mask in dataset.samples:
            # Random angle within range
            angle = rng.uniform(*self.angle_range)

            # Get image center
            h, w = image.shape[:2]
            center = (w / 2, h / 2)

            # Create rotation matrix
            matrix = rotation_matrix(angle, center)

            # Apply to image and mask
            aug_image = apply_affine_to_image(image, matrix, order=1)
            aug_mask = apply_affine_to_mask(mask, matrix)

            augmented_samples.append((aug_image, aug_mask))

        return DatasetInterface.from_pairs(
            augmented_samples,
            metadata={**dataset.metadata, "augmentation": "rotation"},
        )


class HorizontalFlipAugmentator(DataAugmentatorInterface):
    """Flip images and masks horizontally."""

    def augment(self, dataset: DatasetInterface) -> DatasetInterface:
        """Apply horizontal flip to all samples."""
        augmented_samples = []

        for image, mask in dataset.samples:
            aug_image = np.fliplr(image)
            aug_mask = np.fliplr(mask)
            augmented_samples.append((aug_image, aug_mask))

        return DatasetInterface.from_pairs(
            augmented_samples,
            metadata={**dataset.metadata, "augmentation": "horizontal_flip"},
        )


class VerticalFlipAugmentator(DataAugmentatorInterface):
    """Flip images and masks vertically."""

    def augment(self, dataset: DatasetInterface) -> DatasetInterface:
        """Apply vertical flip to all samples."""
        augmented_samples = []

        for image, mask in dataset.samples:
            aug_image = np.flipud(image)
            aug_mask = np.flipud(mask)
            augmented_samples.append((aug_image, aug_mask))

        return DatasetInterface.from_pairs(
            augmented_samples,
            metadata={**dataset.metadata, "augmentation": "vertical_flip"},
        )


class ScaleAugmentator(DataAugmentatorInterface):
    """
    Scale images and masks.

    Applies zoom in/out while maintaining image size through cropping or padding.
    """

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        random_seed: int | None = None,
    ) -> None:
        """
        Initialize the scale augmentator.

        Args:
            scale_range: Range of scale factors (min, max).
                        Values < 1.0 zoom out, > 1.0 zoom in.
            random_seed: Random seed for reproducibility.

        """
        self.scale_range = scale_range
        self.random_seed = random_seed

    def augment(self, dataset: DatasetInterface) -> DatasetInterface:
        """Apply random scaling to all samples."""
        rng = random.Random(self.random_seed)
        augmented_samples = []

        for image, mask in dataset.samples:
            # Random scale factor
            scale = rng.uniform(*self.scale_range)

            # Get image center
            h, w = image.shape[:2]
            center = (w / 2, h / 2)

            # Create scale matrix
            matrix = scale_matrix(scale, scale, center)

            # Apply to image and mask
            aug_image = apply_affine_to_image(image, matrix, order=1)
            aug_mask = apply_affine_to_mask(mask, matrix)

            augmented_samples.append((aug_image, aug_mask))

        return DatasetInterface.from_pairs(
            augmented_samples,
            metadata={**dataset.metadata, "augmentation": "scale"},
        )


class TranslationAugmentator(DataAugmentatorInterface):
    """
    Translate (shift) images and masks.

    Shifts the image content horizontally and/or vertically.
    """

    def __init__(
        self,
        translate_range: Tuple[float, float] = (-0.1, 0.1),
        random_seed: int | None = None,
    ) -> None:
        """
        Initialize the translation augmentator.

        Args:
            translate_range: Range of translation as fraction of image size.
                            E.g., (-0.1, 0.1) means shift by up to ±10%.
            random_seed: Random seed for reproducibility.

        """
        self.translate_range = translate_range
        self.random_seed = random_seed

    def augment(self, dataset: DatasetInterface) -> DatasetInterface:
        """Apply random translation to all samples."""
        rng = random.Random(self.random_seed)
        augmented_samples = []

        for image, mask in dataset.samples:
            h, w = image.shape[:2]

            # Random translation
            tx_frac = rng.uniform(*self.translate_range)
            ty_frac = rng.uniform(*self.translate_range)

            tx = tx_frac * w
            ty = ty_frac * h

            # Create translation matrix
            matrix = translation_matrix(tx, ty)

            # Apply to image and mask
            aug_image = apply_affine_to_image(image, matrix, order=1)
            aug_mask = apply_affine_to_mask(mask, matrix)

            augmented_samples.append((aug_image, aug_mask))

        return DatasetInterface.from_pairs(
            augmented_samples,
            metadata={**dataset.metadata, "augmentation": "translation"},
        )


class RandomCropAugmentator(DataAugmentatorInterface):
    """
    Random crop of images and masks.

    Extracts a random region from the input and resizes back to original size.
    """

    def __init__(
        self,
        crop_size_range: Tuple[float, float] = (0.8, 1.0),
        random_seed: int | None = None,
    ) -> None:
        """
        Initialize the random crop augmentator.

        Args:
            crop_size_range: Range of crop size as fraction of original
                            (min, max). E.g., (0.8, 1.0) crops 80-100%.
            random_seed: Random seed for reproducibility.

        """
        self.crop_size_range = crop_size_range
        self.random_seed = random_seed

    def augment(self, dataset: DatasetInterface) -> DatasetInterface:
        """Apply random crop to all samples."""
        rng = random.Random(self.random_seed)
        augmented_samples = []

        for image, mask in dataset.samples:
            h, w = image.shape[:2]

            # Random crop size
            crop_frac = rng.uniform(*self.crop_size_range)
            crop_h = int(h * crop_frac)
            crop_w = int(w * crop_frac)

            # Random crop position
            top = rng.randint(0, h - crop_h)
            left = rng.randint(0, w - crop_w)

            # Crop
            if image.ndim == 2:
                cropped_image = image[top:top + crop_h, left:left + crop_w]
            else:
                cropped_image = image[top:top + crop_h, left:left + crop_w, :]

            cropped_mask = mask[top:top + crop_h, left:left + crop_w]

            # Resize back to original size
            from scipy import ndimage

            zoom_h = h / crop_h
            zoom_w = w / crop_w

            if image.ndim == 2:
                aug_image = ndimage.zoom(
                    cropped_image,
                    (zoom_h, zoom_w),
                    order=1,
                )
            else:
                aug_image = ndimage.zoom(
                    cropped_image,
                    (zoom_h, zoom_w, 1),
                    order=1,
                )

            aug_mask = ndimage.zoom(
                cropped_mask,
                (zoom_h, zoom_w),
                order=0,
            )

            # Ensure correct types
            aug_image = aug_image.astype(image.dtype)
            aug_mask = aug_mask.astype(mask.dtype)

            augmented_samples.append((aug_image, aug_mask))

        return DatasetInterface.from_pairs(
            augmented_samples,
            metadata={**dataset.metadata, "augmentation": "random_crop"},
        )
