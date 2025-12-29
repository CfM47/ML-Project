"""Test SEM-specific augmentations."""

import numpy as np

from auto_ml.implementations import (
    AdaptiveHistogramEqualizationAugmentator,
    ChargingArtifactAugmentator,
    ElasticDeformationAugmentator,
    ScanLineNoiseAugmentator,
)


class SimpleDataset:
    """Simple dataset for testing."""

    def __init__(self, samples: list[tuple[np.ndarray, np.ndarray]]) -> None:
        """Initialize dataset with samples."""
        self.samples = samples
        self.metadata: dict = {}


def test_elastic_deformation_augmentator() -> None:
    """Test elastic deformation augmentator."""
    # Create sample image and mask
    image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
    dataset = SimpleDataset([(image, mask)])

    # Apply augmentation
    augmentator = ElasticDeformationAugmentator(alpha=50.0, sigma=5.0, random_seed=42)
    augmented = augmentator.augment(dataset)

    # Check that we still have one sample
    assert len(augmented.samples) == 1

    # Check shapes are preserved
    aug_image, aug_mask = augmented.samples[0]
    assert aug_image.shape == image.shape
    assert aug_mask.shape == mask.shape


def test_adaptive_histogram_equalization_augmentator() -> None:
    """Test adaptive histogram equalization augmentator."""
    # Create sample image and mask
    image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
    dataset = SimpleDataset([(image, mask)])

    # Apply augmentation
    augmentator = AdaptiveHistogramEqualizationAugmentator(
        clip_limit=2.0,
        tile_grid_size=(8, 8),
    )
    augmented = augmentator.augment(dataset)

    # Check that we still have one sample
    assert len(augmented.samples) == 1

    # Check shapes are preserved
    aug_image, aug_mask = augmented.samples[0]
    assert aug_image.shape == image.shape
    assert aug_mask.shape == mask.shape

    # Mask should be unchanged
    np.testing.assert_array_equal(aug_mask, mask)


def test_charging_artifact_augmentator() -> None:
    """Test charging artifact augmentator."""
    # Create sample image and mask
    image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
    dataset = SimpleDataset([(image, mask)])

    # Apply augmentation
    augmentator = ChargingArtifactAugmentator(
        num_spots=(2, 5),
        spot_size_range=(5, 15),
        intensity_range=(0.3, 0.7),
        random_seed=42,
    )
    augmented = augmentator.augment(dataset)

    # Check that we still have one sample
    assert len(augmented.samples) == 1

    # Check shapes are preserved
    aug_image, aug_mask = augmented.samples[0]
    assert aug_image.shape == image.shape
    assert aug_mask.shape == mask.shape

    # Mask should be unchanged
    np.testing.assert_array_equal(aug_mask, mask)


def test_scan_line_noise_augmentator() -> None:
    """Test scan line noise augmentator."""
    # Create sample image and mask
    image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
    dataset = SimpleDataset([(image, mask)])

    # Apply augmentation
    augmentator = ScanLineNoiseAugmentator(
        probability=0.5,
        intensity_range=(0.02, 0.05),
        direction="horizontal",
        random_seed=42,
    )
    augmented = augmentator.augment(dataset)

    # Check that we still have one sample
    assert len(augmented.samples) == 1

    # Check shapes are preserved
    aug_image, aug_mask = augmented.samples[0]
    assert aug_image.shape == image.shape
    assert aug_mask.shape == mask.shape

    # Mask should be unchanged
    np.testing.assert_array_equal(aug_mask, mask)


def test_sem_augmentations_preserve_mask() -> None:
    """Test that all SEM augmentations preserve masks."""
    # Create sample image and mask
    image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
    dataset = SimpleDataset([(image, mask)])

    augmentators = [
        ElasticDeformationAugmentator(alpha=50.0, sigma=5.0, random_seed=42),
        AdaptiveHistogramEqualizationAugmentator(clip_limit=2.0, tile_grid_size=(8, 8)),
        ChargingArtifactAugmentator(
            num_spots=(2, 5),
            spot_size_range=(5, 15),
            intensity_range=(0.3, 0.7),
            random_seed=42,
        ),
        ScanLineNoiseAugmentator(
            probability=0.5,
            intensity_range=(0.02, 0.05),
            direction="horizontal",
            random_seed=42,
        ),
    ]

    for augmentator in augmentators:
        augmented = augmentator.augment(dataset)
        aug_image, aug_mask = augmented.samples[0]

        # Check shapes are preserved
        assert aug_image.shape == image.shape, (
            f"Image shape changed by {type(augmentator).__name__}"
        )
        assert aug_mask.shape == mask.shape, (
            f"Mask shape changed by {type(augmentator).__name__}"
        )

        # For all augmentations except elastic deformation, mask should be unchanged
        if not isinstance(augmentator, ElasticDeformationAugmentator):
            np.testing.assert_array_equal(
                aug_mask,
                mask,
                err_msg=f"Mask changed by {type(augmentator).__name__}",
            )
