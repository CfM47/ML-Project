"""Test flip augmentators to ensure they produce contiguous arrays."""

import numpy as np

from auto_ml.implementations.augmentators.geometric import (
    HorizontalFlipAugmentator,
    VerticalFlipAugmentator,
)
from auto_ml.interfaces import SegmentationDatasetInterface


def test_horizontal_flip_produces_contiguous_arrays() -> None:
    """Test that HorizontalFlipAugmentator produces C-contiguous arrays."""
    # Create a simple dataset (512x512 as required by interface)
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
    dataset = SegmentationDatasetInterface()
    dataset.add_sample(image, mask)
    # Apply horizontal flip
    augmentator = HorizontalFlipAugmentator()
    augmented = augmentator.augment(dataset)
    # Get augmented samples
    aug_image, aug_mask = augmented.samples[0]
    # Check that arrays are C-contiguous
    assert aug_image.flags['C_CONTIGUOUS'], "Augmented image is not C-contiguous"
    assert aug_mask.flags['C_CONTIGUOUS'], "Augmented mask is not C-contiguous"
    # Check that strides are all positive
    assert all(s >= 0 for s in aug_image.strides), "Image has negative strides"
    assert all(s >= 0 for s in aug_mask.strides), "Mask has negative strides"
    # Verify flip was actually applied (compare to manual flip)
    expected_image = np.ascontiguousarray(np.fliplr(image))
    expected_mask = np.ascontiguousarray(np.fliplr(mask))
    assert np.array_equal(aug_image, expected_image), "Image flip incorrect"
    assert np.array_equal(aug_mask, expected_mask), "Mask flip incorrect"
    print(
        "✓ HorizontalFlipAugmentator produces contiguous arrays with positive strides",
    )


def test_vertical_flip_produces_contiguous_arrays() -> None:
    """Test that VerticalFlipAugmentator produces C-contiguous arrays."""
    # Create a simple dataset (512x512 as required by interface)
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
    dataset = SegmentationDatasetInterface()
    dataset.add_sample(image, mask)
    # Apply vertical flip
    augmentator = VerticalFlipAugmentator()
    augmented = augmentator.augment(dataset)
    # Get augmented samples
    aug_image, aug_mask = augmented.samples[0]
    # Check that arrays are C-contiguous
    assert aug_image.flags['C_CONTIGUOUS'], "Augmented image is not C-contiguous"
    assert aug_mask.flags['C_CONTIGUOUS'], "Augmented mask is not C-contiguous"
    # Check that strides are all positive
    assert all(s >= 0 for s in aug_image.strides), "Image has negative strides"
    assert all(s >= 0 for s in aug_mask.strides), "Mask has negative strides"
    # Verify flip was actually applied (compare to manual flip)
    expected_image = np.ascontiguousarray(np.flipud(image))
    expected_mask = np.ascontiguousarray(np.flipud(mask))
    assert np.array_equal(aug_image, expected_image), "Image flip incorrect"
    assert np.array_equal(aug_mask, expected_mask), "Mask flip incorrect"
    print("✓ VerticalFlipAugmentator produces contiguous arrays with positive strides")


def test_flips_with_multiple_samples() -> None:
    """Test flip augmentators with multiple samples."""
    # Create dataset with multiple samples (512x512 as required)
    dataset = SegmentationDatasetInterface()
    for _ in range(10):
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
        dataset.add_sample(image, mask)
    # Test horizontal flip
    h_augmentator = HorizontalFlipAugmentator()
    h_augmented = h_augmentator.augment(dataset)
    for aug_image, aug_mask in h_augmented.samples:
        assert aug_image.flags['C_CONTIGUOUS'], "H-flip: Image not C-contiguous"
        assert aug_mask.flags['C_CONTIGUOUS'], "H-flip: Mask not C-contiguous"
        assert all(
            s >= 0 for s in aug_image.strides
        ), "H-flip: Image has negative strides"
        assert all(
            s >= 0 for s in aug_mask.strides
        ), "H-flip: Mask has negative strides"
    # Test vertical flip
    v_augmentator = VerticalFlipAugmentator()
    v_augmented = v_augmentator.augment(dataset)
    for aug_image, aug_mask in v_augmented.samples:
        assert aug_image.flags['C_CONTIGUOUS'], "V-flip: Image not C-contiguous"
        assert aug_mask.flags['C_CONTIGUOUS'], "V-flip: Mask not C-contiguous"
        assert all(
            s >= 0 for s in aug_image.strides
        ), "V-flip: Image has negative strides"
        assert all(
            s >= 0 for s in aug_mask.strides
        ), "V-flip: Mask has negative strides"
    print("✓ Both flip augmentators work correctly with multiple samples")


def test_flip_preserves_dtype() -> None:
    """Test that flip augmentators preserve data types."""
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
    dataset = SegmentationDatasetInterface()
    dataset.add_sample(image, mask)
    # Test horizontal flip
    h_aug = HorizontalFlipAugmentator()
    h_result = h_aug.augment(dataset)
    h_img, h_mask = h_result.samples[0]
    assert h_img.dtype == np.uint8, f"H-flip: Image dtype changed to {h_img.dtype}"
    assert h_mask.dtype == np.uint8, f"H-flip: Mask dtype changed to {h_mask.dtype}"
    # Test vertical flip
    v_aug = VerticalFlipAugmentator()
    v_result = v_aug.augment(dataset)
    v_img, v_mask = v_result.samples[0]
    assert v_img.dtype == np.uint8, f"V-flip: Image dtype changed to {v_img.dtype}"
    assert v_mask.dtype == np.uint8, f"V-flip: Mask dtype changed to {v_mask.dtype}"
    print("✓ Flip augmentators preserve data types")


if __name__ == "__main__":
    test_horizontal_flip_produces_contiguous_arrays()
    test_vertical_flip_produces_contiguous_arrays()
    test_flips_with_multiple_samples()
    test_flip_preserves_dtype()
    print("\n✅ All flip augmentator tests passed!")
