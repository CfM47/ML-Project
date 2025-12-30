"""
Integration test for CNN classifier with Quadtree segmentation.

This test trains a CNN model with deterministic dummy data,
then uses the trained model with the QuadtreeSegmentationModel
to perform segmentation and verify expected results.
"""

import numpy as np
import torch

from auto_ml.implementations import QuadtreeSegmentationModel
from auto_ml.implementations.classifiers.cnn import CNNModel
from auto_ml.interfaces import (
    ClassificationDatasetInterface,
    SegmentationDatasetInterface,
)


def set_deterministic_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make PyTorch operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_binary_training_data(
    num_samples_per_class: int = 50,
    min_size: int = 32,
    max_size: int = 256,
) -> tuple[list[np.ndarray], list[int]]:
    """
    Create deterministic training data for binary classification (2 classes).

    Class 0: Dark images (low intensity 0-80)
    Class 1: Bright images (high intensity 180-255)

    Generates images of varying sizes to help the model generalize across scales.

    Args:
        num_samples_per_class: Number of samples per class.
        min_size: Minimum image size.
        max_size: Maximum image size.

    Returns:
        Tuple of (images list, labels list).

    """
    images = []
    labels = []

    rng = np.random.RandomState(42)

    # Generate varied sizes between min_size and max_size
    sizes = [32, 64, 128, 256]

    # Class 0: Dark images (intensity range 0-80)
    for i in range(num_samples_per_class):
        size = sizes[i % len(sizes)]
        # Use different base intensities for variety
        base_intensity = 20 + (i % 10) * 5  # 20, 25, 30, ..., 65
        img = np.full((size, size), base_intensity, dtype=np.uint8)
        # Add small structured pattern
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        pattern = (10 * np.sin((x + y) * 0.2 / size * 32 + i * 0.1)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + pattern, 0, 80).astype(np.uint8)
        images.append(img)
        labels.append(0)

    # Class 1: Bright images (intensity range 180-255)
    for i in range(num_samples_per_class):
        size = sizes[i % len(sizes)]
        base_intensity = 200 + (i % 10) * 5  # 200, 205, 210, ..., 245
        img = np.full((size, size), base_intensity, dtype=np.uint8)
        # Add different pattern
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        pattern = (10 * np.cos((x - y) * 0.2 / size * 32 + i * 0.1)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + pattern, 180, 255).astype(np.uint8)
        images.append(img)
        labels.append(1)

    # Shuffle deterministically
    indices = rng.permutation(len(labels))
    images = [images[i] for i in indices]
    labels = [labels[i] for i in indices]

    return images, labels


def test_cnn_binary_classifier_training() -> None:
    """Test that the CNN classifier can be trained for binary classification."""
    print("Testing CNN binary classifier training with deterministic data...")

    set_deterministic_seed(42)

    # Create CNN model with 2 classes for binary classification
    cnn_model = CNNModel(
        num_classes=2,
        channels=1,
        base_filters=16,
        dropout=0.1,
        device="cpu",
    )

    # Create binary training data
    images, labels = create_binary_training_data(
        num_samples_per_class=50,
        image_size=32,
    )

    # Create dataset using the interface
    dataset = ClassificationDatasetInterface()
    for img, lbl in zip(images, labels):
        dataset.add_sample(img, lbl)

    # Train the model using the interface method
    metrics = cnn_model.train(
        dataset=dataset,
    )

    print(f"Final training loss: {metrics.loss:.4f}")
    print(f"Final training accuracy: {metrics.accuracy:.4f}")
    assert metrics.loss < 0.5, f"Training loss should decrease, got {metrics.loss}"

    # Test classification on representative samples
    # Dark region (class 0) - use intensity ~40
    dark_region = np.full((32, 32), 40, dtype=np.uint8)
    label_dark, conf_dark = cnn_model.classify(dark_region, 0, 0, 32, 32)
    print(f"Dark region: class={label_dark}, confidence={conf_dark:.3f}")

    # Bright region (class 1) - use intensity ~220
    bright_region = np.full((32, 32), 220, dtype=np.uint8)
    label_bright, conf_bright = cnn_model.classify(bright_region, 0, 0, 32, 32)
    print(f"Bright region: class={label_bright}, confidence={conf_bright:.3f}")

    # Verify that the model learned to distinguish the classes
    assert label_dark == 0, f"Expected dark region to be class 0, got {label_dark}"
    assert label_bright == 1, (
        f"Expected bright region to be class 1, got {label_bright}"
    )

    print("CNN binary classifier training test PASSED!")


def test_cnn_quadtree_binary_integration() -> None:
    """
    Integration test: Train binary CNN, use with QuadtreeSegmentationModel.

    This test:
    1. Trains a binary CNN classifier on dummy data (dark vs bright)
    2. Creates a test image with 4 quadrants of different intensities
    3. Uses QuadtreeSegmentationModel with the trained CNN to segment the image
    4. Verifies the segmentation produces expected class labels
    """
    print("Testing CNN + Quadtree segmentation integration (binary)...")

    set_deterministic_seed(42)

    # 1. Create and train CNN model for binary classification
    print("Step 1: Creating and training binary CNN model...")
    cnn_model = CNNModel(
        num_classes=2,
        channels=1,
        base_filters=16,
        dropout=0.1,
        device="cpu",
    )

    images, labels = create_binary_training_data(
        num_samples_per_class=50,
    )
    dataset = ClassificationDatasetInterface()
    for img, lbl in zip(images, labels):
        dataset.add_sample(img, lbl)
    metrics = cnn_model.train(
        dataset=dataset,
    )
    print(f"Training complete. Final loss: {metrics.loss:.4f}")

    # 2. Create test image with 4 quadrants (alternating dark and bright)
    print("Step 2: Creating test image with 4 quadrants...")
    image_size = 512
    test_image = np.zeros((image_size, image_size), dtype=np.uint8)

    # Top-left quadrant: Dark (class 0) - intensity ~40
    test_image[0:256, 0:256] = 40
    # Top-right quadrant: Bright (class 1) - intensity ~220
    test_image[0:256, 256:512] = 220
    # Bottom-left quadrant: Bright (class 1) - intensity ~220
    test_image[256:512, 0:256] = 220
    # Bottom-right quadrant: Dark (class 0) - intensity ~40
    test_image[256:512, 256:512] = 40

    # 3. Create QuadtreeSegmentationModel with the trained CNN
    # Set threshold very high to force subdivision until min_region_size is reached
    print("Step 3: Creating QuadtreeSegmentationModel...")
    quadtree_model = QuadtreeSegmentationModel(
        classifier=cnn_model,
        classifier_dataset_dir=None,
        threshold=1.0,  # Force subdivision (no confidence can be >= 1.0)
        min_region_size=256,  # Stop at quadrant level (512/2 = 256)
        max_depth=None,  # No depth limit
    )

    # 4. Create dataset and evaluate
    print("Step 4: Running segmentation...")
    dummy_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    seg_dataset = SegmentationDatasetInterface()
    seg_dataset.add_sample(test_image, dummy_mask)

    mask_pairs = quadtree_model.evaluate(seg_dataset)

    assert len(mask_pairs) == 1, f"Expected 1 mask pair, got {len(mask_pairs)}"
    predicted_mask, _ = mask_pairs[0]

    # 5. Verify segmentation results
    print("Step 5: Verifying segmentation results...")

    # Check each quadrant
    top_left_class = int(np.median(predicted_mask[0:256, 0:256]))
    top_right_class = int(np.median(predicted_mask[0:256, 256:512]))
    bottom_left_class = int(np.median(predicted_mask[256:512, 0:256]))
    bottom_right_class = int(np.median(predicted_mask[256:512, 256:512]))

    print(f"Top-left quadrant (dark): class {top_left_class}")
    print(f"Top-right quadrant (bright): class {top_right_class}")
    print(f"Bottom-left quadrant (bright): class {bottom_left_class}")
    print(f"Bottom-right quadrant (dark): class {bottom_right_class}")

    # Build expected mask (checkerboard pattern)
    expected_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    expected_mask[0:256, 0:256] = 0  # Dark -> class 0
    expected_mask[0:256, 256:512] = 1  # Bright -> class 1
    expected_mask[256:512, 0:256] = 1  # Bright -> class 1
    expected_mask[256:512, 256:512] = 0  # Dark -> class 0

    # Verify classifications
    assert top_left_class == 0, f"Top-left should be class 0, got {top_left_class}"
    assert top_right_class == 1, f"Top-right should be class 1, got {top_right_class}"
    assert bottom_left_class == 1, (
        f"Bottom-left should be class 1, got {bottom_left_class}"
    )
    assert bottom_right_class == 0, (
        f"Bottom-right should be class 0, got {bottom_right_class}"
    )

    # Verify full mask matches expected
    assert np.array_equal(predicted_mask, expected_mask), (
        "Predicted mask does not match expected mask"
    )

    print("CNN + Quadtree segmentation integration test PASSED!")


def test_cnn_quadtree_deeper_recursion() -> None:
    """
    Test Quadtree with deeper recursion using trained CNN.

    This test verifies that the quadtree can recursively subdivide
    regions when needed based on the classifier's confidence.
    """
    print("Testing CNN + Quadtree with deeper recursion...")

    set_deterministic_seed(42)

    # Create and train CNN model for binary classification
    cnn_model = CNNModel(
        num_classes=2,
        channels=1,
        base_filters=16,
        dropout=0.1,
        device="cpu",
    )

    images, labels = create_binary_training_data(
        num_samples_per_class=50,
    )
    dataset = ClassificationDatasetInterface()
    for img, lbl in zip(images, labels):
        dataset.add_sample(img, lbl)
    cnn_model.train(dataset=dataset)

    # Create test image with checkerboard pattern at 128x128 level
    image_size = 512
    test_image = np.zeros((image_size, image_size), dtype=np.uint8)

    # Create a 4x4 grid of alternating dark (class 0) and bright (class 1) regions
    block_size = 128
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                test_image[
                    i * block_size : (i + 1) * block_size,
                    j * block_size : (j + 1) * block_size,
                ] = 40  # Dark
            else:
                test_image[
                    i * block_size : (i + 1) * block_size,
                    j * block_size : (j + 1) * block_size,
                ] = 220  # Bright

    # Create QuadtreeSegmentationModel with smaller min_region_size
    # This forces deeper recursion to classify the checkerboard pattern
    quadtree_model = QuadtreeSegmentationModel(
        classifier=cnn_model,
        classifier_dataset_dir=None,
        threshold=0.99,  # High threshold to force recursion
        min_region_size=128,  # Stop at 128x128 level
        max_depth=2,  # Allow 2 levels of recursion (512 -> 256 -> 128)
    )

    # Evaluate
    dummy_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    seg_dataset = SegmentationDatasetInterface()
    seg_dataset.add_sample(test_image, dummy_mask)

    mask_pairs = quadtree_model.evaluate(seg_dataset)
    predicted_mask, _ = mask_pairs[0]

    # Build expected checkerboard mask
    expected_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                expected_mask[
                    i * block_size : (i + 1) * block_size,
                    j * block_size : (j + 1) * block_size,
                ] = 0  # Dark -> class 0
            else:
                expected_mask[
                    i * block_size : (i + 1) * block_size,
                    j * block_size : (j + 1) * block_size,
                ] = 1  # Bright -> class 1

    # Verify at least the pattern type is correct (checkerboard)
    # Check a few specific blocks
    assert predicted_mask[64, 64] == 0, "Block (0,0) should be class 0"
    assert predicted_mask[64, 192] == 1, "Block (0,1) should be class 1"
    assert predicted_mask[192, 64] == 1, "Block (1,0) should be class 1"
    assert predicted_mask[192, 192] == 0, "Block (1,1) should be class 0"

    print("CNN + Quadtree deeper recursion test PASSED!")


if __name__ == "__main__":
    print("=" * 60)
    print("Running CNN + Quadtree Integration Tests")
    print("=" * 60)

    test_cnn_binary_classifier_training()
    print()

    test_cnn_quadtree_binary_integration()
    print()

    test_cnn_quadtree_deeper_recursion()
    print()

    print("=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
