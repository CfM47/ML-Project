"""
Integration test for CNN classifier with Quadtree segmentation.

This test trains a CNN model with deterministic dummy data,
then uses the trained model with the QuadtreeSegmentationModel
to perform segmentation and verify expected results.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from auto_ml.implementations import QuadtreeSegmentationModel
from auto_ml.implementations.classifiers.cnn import CNNModel
from auto_ml.interfaces import DatasetInterface


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
    image_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create deterministic training data for binary classification (2 classes).

    Class 0: Dark images (low intensity 0-80)
    Class 1: Bright images (high intensity 180-255)

    This simpler binary classification is more reliable for testing.

    Args:
        num_samples_per_class: Number of samples per class.
        image_size: Size of the square images.

    Returns:
        Tuple of (images tensor, labels tensor).

    """
    images = []
    labels = []

    # Class 0: Dark images (intensity range 0-80)
    for i in range(num_samples_per_class):
        # Use different base intensities for variety
        base_intensity = 20 + (i % 10) * 5  # 20, 25, 30, ..., 65
        img = np.full((image_size, image_size), base_intensity, dtype=np.float32)
        # Add small structured pattern
        x, y = np.meshgrid(np.arange(image_size), np.arange(image_size))
        pattern = 10 * np.sin((x + y) * 0.2 + i * 0.1)
        img = img + pattern
        img = np.clip(img, 0, 80).astype(np.float32) / 255.0
        images.append(img)
        labels.append(0)

    # Class 1: Bright images (intensity range 180-255)
    for i in range(num_samples_per_class):
        base_intensity = 200 + (i % 10) * 5  # 200, 205, 210, ..., 245
        img = np.full((image_size, image_size), base_intensity, dtype=np.float32)
        # Add different pattern
        x, y = np.meshgrid(np.arange(image_size), np.arange(image_size))
        pattern = 10 * np.cos((x - y) * 0.2 + i * 0.1)
        img = img + pattern
        img = np.clip(img, 180, 255).astype(np.float32) / 255.0
        images.append(img)
        labels.append(1)

    # Shuffle deterministically
    rng = np.random.RandomState(42)
    indices = np.arange(len(images))
    rng.shuffle(indices)
    images = [images[i] for i in indices]
    labels = [labels[i] for i in indices]

    # Convert to tensors: (N, 1, H, W)
    images_tensor = torch.tensor(np.array(images), dtype=torch.float32).unsqueeze(1)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return images_tensor, labels_tensor


def train_cnn_model(
    model: CNNModel,
    images: torch.Tensor,
    labels: torch.Tensor,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 0.01,
) -> float:
    """
    Train the CNN model on the provided data.

    Args:
        model: The CNNModel to train.
        images: Training images tensor.
        labels: Training labels tensor.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate.

    Returns:
        Final training loss.

    """
    # Set model to training mode
    model.model.train()

    # Create data loader
    dataset = TensorDataset(images.to(model.device), labels.to(model.device))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.model.parameters(), lr=learning_rate)

    final_loss = 0.0

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_images, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model.model(batch_images, return_logits=True)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_labels).sum().item()
            total += batch_labels.size(0)

        final_loss = epoch_loss / total

    # Set model back to eval mode
    model.model.eval()

    return final_loss


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
    images, labels = create_binary_training_data(num_samples_per_class=50, image_size=32)

    # Train the model
    final_loss = train_cnn_model(
        model=cnn_model,
        images=images,
        labels=labels,
        epochs=20,
        batch_size=16,
        learning_rate=0.005,
    )

    print(f"Final training loss: {final_loss:.4f}")
    assert final_loss < 0.5, f"Training loss should decrease, got {final_loss}"

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
    assert label_bright == 1, f"Expected bright region to be class 1, got {label_bright}"

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

    images, labels = create_binary_training_data(num_samples_per_class=50, image_size=32)
    final_loss = train_cnn_model(
        model=cnn_model,
        images=images,
        labels=labels,
        epochs=20,
        batch_size=16,
        learning_rate=0.005,
    )
    print(f"Training complete. Final loss: {final_loss:.4f}")

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
        threshold=1.0,  # Force subdivision (no confidence can be >= 1.0)
        min_region_size=256,  # Stop at quadrant level (512/2 = 256)
        max_depth=None,  # No depth limit
    )

    # 4. Create dataset and evaluate
    print("Step 4: Running segmentation...")
    dummy_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    dataset = DatasetInterface()
    dataset.add_sample(test_image, dummy_mask)

    mask_pairs = quadtree_model.evaluate(dataset)

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
    assert bottom_left_class == 1, f"Bottom-left should be class 1, got {bottom_left_class}"
    assert bottom_right_class == 0, f"Bottom-right should be class 0, got {bottom_right_class}"

    # Verify full mask matches expected
    assert np.array_equal(predicted_mask, expected_mask), "Predicted mask does not match expected mask"

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

    images, labels = create_binary_training_data(num_samples_per_class=50, image_size=32)
    train_cnn_model(
        model=cnn_model,
        images=images,
        labels=labels,
        epochs=20,
        batch_size=16,
        learning_rate=0.005,
    )

    # Create test image with checkerboard pattern at 128x128 level
    image_size = 512
    test_image = np.zeros((image_size, image_size), dtype=np.uint8)

    # Create a 4x4 grid of alternating dark (class 0) and bright (class 1) regions
    block_size = 128
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                test_image[i * block_size : (i + 1) * block_size,
                          j * block_size : (j + 1) * block_size] = 40  # Dark
            else:
                test_image[i * block_size : (i + 1) * block_size,
                          j * block_size : (j + 1) * block_size] = 220  # Bright

    # Create QuadtreeSegmentationModel with smaller min_region_size
    # This forces deeper recursion to classify the checkerboard pattern
    quadtree_model = QuadtreeSegmentationModel(
        classifier=cnn_model,
        threshold=0.99,  # High threshold to force recursion
        min_region_size=128,  # Stop at 128x128 level
        max_depth=2,  # Allow 2 levels of recursion (512 -> 256 -> 128)
    )

    # Evaluate
    dummy_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    dataset = DatasetInterface()
    dataset.add_sample(test_image, dummy_mask)

    mask_pairs = quadtree_model.evaluate(dataset)
    predicted_mask, _ = mask_pairs[0]

    # Build expected checkerboard mask
    expected_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                expected_mask[i * block_size : (i + 1) * block_size,
                             j * block_size : (j + 1) * block_size] = 0  # Dark -> class 0
            else:
                expected_mask[i * block_size : (i + 1) * block_size,
                             j * block_size : (j + 1) * block_size] = 1  # Bright -> class 1

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
