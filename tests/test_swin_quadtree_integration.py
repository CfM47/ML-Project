"""Integration test for Swin classifier with Quadtree segmentation."""

import numpy as np
import torch

from auto_ml.implementations import QuadtreeSegmentationModel
from auto_ml.implementations.classifiers.swin import SwinModel
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_binary_training_data(
    num_samples_per_class: int = 50,
    size: int = 64,
) -> tuple[list[np.ndarray], list[int]]:
    """
    Create deterministic training data for binary classification.

    Class 0: Dark images
    Class 1: Bright images
    """
    images = []
    labels = []
    rng = np.random.RandomState(42)

    for i in range(num_samples_per_class):
        base_intensity_dark = 20 + (i % 10) * 5
        img_dark = np.full((size, size), base_intensity_dark, dtype=np.uint8)
        images.append(img_dark)
        labels.append(0)

        base_intensity_bright = 200 + (i % 10) * 5
        img_bright = np.full((size, size), base_intensity_bright, dtype=np.uint8)
        images.append(img_bright)
        labels.append(1)

    indices = rng.permutation(len(labels))
    images = [images[i] for i in indices]
    labels = [labels[i] for i in indices]

    return images, labels


def test_swin_binary_classifier_training() -> None:
    """Test that the Swin classifier can be trained for binary classification."""
    print("Testing Swin binary classifier training...")
    set_deterministic_seed(42)

    swin_model = SwinModel(
        num_classes=2,
        channels=1,
        image_size=64,
        patch_size=4,
        window_size=4,
        embed_dim=48,
        depths=[2, 4],
        num_heads=[3, 6],
        dropout=0.1,
        device="cpu",
        train_epochs=5,
    )

    images, labels = create_binary_training_data(num_samples_per_class=25, size=64)
    dataset = ClassificationDatasetInterface.from_pairs(list(zip(images, labels)))

    metrics = swin_model.train(dataset=dataset)
    print(f"Final training loss: {metrics.loss:.4f}")
    print(f"Final training accuracy: {metrics.accuracy:.4f}")
    assert metrics.loss < 0.5

    dark_region = np.full((64, 64), 40, dtype=np.uint8)
    label_dark, conf_dark = swin_model.classify(dark_region, 0, 0, 64, 64)
    print(f"Dark region: class={label_dark}, confidence={conf_dark:.3f}")

    bright_region = np.full((64, 64), 220, dtype=np.uint8)
    label_bright, conf_bright = swin_model.classify(bright_region, 0, 0, 64, 64)
    print(f"Bright region: class={label_bright}, confidence={conf_bright:.3f}")

    assert label_dark == 0
    assert label_bright == 1
    print("Swin binary classifier training test PASSED!")


def test_swin_quadtree_binary_integration() -> None:
    """Integration test: Train binary Swin, use with QuadtreeSegmentationModel."""
    print("Testing Swin + Quadtree segmentation integration (binary)...")
    set_deterministic_seed(42)

    print("Step 1: Creating and training binary Swin model...")
    swin_model = SwinModel(
        num_classes=2,
        channels=1,
        image_size=64,
        patch_size=4,
        window_size=4,
        embed_dim=48,
        depths=[2, 4],
        num_heads=[3, 6],
        dropout=0.1,
        device="cpu",
        train_epochs=5,
    )

    images, labels = create_binary_training_data(num_samples_per_class=25, size=64)
    dataset = ClassificationDatasetInterface.from_pairs(list(zip(images, labels)))
    metrics = swin_model.train(dataset=dataset)
    print(f"Training complete. Final loss: {metrics.loss:.4f}")

    print("Step 2: Creating test image with 4 quadrants...")
    image_size = 512
    test_image = np.zeros((image_size, image_size), dtype=np.uint8)
    test_image[0:256, 0:256] = 40
    test_image[0:256, 256:512] = 220
    test_image[256:512, 0:256] = 220
    test_image[256:512, 256:512] = 40

    print("Step 3: Creating QuadtreeSegmentationModel...")
    quadtree_model = QuadtreeSegmentationModel(
        classifier=swin_model,
        classifier_dataset_dir=None,
        threshold=1.0,
        min_region_size=256,
        max_depth=None,
    )

    print("Step 4: Running segmentation...")
    dummy_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    seg_dataset = SegmentationDatasetInterface.from_pairs([(test_image, dummy_mask)])

    mask_pairs = quadtree_model.evaluate(seg_dataset)
    predicted_mask, _ = mask_pairs[0]

    print("Step 5: Verifying segmentation results...")
    top_left_class = int(np.median(predicted_mask[0:256, 0:256]))
    top_right_class = int(np.median(predicted_mask[0:256, 256:512]))
    bottom_left_class = int(np.median(predicted_mask[256:512, 0:256]))
    bottom_right_class = int(np.median(predicted_mask[256:512, 256:512]))

    assert top_left_class == 0
    assert top_right_class == 1
    assert bottom_left_class == 1
    assert bottom_right_class == 0
    print("Swin + Quadtree segmentation integration test PASSED!")


def test_swin_quadtree_deeper_recursion() -> None:
    """Test Quadtree with deeper recursion using trained Swin."""
    print("Testing Swin + Quadtree with deeper recursion...")
    set_deterministic_seed(42)

    swin_model = SwinModel(
        num_classes=2,
        channels=1,
        image_size=64,
        patch_size=4,
        window_size=4,
        embed_dim=48,
        depths=[2, 4],
        num_heads=[3, 6],
        dropout=0.1,
        device="cpu",
        train_epochs=5,
    )

    images, labels = create_binary_training_data(num_samples_per_class=25, size=64)
    dataset = ClassificationDatasetInterface.from_pairs(list(zip(images, labels)))
    swin_model.train(dataset=dataset)

    image_size = 512
    test_image = np.zeros((image_size, image_size), dtype=np.uint8)
    block_size = 128
    for i in range(4):
        for j in range(4):
            x_start, y_start = i * block_size, j * block_size
            x_end, y_end = (i + 1) * block_size, (j + 1) * block_size
            if (i + j) % 2 == 0:
                test_image[x_start:x_end, y_start:y_end] = 40
            else:
                test_image[x_start:x_end, y_start:y_end] = 220

    quadtree_model = QuadtreeSegmentationModel(
        classifier=swin_model,
        classifier_dataset_dir=None,
        threshold=0.99,
        min_region_size=128,
        max_depth=2,
    )

    dummy_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    seg_dataset = SegmentationDatasetInterface.from_pairs([(test_image, dummy_mask)])

    mask_pairs = quadtree_model.evaluate(seg_dataset)
    predicted_mask, _ = mask_pairs[0]

    assert predicted_mask[64, 64] == 0
    assert predicted_mask[64, 192] == 1
    assert predicted_mask[192, 64] == 1
    assert predicted_mask[192, 192] == 0
    print("Swin + Quadtree deeper recursion test PASSED!")
