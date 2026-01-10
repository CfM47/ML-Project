import numpy as np
import pytest

from auto_ml.implementations.segmentators import SlidingWindowSegmentationModel
from auto_ml.interfaces import (
    ClassificationDatasetInterface,
    ClassificationModelInterface,
    ImageArray,
    MetricsResultInterface,
    SegmentationDatasetInterface,
)


class DummyClassifier(ClassificationModelInterface):
    """
    Dummy classifier for testing SlidingWindowSegmentationModel.

    Classifies regions based on the average brightness within the region.
    The confidence is proportional to how uniform the region is.
    """

    def classify(
        self,
        image: ImageArray,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> tuple[int, float]:
        """
        Classify an image region based on average brightness.

        Args:
            image: Image as a numpy array (H, W, C or H, W).
            x: x-coordinate of the region (top-left).
            y: y-coordinate of the region (top-left).
            width: Width of the region.
            height: Height of the region.

        Returns:
            Tuple of (class_label, confidence).

        """
        region = image[y : y + height, x : x + width]

        if region.size == 0:
            return 0, 0.0

        # Convert to grayscale if color image
        if region.ndim == 3:
            region_flat = np.mean(region, axis=2).flatten()
        else:
            region_flat = region.flatten()

        # Count occurrences of each unique pixel value
        _, counts = np.unique(region_flat, return_counts=True)

        # Confidence is the proportion of the dominant pixel
        dominant_count = np.max(counts)
        confidence = dominant_count / region_flat.size

        # Map average brightness to 3 classes (0, 1, 2)
        avg_brightness = np.mean(region_flat)
        if avg_brightness < 85:
            class_label = 0
        elif avg_brightness < 170:
            class_label = 1
        else:
            class_label = 2

        return class_label, confidence

    def train(
        self,
        dataset: ClassificationDatasetInterface,
    ) -> MetricsResultInterface:
        """Train the model on the provided dataset."""
        return MetricsResultInterface()

    def evaluate(
        self,
        dataset: ClassificationDatasetInterface,
    ) -> MetricsResultInterface:
        """Evaluate the model on the provided dataset."""
        return MetricsResultInterface()


def test_sliding_window_init() -> None:
    """Test the initialization of SlidingWindowSegmentationModel."""
    print(
        "Initializing Verification for SlidingWindowSegmentationModel "
        "initialization...",
    )
    classifier = DummyClassifier()

    # Test valid initialization
    model = SlidingWindowSegmentationModel(
        classifier=classifier,
        classifier_dataset_dir=None,
        window_size=128,
        stride=64,
        aggregation_method="majority_vote",
    )

    assert model.classifier == classifier
    assert model.window_size == 128
    assert model.stride == 64
    assert model.aggregation_method == "majority_vote"
    print("SlidingWindowSegmentationModel initialization VERIFICATION SUCCESSFUL!")


def test_sliding_window_parameter_validation() -> None:
    """Test parameter validation in SlidingWindowSegmentationModel."""
    print("Initializing Verification for parameter validation...")
    classifier = DummyClassifier()

    # Test stride > window_size (should raise ValueError)
    with pytest.raises(ValueError, match="stride.*must be <= window_size"):
        SlidingWindowSegmentationModel(
            classifier=classifier,
            window_size=64,
            stride=128,
        )

    # Test window_size too small (should raise ValueError)
    with pytest.raises(ValueError, match="window_size.*must be in range"):
        SlidingWindowSegmentationModel(
            classifier=classifier,
            window_size=4,
            stride=2,
        )

    # Test window_size too large (should raise ValueError)
    with pytest.raises(ValueError, match="window_size.*must be in range"):
        SlidingWindowSegmentationModel(
            classifier=classifier,
            window_size=1024,
            stride=512,
        )

    # Test invalid aggregation method
    with pytest.raises(ValueError, match="aggregation_method.*not supported"):
        SlidingWindowSegmentationModel(
            classifier=classifier,
            window_size=64,
            stride=32,
            aggregation_method="invalid_method",
        )

    print("Parameter validation VERIFICATION SUCCESSFUL!")


def test_sliding_window_basic_segmentation() -> None:
    """Test basic segmentation with non-overlapping windows."""
    print("Initializing Verification for basic sliding window segmentation...")

    image_size = 512

    # Create test image with 4 quadrants (different brightness)
    dummy_image = np.zeros((image_size, image_size), dtype=np.uint8)
    dummy_image[0 : image_size // 2, 0 : image_size // 2] = 40  # Class 0
    dummy_image[0 : image_size // 2, image_size // 2 : image_size] = 120
    dummy_image[image_size // 2 : image_size, 0 : image_size // 2] = 200
    dummy_image[image_size // 2 : image_size, image_size // 2 : image_size] = 60

    dummy_real_mask = np.zeros((image_size, image_size), dtype=np.uint8)

    dataset = SegmentationDatasetInterface()
    dataset.add_sample(dummy_image, dummy_real_mask)

    # Use non-overlapping windows (256x256 stride=256)
    # This should align perfectly with quadrants
    classifier = DummyClassifier()
    model = SlidingWindowSegmentationModel(
        classifier=classifier,
        classifier_dataset_dir=None,
        window_size=256,
        stride=256,
    )

    # Evaluate
    mask_pairs = model.evaluate(dataset)
    assert len(mask_pairs) == 1

    predicted_mask, real_mask = mask_pairs[0]
    assert predicted_mask.shape == (image_size, image_size)

    # Expected mask based on the dummy_image and DummyClassifier logic
    expected_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    expected_mask[0 : image_size // 2, 0 : image_size // 2] = 0
    expected_mask[0 : image_size // 2, image_size // 2 : image_size] = 1
    expected_mask[image_size // 2 : image_size, 0 : image_size // 2] = 2
    expected_mask[image_size // 2 : image_size, image_size // 2 : image_size] = 0

    assert np.array_equal(predicted_mask, expected_mask)
    print("Basic sliding window segmentation VERIFICATION SUCCESSFUL!")


def test_sliding_window_overlapping_votes() -> None:
    """Test vote aggregation with overlapping windows."""
    print("Initializing Verification for overlapping window votes...")

    image_size = 512

    # Create simple test image: left half dark (class 0), right half bright (class 2)
    dummy_image = np.zeros((image_size, image_size), dtype=np.uint8)
    dummy_image[:, 0 : image_size // 2] = 40  # Class 0 (dark)
    dummy_image[:, image_size // 2 : image_size] = 200  # Class 2 (bright)

    dummy_real_mask = np.zeros((image_size, image_size), dtype=np.uint8)

    dataset = SegmentationDatasetInterface()
    dataset.add_sample(dummy_image, dummy_real_mask)

    # Use overlapping windows (128x128, stride=64)
    classifier = DummyClassifier()
    model = SlidingWindowSegmentationModel(
        classifier=classifier,
        classifier_dataset_dir=None,
        window_size=128,
        stride=64,
    )

    # Evaluate
    mask_pairs = model.evaluate(dataset)
    assert len(mask_pairs) == 1

    predicted_mask, real_mask = mask_pairs[0]
    assert predicted_mask.shape == (image_size, image_size)

    # Expected: left half should be class 0, right half should be class 2
    # Center pixels might be affected by overlapping windows at the boundary
    # Check that left quarter is definitely class 0
    left_quarter = predicted_mask[:, 0 : image_size // 4]
    assert np.all(left_quarter == 0), "Left quarter should be class 0"

    # Check that right quarter is definitely class 2
    right_quarter = predicted_mask[:, 3 * image_size // 4 : image_size]
    assert np.all(right_quarter == 2), "Right quarter should be class 2"

    print("Overlapping window votes VERIFICATION SUCCESSFUL!")


def test_sliding_window_edge_handling() -> None:
    """Test that edge pixels are handled correctly."""
    print("Initializing Verification for edge pixel handling...")

    image_size = 512

    # Create test image with left half dark, right half bright
    dummy_image = np.zeros((image_size, image_size), dtype=np.uint8)
    dummy_image[:, 0 : image_size // 2] = 40  # Left half dark (class 0)
    dummy_image[:, image_size // 2 : image_size] = 200  # Right half bright (class 2)

    dummy_real_mask = np.zeros((image_size, image_size), dtype=np.uint8)

    dataset = SegmentationDatasetInterface()
    dataset.add_sample(dummy_image, dummy_real_mask)

    # Use windows that will test edge boundaries
    classifier = DummyClassifier()
    model = SlidingWindowSegmentationModel(
        classifier=classifier,
        classifier_dataset_dir=None,
        window_size=64,
        stride=32,
    )

    # Evaluate
    mask_pairs = model.evaluate(dataset)
    assert len(mask_pairs) == 1

    predicted_mask, real_mask = mask_pairs[0]
    assert predicted_mask.shape == (image_size, image_size)

    # Check that left edge pixels are class 0 (dark)
    assert predicted_mask[0, 0] == 0, "Top-left corner should be class 0"
    assert (
        predicted_mask[image_size - 1, 0] == 0
    ), "Bottom-left corner should be class 0"

    # Check that right edge pixels are class 2 (bright)
    assert (
        predicted_mask[0, image_size - 1] == 2
    ), "Top-right corner should be class 2"
    assert (
        predicted_mask[image_size - 1, image_size - 1] == 2
    ), "Bottom-right corner should be class 2"

    print("Edge pixel handling VERIFICATION SUCCESSFUL!")


def test_sliding_window_train_and_evaluate() -> None:
    """Test train() and evaluate() methods."""
    print("Initializing Verification for train() and evaluate()...")

    image_size = 512

    # Create simple dataset
    dummy_image = np.full((image_size, image_size), 100, dtype=np.uint8)  # Class 1
    dummy_real_mask = np.ones((image_size, image_size), dtype=np.uint8)  # All class 1

    dataset = SegmentationDatasetInterface()
    dataset.add_sample(dummy_image, dummy_real_mask)

    classifier = DummyClassifier()
    model = SlidingWindowSegmentationModel(
        classifier=classifier,
        classifier_dataset_dir=None,
        window_size=128,
        stride=128,
    )

    # Train (without classifier training)
    metrics = model.train(dataset)

    # Verify metrics interface
    assert isinstance(metrics, MetricsResultInterface)
    assert hasattr(metrics, "accuracy")
    assert hasattr(metrics, "loss")
    assert hasattr(metrics, "iou")
    assert hasattr(metrics, "precision")
    assert hasattr(metrics, "recall")
    assert hasattr(metrics, "f1_score")

    # The predicted mask should be all class 1 (brightness ~100)
    # So accuracy should be high
    assert metrics.accuracy > 0.8, f"Expected high accuracy, got {metrics.accuracy}"

    print("train() and evaluate() VERIFICATION SUCCESSFUL!")


def test_sliding_window_metrics_computation() -> None:
    """Test that metrics are computed correctly."""
    print("Initializing Verification for metrics computation...")

    image_size = 512

    # Create test case where we know the exact metrics
    # Image: all class 0
    dummy_image = np.full((image_size, image_size), 40, dtype=np.uint8)  # Class 0
    # Mask: all class 0
    dummy_real_mask = np.zeros((image_size, image_size), dtype=np.uint8)  # Class 0

    dataset = SegmentationDatasetInterface()
    dataset.add_sample(dummy_image, dummy_real_mask)

    classifier = DummyClassifier()
    model = SlidingWindowSegmentationModel(
        classifier=classifier,
        classifier_dataset_dir=None,
        window_size=256,
        stride=256,
    )

    # Evaluate
    mask_pairs = model.evaluate(dataset)
    metrics = model._compute_metrics(mask_pairs)

    # Since predicted and real masks are both all class 0, accuracy should be 1.0
    assert metrics.accuracy == 1.0, f"Expected accuracy 1.0, got {metrics.accuracy}"
    assert metrics.loss == 0.0, f"Expected loss 0.0, got {metrics.loss}"

    # Mean IoU across 3 classes where only class 0 appears = (1.0 + 0.0 + 0.0) / 3
    # This is expected behavior - matches quadtree implementation
    assert (
        0.3 <= metrics.iou <= 0.4
    ), f"Expected IoU ~0.33 (mean across 3 classes), got {metrics.iou}"

    # Mean precision/recall/F1 will also be ~0.33 for same reason
    assert (
        0.3 <= metrics.precision <= 0.4
    ), f"Expected precision ~0.33, got {metrics.precision}"
    assert 0.3 <= metrics.recall <= 0.4, f"Expected recall ~0.33, got {metrics.recall}"
    assert (
        0.3 <= metrics.f1_score <= 0.4
    ), f"Expected F1 ~0.33, got {metrics.f1_score}"

    print("Metrics computation VERIFICATION SUCCESSFUL!")
