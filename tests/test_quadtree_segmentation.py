import numpy as np

from auto_ml.implementations import QuadtreeSegmentationModel
from auto_ml.interfaces import (
    ClassificationDatasetInterface,
    ClassificationModelInterface,
    ImageArray,
    MetricsResultInterface,
    SegmentationDatasetInterface,
)


class DummyClassifier(ClassificationModelInterface):
    """
    A dummy classifier for testing QuadtreeSegmentationModel.

    Classifies regions based on the dominant color within the region.
    The confidence is the proportion of the dominant color's pixels.
    Assumes image is grayscale for simplicity or takes the average color.
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
        Classify an image region based on dominant color.

        Args:
            image: Image as a numpy array (H, W, C or H, W).
            x: x-coordinate of the region (top-left).
            y: y-coordinate of the region (top-left).
            width: Width of the region.
            height: Height of the region.

        Returns:
            Tuple of (class_label, confidence),
            where class_label is the dominant color value and confidence
            is the proportion of pixels with that color.

        """
        region = image[y : y + height, x : x + width]

        if region.size == 0:
            return 0, 0.0  # Default if region is empty

        # For a simple dummy, let's just use the average pixel value in the region
        # and map it to a class.
        if region.ndim == 3:
            # If it's a color image, convert to grayscale average for simplicity
            region_flat = np.mean(region, axis=2).flatten()
        else:
            region_flat = region.flatten()

        # Count occurrences of each unique pixel value
        _, counts = np.unique(region_flat, return_counts=True)

        # Find the dominant pixel
        dominant_count = np.max(counts)

        # Confidence is the proportion of the dominant pixel
        confidence = dominant_count / region_flat.size

        # Map average brightness to 3 classes (0, 1, 2) for this dummy.
        # This assumes input images will have varying brightness
        # levels that map to these classes.
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


def test_quadtree_segmentation_init() -> None:
    """Test the initialization of QuadtreeSegmentationModel."""
    print("Initializing Verification for QuadtreeSegmentationModel initialization...")
    classifier = DummyClassifier()
    model = QuadtreeSegmentationModel(
        classifier=classifier,
        classifier_dataset_dir=None,
        threshold=0.7,
    )

    assert model.classifier == classifier
    assert model.threshold == 0.7
    assert model.min_region_size == 1
    assert model.max_depth is None
    print("QuadtreeSegmentationModel initialization VERIFICATION SUCCESSFUL!")


def test_quadtree_segmentation_evaluate() -> None:
    """Test the evaluate method and recursive segmentation logic."""
    print("Initializing Verification for QuadtreeSegmentationModel evaluate method...")

    # Setup dummy data with distinct quadrants

    image_size = 512

    dummy_image = np.zeros((image_size, image_size), dtype=np.uint8)  # Grayscale image

    # Create four distinct quadrants with different average brightness
    # These brightness values should map to different class labels (0, 1, 2)
    # in DiagonalDummyClassifier (avg_brightness < 85 -> 0, < 170 -> 1, else -> 2)
    # Top-left (0): avg_brightness ~ 40 -> class 0
    dummy_image[0 : image_size // 2, 0 : image_size // 2] = 40
    # Top-right (1): avg_brightness ~ 120 -> class 1
    dummy_image[0 : image_size // 2, image_size // 2 : image_size] = 120
    # Bottom-left (2): avg_brightness ~ 200 -> class 2
    dummy_image[image_size // 2 : image_size, 0 : image_size // 2] = 200
    # Bottom-right (0): avg_brightness ~ 60 -> class 0
    dummy_image[image_size // 2 : image_size, image_size // 2 : image_size] = 60

    dummy_real_mask = np.zeros(
        (image_size, image_size),
        dtype=np.uint8,
    )  # Not used by dummy classifier

    dataset = SegmentationDatasetInterface()
    dataset.add_sample(dummy_image, dummy_real_mask)

    # Classifier and Model setup
    classifier = DummyClassifier()

    # Set threshold to a high value to force recursion until min_region_size
    # Set min_region_size to allow clear quadrant segmentation, e.g., 256
    model = QuadtreeSegmentationModel(
        classifier=classifier,
        classifier_dataset_dir=None,
        threshold=0.99,
        min_region_size=image_size // 2,  # 256
    )

    # Evaluate
    mask_pairs = model.evaluate(dataset)
    assert len(mask_pairs) == 1

    predicted_mask, real_mask = mask_pairs[0]
    assert predicted_mask.shape == (image_size, image_size)
    assert np.array_equal(real_mask, dummy_real_mask)  # Real mask should be unchanged

    # Expected mask based on the dummy_image and DiagonalDummyClassifier logic:
    expected_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    expected_mask[0 : image_size // 2, 0 : image_size // 2] = 0  # 40 -> class 0
    expected_mask[0 : image_size // 2, image_size // 2 : image_size] = (
        1  # 120 -> class 1
    )
    expected_mask[image_size // 2 : image_size, 0 : image_size // 2] = (
        2  # 200 -> class 2
    )
    expected_mask[image_size // 2 : image_size, image_size // 2 : image_size] = (
        0  # 60 -> class 0
    )
    assert np.array_equal(predicted_mask, expected_mask)
    print("QuadtreeSegmentationModel evaluate method VERIFICATION SUCCESSFUL!")


def test__should_stop_recursion() -> None:
    """Test the _should_stop_recursion utility method."""
    print("Initializing Verification for _should_stop_recursion method...")

    classifier = DummyClassifier()
    model = QuadtreeSegmentationModel(
        classifier=classifier,
        classifier_dataset_dir=None,
        threshold=0.7,
        min_region_size=10,
        max_depth=3,
    )

    # Case 1: Confidence meets threshold
    assert (
        model._should_stop_recursion(confidence=0.8, width=20, height=20, depth=1)
        is True
    )

    # Case 2: Width meets min_region_size
    assert (
        model._should_stop_recursion(confidence=0.6, width=5, height=20, depth=1)
        is True
    )

    # Case 3: Height meets min_region_size
    assert (
        model._should_stop_recursion(confidence=0.6, width=20, height=5, depth=1)
        is True
    )

    # Case 4: Max depth reached
    assert (
        model._should_stop_recursion(confidence=0.6, width=20, height=20, depth=3)
        is True
    )

    # Case 5: No stop condition met (should return False)
    assert (
        model._should_stop_recursion(confidence=0.6, width=20, height=20, depth=1)
        is False
    )

    # Case 6: max_depth is None, so depth should not stop recursion
    model_no_max_depth = QuadtreeSegmentationModel(
        classifier=classifier,
        classifier_dataset_dir=None,
        threshold=0.7,
        min_region_size=10,
        max_depth=None,
    )
    assert (
        model_no_max_depth._should_stop_recursion(
            confidence=0.6,
            width=20,
            height=20,
            depth=100,
        )
        is False
    )

    print("_should_stop_recursion method VERIFICATION SUCCESSFUL!")
