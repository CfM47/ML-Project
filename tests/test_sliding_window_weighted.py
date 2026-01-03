import numpy as np

from auto_ml.implementations.segmentators import SlidingWindowSegmentationModel
from auto_ml.interfaces import (
    ClassificationDatasetInterface,
    ClassificationModelInterface,
    ImageArray,
    MetricsResultInterface,
    SegmentationDatasetInterface,
)


class ConfidenceTestClassifier(ClassificationModelInterface):
    """
    Test classifier that returns controlled confidence values.

    Return high confidence for specific regions, low for others.
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
        Return different confidence scores based on region location.

        Top half: class 0 with low confidence (0.3)
        Bottom half: class 1 with high confidence (0.9)
        """
        avg_y = y + height // 2

        if avg_y < 256:
            # Top half - class 0, low confidence
            return 0, 0.3
        else:
            # Bottom half - class 1, high confidence
            return 1, 0.9

    def train(
        self,
        dataset: ClassificationDatasetInterface,
    ) -> MetricsResultInterface:
        """Train the model."""
        return MetricsResultInterface()


def test_confidence_weighted_voting() -> None:
    """Test that confidence weighting favors high-confidence predictions."""
    print("Testing confidence-weighted voting...")

    image_size = 512

    # Create uniform test image (brightness doesn't matter for this test)
    dummy_image = np.full((image_size, image_size), 100, dtype=np.uint8)
    dummy_real_mask = np.zeros((image_size, image_size), dtype=np.uint8)

    dataset = SegmentationDatasetInterface()
    dataset.add_sample(dummy_image, dummy_real_mask)

    # Test with confidence weighting
    classifier = ConfidenceTestClassifier()
    model_weighted = SlidingWindowSegmentationModel(
        classifier=classifier,
        window_size=128,
        stride=64,
        aggregation_method="confidence_weighted",
    )

    mask_pairs = model_weighted.evaluate(dataset)
    predicted_mask, _ = mask_pairs[0]

    # Verify top quarter is class 0 (away from boundary)
    # Checking top quarter to avoid boundary effects from overlapping windows
    top_quarter = predicted_mask[0:128, :]
    assert np.all(top_quarter == 0), "Top quarter should be class 0"

    # Verify bottom quarter is class 1 (high confidence, away from boundary)
    bottom_quarter = predicted_mask[384:512, :]
    assert np.all(bottom_quarter == 1), "Bottom quarter should be class 1"

    print("Confidence-weighted voting test PASSED!")


def test_majority_vote_still_works() -> None:
    """Test that majority_vote aggregation still works correctly."""
    print("Testing backward compatibility with majority_vote...")

    image_size = 512
    dummy_image = np.full((image_size, image_size), 100, dtype=np.uint8)
    dummy_real_mask = np.zeros((image_size, image_size), dtype=np.uint8)

    dataset = SegmentationDatasetInterface()
    dataset.add_sample(dummy_image, dummy_real_mask)

    classifier = ConfidenceTestClassifier()
    model_majority = SlidingWindowSegmentationModel(
        classifier=classifier,
        window_size=128,
        stride=64,
        aggregation_method="majority_vote",
    )

    # Should not raise any errors
    mask_pairs = model_majority.evaluate(dataset)
    predicted_mask, _ = mask_pairs[0]

    assert predicted_mask.shape == (512, 512)
    print("Majority vote backward compatibility test PASSED!")


def test_default_is_confidence_weighted() -> None:
    """Test that the default aggregation method is confidence_weighted."""
    print("Testing default aggregation method...")

    classifier = ConfidenceTestClassifier()
    model = SlidingWindowSegmentationModel(
        classifier=classifier,
        window_size=64,
        stride=32,
    )

    assert model.aggregation_method == "confidence_weighted", (
        "Default aggregation method should be 'confidence_weighted'"
    )
    print("Default aggregation method test PASSED!")
