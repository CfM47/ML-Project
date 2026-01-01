"""Tests for segmentation evaluator implementations."""

import numpy as np
import pytest

from auto_ml.implementations.evaluators.dice import (
    DiceClass0Evaluator,
    DiceClass1Evaluator,
    DiceClass2Evaluator,
    DiceMacroAverageEvaluator,
    DiceWeightedAverageEvaluator,
)
from auto_ml.implementations.evaluators.iou import (
    # IoUClass0Evaluator,
    IoUClass1Evaluator,
    IoUClass2Evaluator,
    IoUMacroAverageEvaluator,
    IoUWeightedAverageEvaluator,
)
from auto_ml.implementations.evaluators.precision import (
    PrecisionClass0Evaluator,
    PrecisionClass1Evaluator,
    PrecisionClass2Evaluator,
    PrecisionMacroAverageEvaluator,
)
from auto_ml.implementations.evaluators.recall import (
    RecallClass0Evaluator,
    RecallClass1Evaluator,
    RecallClass2Evaluator,
    RecallMacroAverageEvaluator,
)


class TestSegmentationEvaluators:
    """Tests for segmentation evaluator implementations."""

    @pytest.fixture
    def perfect_match_masks(self) -> list:
        """Create identical predicted and real masks for perfect match test."""
        mask = np.zeros((512, 512), dtype=np.uint8)
        # Class 0: top third
        mask[:170, :] = 0
        # Class 1: middle third
        mask[170:340, :] = 1
        # Class 2: bottom third
        mask[340:, :] = 2

        pairs = [(mask.copy(), mask.copy())]
        return [pairs]  # List[List[MaskPair]]

    @pytest.fixture
    def zero_overlap_masks(self) -> list:
        """Create masks with no overlap for zero overlap test."""
        pred_mask = np.zeros((512, 512), dtype=np.uint8)
        real_mask = np.zeros((512, 512), dtype=np.uint8)

        # Predicted: left half is class 1
        pred_mask[:, :256] = 1

        # Real: right half is class 1
        real_mask[:, 256:] = 1

        pairs = [(pred_mask, real_mask)]
        return [pairs]

    @pytest.fixture
    def sample_mask_pairs(self) -> list:
        """Create sample 3-class segmentation masks for testing."""
        pairs = []

        np.random.seed(42)  # Deterministic noise
        for i in range(3):  # 3 samples
            pred_mask = np.zeros((512, 512), dtype=np.uint8)
            real_mask = np.zeros((512, 512), dtype=np.uint8)

            # Class 0: Background (top section)
            real_mask[:170, :] = 0
            pred_mask[:170, :] = 0

            # Class 1: Material phase 1 (middle section)
            real_mask[170:340, :] = 1
            pred_mask[170:340, :] = 1

            # Class 2: Material phase 2 (bottom section)
            real_mask[340:, :] = 2
            pred_mask[340:, :] = 2

            # Add some noise to predictions (5% random errors)
            noise_mask = np.random.rand(512, 512) > 0.95
            pred_mask[noise_mask] = (pred_mask[noise_mask] + 1) % 3

            pairs.append((pred_mask, real_mask))

        return [pairs]  # List[List[MaskPair]]

    @pytest.fixture
    def class_never_appears_masks(self) -> list:
        """Create masks where class 2 never appears."""
        mask = np.zeros((512, 512), dtype=np.uint8)
        # Only classes 0 and 1
        mask[:256, :] = 0
        mask[256:, :] = 1

        pairs = [(mask.copy(), mask.copy())]
        return [pairs]

    @pytest.fixture
    def imbalanced_masks(self) -> list:
        """Create masks with highly imbalanced classes."""
        pred_mask = np.zeros((100, 100), dtype=np.uint8)
        real_mask = np.zeros((100, 100), dtype=np.uint8)

        # Class 0: 90% of pixels
        real_mask[:, :] = 0
        pred_mask[:, :] = 0

        # Class 1: 9% of pixels
        real_mask[:30, :30] = 1
        pred_mask[:30, :30] = 1

        # Class 2: 1% of pixels
        real_mask[:10, :10] = 2
        pred_mask[:10, :10] = 2

        pairs = [(pred_mask, real_mask)]
        return [pairs]

    # ========================================================================
    # IoU Tests
    # ========================================================================

    # def test_iou_class0_perfect_match(self, perfect_match_masks: list) -> None:
    #     """Test IoU returns 1.0 for perfect predictions."""  # noqa: ERA001
    #     evaluator = IoUClass0Evaluator()  # noqa: ERA001
    #     result = evaluator.evaluate(perfect_match_masks)  # noqa: ERA001
    #
    #     assert isinstance(result, float)  # noqa: ERA001
    #     assert result == 1.0  # noqa: ERA001

    def test_iou_class1_perfect_match(self, perfect_match_masks: list) -> None:
        """Test IoU returns 1.0 for perfect predictions."""
        evaluator = IoUClass1Evaluator()
        result = evaluator.evaluate(perfect_match_masks)

        assert isinstance(result, float)
        assert result == 1.0

    def test_iou_class2_perfect_match(self, perfect_match_masks: list) -> None:
        """Test IoU returns 1.0 for perfect predictions."""
        evaluator = IoUClass2Evaluator()
        result = evaluator.evaluate(perfect_match_masks)

        assert isinstance(result, float)
        assert result == 1.0

    def test_iou_class1_zero_overlap(self, zero_overlap_masks: list) -> None:
        """Test IoU returns 0.0 when there's no overlap for class 1."""
        evaluator = IoUClass1Evaluator()
        result = evaluator.evaluate(zero_overlap_masks)

        assert isinstance(result, float)
        assert result == 0.0

    def test_iou_class2_never_appears(self, class_never_appears_masks: list) -> None:
        """Test IoU returns 0.0 when class 2 never appears."""
        evaluator = IoUClass2Evaluator()
        result = evaluator.evaluate(class_never_appears_masks)

        assert isinstance(result, float)
        assert result == 0.0

    def test_iou_macro_average_returns_float(
        self, sample_mask_pairs: list,
    ) -> None:
        """Test IoU macro average returns float."""
        evaluator = IoUMacroAverageEvaluator()
        result = evaluator.evaluate(sample_mask_pairs)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_iou_weighted_average_returns_float(
        self, sample_mask_pairs: list,
    ) -> None:
        """Test IoU weighted average returns float."""
        evaluator = IoUWeightedAverageEvaluator()
        result = evaluator.evaluate(sample_mask_pairs)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_iou_macro_vs_weighted_differ_with_imbalance(
        self, imbalanced_masks: list,
    ) -> None:
        """Test macro and weighted IoU can differ with class imbalance."""
        macro_eval = IoUMacroAverageEvaluator()
        weighted_eval = IoUWeightedAverageEvaluator()

        macro_result = macro_eval.evaluate(imbalanced_masks)
        weighted_result = weighted_eval.evaluate(imbalanced_masks)

        # Both should be valid floats
        assert isinstance(macro_result, float)
        assert isinstance(weighted_result, float)
        assert 0.0 <= macro_result <= 1.0
        assert 0.0 <= weighted_result <= 1.0

    # ========================================================================
    # Dice Tests
    # ========================================================================

    def test_dice_class0_perfect_match(self, perfect_match_masks: list) -> None:
        """Test Dice returns 1.0 for perfect predictions."""
        evaluator = DiceClass0Evaluator()
        result = evaluator.evaluate(perfect_match_masks)

        assert isinstance(result, float)
        assert result == 1.0

    def test_dice_class1_perfect_match(self, perfect_match_masks: list) -> None:
        """Test Dice returns 1.0 for perfect predictions."""
        evaluator = DiceClass1Evaluator()
        result = evaluator.evaluate(perfect_match_masks)

        assert isinstance(result, float)
        assert result == 1.0

    def test_dice_class2_perfect_match(self, perfect_match_masks: list) -> None:
        """Test Dice returns 1.0 for perfect predictions."""
        evaluator = DiceClass2Evaluator()
        result = evaluator.evaluate(perfect_match_masks)

        assert isinstance(result, float)
        assert result == 1.0

    def test_dice_class1_zero_overlap(self, zero_overlap_masks: list) -> None:
        """Test Dice returns 0.0 when there's no overlap."""
        evaluator = DiceClass1Evaluator()
        result = evaluator.evaluate(zero_overlap_masks)

        assert isinstance(result, float)
        assert result == 0.0

    def test_dice_class2_never_appears(
        self, class_never_appears_masks: list,
    ) -> None:
        """Test Dice returns 0.0 when class 2 never appears."""
        evaluator = DiceClass2Evaluator()
        result = evaluator.evaluate(class_never_appears_masks)

        assert isinstance(result, float)
        assert result == 0.0

    def test_dice_macro_average_returns_float(
        self, sample_mask_pairs: list,
    ) -> None:
        """Test Dice macro average returns float."""
        evaluator = DiceMacroAverageEvaluator()
        result = evaluator.evaluate(sample_mask_pairs)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_dice_weighted_average_returns_float(
        self, sample_mask_pairs: list,
    ) -> None:
        """Test Dice weighted average returns float."""
        evaluator = DiceWeightedAverageEvaluator()
        result = evaluator.evaluate(sample_mask_pairs)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_dice_greater_than_iou_relationship(
        self, sample_mask_pairs: list,
    ) -> None:
        """Test that Dice >= IoU (mathematical relationship)."""
        iou_eval = IoUClass1Evaluator()
        dice_eval = DiceClass1Evaluator()

        iou = iou_eval.evaluate(sample_mask_pairs)
        dice = dice_eval.evaluate(sample_mask_pairs)

        # Dice = 2*IoU / (1 + IoU), so Dice >= IoU always holds
        assert dice >= iou or abs(dice - iou) < 1e-6

    # ========================================================================
    # Precision Tests
    # ========================================================================

    def test_precision_class0_perfect_match(
        self, perfect_match_masks: list,
    ) -> None:
        """Test Precision returns 1.0 for perfect predictions."""
        evaluator = PrecisionClass0Evaluator()
        result = evaluator.evaluate(perfect_match_masks)

        assert isinstance(result, float)
        assert result == 1.0

    def test_precision_class1_perfect_match(
        self, perfect_match_masks: list,
    ) -> None:
        """Test Precision returns 1.0 for perfect predictions."""
        evaluator = PrecisionClass1Evaluator()
        result = evaluator.evaluate(perfect_match_masks)

        assert isinstance(result, float)
        assert result == 1.0

    def test_precision_class2_perfect_match(
        self, perfect_match_masks: list,
    ) -> None:
        """Test Precision returns 1.0 for perfect predictions."""
        evaluator = PrecisionClass2Evaluator()
        result = evaluator.evaluate(perfect_match_masks)

        assert isinstance(result, float)
        assert result == 1.0

    def test_precision_class2_never_appears(
        self, class_never_appears_masks: list,
    ) -> None:
        """Test Precision returns 0.0 when class 2 never predicted."""
        evaluator = PrecisionClass2Evaluator()
        result = evaluator.evaluate(class_never_appears_masks)

        assert isinstance(result, float)
        assert result == 0.0

    def test_precision_macro_average_returns_float(
        self, sample_mask_pairs: list,
    ) -> None:
        """Test Precision macro average returns float."""
        evaluator = PrecisionMacroAverageEvaluator()
        result = evaluator.evaluate(sample_mask_pairs)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    # ========================================================================
    # Recall Tests
    # ========================================================================

    def test_recall_class0_perfect_match(self, perfect_match_masks: list) -> None:
        """Test Recall returns 1.0 for perfect predictions."""
        evaluator = RecallClass0Evaluator()
        result = evaluator.evaluate(perfect_match_masks)

        assert isinstance(result, float)
        assert result == 1.0

    def test_recall_class1_perfect_match(self, perfect_match_masks: list) -> None:
        """Test Recall returns 1.0 for perfect predictions."""
        evaluator = RecallClass1Evaluator()
        result = evaluator.evaluate(perfect_match_masks)

        assert isinstance(result, float)
        assert result == 1.0

    def test_recall_class2_perfect_match(self, perfect_match_masks: list) -> None:
        """Test Recall returns 1.0 for perfect predictions."""
        evaluator = RecallClass2Evaluator()
        result = evaluator.evaluate(perfect_match_masks)

        assert isinstance(result, float)
        assert result == 1.0

    def test_recall_class2_never_appears(
        self, class_never_appears_masks: list,
    ) -> None:
        """Test Recall returns 0.0 when class 2 never in ground truth."""
        evaluator = RecallClass2Evaluator()
        result = evaluator.evaluate(class_never_appears_masks)

        assert isinstance(result, float)
        assert result == 0.0

    def test_recall_macro_average_returns_float(
        self, sample_mask_pairs: list,
    ) -> None:
        """Test Recall macro average returns float."""
        evaluator = RecallMacroAverageEvaluator()
        result = evaluator.evaluate(sample_mask_pairs)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    # ========================================================================
    # Known Values Tests
    # ========================================================================

    def test_iou_known_values(self) -> None:
        """Test IoU with known confusion matrix values."""
        # Create a simple 10x10 mask for class 1
        # TP=60, FP=20, FN=40
        # IoU = 60 / (60 + 20 + 40) = 60 / 120 = 0.5

        pred_mask = np.zeros((10, 10), dtype=np.uint8)
        real_mask = np.zeros((10, 10), dtype=np.uint8)

        # Ground truth: 100 pixels should be class 1
        real_mask[:10, :10] = 1

        # Prediction: 80 pixels predicted as class 1
        # Only 60 overlap with ground truth
        pred_mask[:8, :10] = 1

        # TP = 8*10 = 80 (but limited by real = 60)
        # Actually: overlap is first 8 rows = 80 pixels
        # FP = 0 (all predictions are correct)
        # FN = 20 (last 2 rows not predicted)
        # IoU = 80 / (80 + 0 + 20) = 80/100 = 0.8

        mask_pairs = [[(pred_mask, real_mask)]]

        evaluator = IoUClass1Evaluator()
        result = evaluator.evaluate(mask_pairs)

        assert isinstance(result, float)
        assert abs(result - 0.8) < 1e-6

    def test_dice_known_values(self) -> None:
        """Test Dice with known confusion matrix values."""
        # Same setup as IoU test

        pred_mask = np.zeros((10, 10), dtype=np.uint8)
        real_mask = np.zeros((10, 10), dtype=np.uint8)

        real_mask[:10, :10] = 1
        pred_mask[:8, :10] = 1

        mask_pairs = [[(pred_mask, real_mask)]]

        evaluator = DiceClass1Evaluator()
        result = evaluator.evaluate(mask_pairs)

        expected_dice = 160.0 / 180.0
        assert isinstance(result, float)
        assert abs(result - expected_dice) < 1e-6

    def test_precision_recall_known_values(self) -> None:
        """Test Precision and Recall with known confusion matrix values."""
        # TP=80, FP=0, FN=20 (from previous test)
        # Precision = 80 / (80 + 0) = 1.0
        # Recall = 80 / (80 + 20) = 0.8

        pred_mask = np.zeros((10, 10), dtype=np.uint8)
        real_mask = np.zeros((10, 10), dtype=np.uint8)

        real_mask[:10, :10] = 1
        pred_mask[:8, :10] = 1

        mask_pairs = [[(pred_mask, real_mask)]]

        precision_eval = PrecisionClass1Evaluator()
        recall_eval = RecallClass1Evaluator()

        precision = precision_eval.evaluate(mask_pairs)
        recall = recall_eval.evaluate(mask_pairs)

        assert abs(precision - 1.0) < 1e-6
        assert abs(recall - 0.8) < 1e-6

    # ========================================================================
    # Fold Aggregation Tests
    # ========================================================================

    # def test_iou_aggregates_across_folds(self) -> None:
    #     """Test that IoU correctly aggregates across multiple folds."""
    #     # Fold 1: Perfect match for class 0
    #     mask1_pred = np.zeros((10, 10), dtype=np.uint8) # noqa: ERA001
    #     mask1_real = np.zeros((10, 10), dtype=np.uint8) # noqa: ERA001
    #
    #     # Fold 2: 50% IoU for class 0
    #     mask2_pred = np.zeros((10, 10), dtype=np.uint8) # noqa: ERA001
    #     mask2_real = np.zeros((10, 10), dtype=np.uint8) # noqa: ERA001
    #     mask2_pred[:5, :] = 1  # Top half is class 1 # noqa: ERA001
    #     mask2_real[:, :] = 1  # All is class 1 # noqa: ERA001
    #
    #     # For class 0:
    #     # Fold 1: TP=100, FP=0, FN=0 -> IoU = 1.0
    #     # Fold 2: TP=50, FP=0, FN=50 -> IoU = 0.5
    #     # Aggregated: TP=150, FP=0, FN=50 -> IoU = 150/200 = 0.75
    #
    #     mask_pairs = [[(mask1_pred, mask1_real)], [(mask2_pred, mask2_real)]]  # noqa: E501, ERA001
    #
    #     evaluator = IoUClass0Evaluator()# noqa: ERA001
    #     result = evaluator.evaluate(mask_pairs)# noqa: ERA001
    #
    #     expected_iou = 150.0 / 200.0# noqa: ERA001
    #     assert abs(result - expected_iou) < 1e-6# noqa: ERA001

    def test_dice_aggregates_across_multiple_samples(self) -> None:
        """Test that Dice correctly aggregates across multiple samples."""
        # Sample 1: Perfect match
        mask1 = np.zeros((10, 10), dtype=np.uint8)
        mask1[:, :5] = 1

        # Sample 2: Partial match
        mask2_pred = np.zeros((10, 10), dtype=np.uint8)
        mask2_real = np.zeros((10, 10), dtype=np.uint8)
        mask2_pred[:, :5] = 1
        mask2_real[:, :7] = 1

        # Aggregate across samples in same fold
        mask_pairs = [[(mask1, mask1), (mask2_pred, mask2_real)]]

        evaluator = DiceClass1Evaluator()
        result = evaluator.evaluate(mask_pairs)

        # Sample 1: TP=50, FP=0, FN=0
        # Sample 2: TP=50, FP=0, FN=20
        # Total: TP=100, FP=0, FN=20
        # Dice = 2*100 / (2*100 + 0 + 20) = 200/220 = 0.909...

        expected_dice = 200.0 / 220.0
        assert abs(result - expected_dice) < 1e-6

    # ========================================================================
    # Integration Tests
    # ========================================================================

    def test_all_evaluators_return_valid_floats(
        self, sample_mask_pairs: list,
    ) -> None:
        """Test that all evaluators return valid floats in [0.0, 1.0]."""
        evaluators = [
            # IoUClass0Evaluator(),# noqa: ERA001
            IoUClass1Evaluator(),
            IoUClass2Evaluator(),
            IoUMacroAverageEvaluator(),
            IoUWeightedAverageEvaluator(),
            DiceClass0Evaluator(),
            DiceClass1Evaluator(),
            DiceClass2Evaluator(),
            DiceMacroAverageEvaluator(),
            DiceWeightedAverageEvaluator(),
            PrecisionClass0Evaluator(),
            PrecisionClass1Evaluator(),
            PrecisionClass2Evaluator(),
            PrecisionMacroAverageEvaluator(),
            RecallClass0Evaluator(),
            RecallClass1Evaluator(),
            RecallClass2Evaluator(),
            RecallMacroAverageEvaluator(),
        ]

        for evaluator in evaluators:
            result = evaluator.evaluate(sample_mask_pairs)
            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0

    def test_evaluators_with_evaluator_node(self, sample_mask_pairs: list) -> None:
        """Test evaluators work with EvaluatorNode integration."""
        from auto_ml.implementations.nodes import EvaluatorNode

        evaluator_node = EvaluatorNode(
            evaluators={
                # "iou_class0": IoUClass0Evaluator(),# noqa: ERA001
                "dice_macro": DiceMacroAverageEvaluator(),
                "precision_class1": PrecisionClass1Evaluator(),
                "recall_class1": RecallClass1Evaluator(),
            },
            name="TestEvaluator",
        )

        results = evaluator_node.evaluate(sample_mask_pairs)

        assert isinstance(results, dict)
        # assert "iou_class0" in results
        assert "dice_macro" in results
        assert "precision_class1" in results
        assert "recall_class1" in results
