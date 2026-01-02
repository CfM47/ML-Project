"""Sliding window-based segmentation model implementation."""

from pathlib import Path
from typing import List, Optional

import numpy as np

from auto_ml.implementations.datasets import load_classification_dataset_from_dir
from auto_ml.interfaces import (
    ClassificationModelInterface,
    ImageArray,
    MaskArray,
    MaskPair,
    MetricsResultInterface,
    SegmentationDatasetInterface,
    SegmentationModelInterface,
)


class SlidingWindowSegmentationModel(SegmentationModelInterface):
    """
    Sliding window-based image segmentation model.

    The model applies a fixed-size sliding window across the entire 512x512 image,
    classifies each window position using an injected ClassificationModelInterface,
    and aggregates overlapping classifications via majority voting to produce a
    final per-pixel segmentation mask.
    """

    def __init__(
        self,
        classifier: ClassificationModelInterface,
        classifier_dataset_dir: Optional[Path] = None,
        window_size: int = 64,
        stride: int = 32,
        aggregation_method: str = "majority_vote",
    ) -> None:
        """
        Initialize the sliding window segmentation model.

        Args:
            classifier: Region classifier implementing
                        ClassificationModelInterface.
            classifier_dataset_dir: Directory containing dataset for training
                                    the classifier. If None, assumes classifier
                                    is already trained.
            window_size: Size of the sliding window (e.g., 32, 64, 128, 256).
            stride: Step size for moving the window across the image.
            aggregation_method: Method for aggregating votes from overlapping
                                windows. Currently only "majority_vote" is
                                supported.

        """
        self.classifier = classifier
        self.window_size = window_size
        self.stride = stride
        self.aggregation_method = aggregation_method
        self.classifier_dataset_dir = classifier_dataset_dir

        self._validate_parameters()

    def train(
        self,
        dataset: SegmentationDatasetInterface,
        validation_dataset: SegmentationDatasetInterface | None = None,
    ) -> MetricsResultInterface:
        """
        Train the sliding window segmenter.

        Args:
            dataset: Segmentation dataset for training.
            validation_dataset: Optional validation dataset (currently unused).

        Returns:
            MetricsResultInterface containing segmentation quality metrics.

        """
        print("[SlidingWindowSegmentationModel] Starting training...")
        print(f"[SlidingWindowSegmentationModel] Dataset size: {len(dataset)}")
        print(f"[SlidingWindowSegmentationModel] Window size: {self.window_size}")
        print(f"[SlidingWindowSegmentationModel] Stride: {self.stride}")
        print(
            "[SlidingWindowSegmentationModel] "
            f"Aggregation method: {self.aggregation_method}",
        )

        if self.classifier_dataset_dir is not None:
            dir_path = self.classifier_dataset_dir
            print(
                "[SlidingWindowSegmentationModel] Training classifier "
                f"from: {dir_path}",
            )
            classifier_dataset = load_classification_dataset_from_dir(
                self.classifier_dataset_dir,
            )
            ds_size = len(classifier_dataset)
            print(
                "[SlidingWindowSegmentationModel] Classifier "
                f"dataset size: {ds_size}",
            )
            self.classifier.train(classifier_dataset)
            print("[SlidingWindowSegmentationModel] Classifier training completed")
        else:
            print("[SlidingWindowSegmentationModel] Using pre-trained classifier")

        # Evaluate on training dataset to compute baseline metrics
        print(
            "[SlidingWindowSegmentationModel] "
            "Evaluating on training dataset...",
        )
        predicted_real_pairs = self.evaluate(dataset)
        metrics = self._compute_metrics(predicted_real_pairs)

        print("[SlidingWindowSegmentationModel] Training metrics:")
        print(f"  - Accuracy: {metrics.accuracy:.4f}")
        print(f"  - Loss: {metrics.loss:.4f}")
        print(f"  - IoU: {metrics.iou:.4f}")
        print(f"  - Precision: {metrics.precision:.4f}")
        print(f"  - Recall: {metrics.recall:.4f}")
        print(f"  - F1 Score: {metrics.f1_score:.4f}")

        return metrics

    def evaluate(self, dataset: SegmentationDatasetInterface) -> List[MaskPair]:
        """
        Evaluate the model on a dataset.

        For each image, a segmentation mask is produced using sliding window
        classification and vote aggregation.

        Returns:
            List of (predicted_mask, real_mask) tuples.

        """
        print(
            "[SlidingWindowSegmentationModel.evaluate] Evaluating "
            f"{len(dataset)} images...",
        )
        results = []
        for idx, (image, real_mask) in enumerate(dataset):
            if (idx + 1) % max(1, len(dataset) // 10) == 0 or idx == 0:
                print(f"  Processing image {idx + 1}/{len(dataset)}")
            results.append((self._segment_image(image), real_mask))
        print("[SlidingWindowSegmentationModel.evaluate] Evaluation completed")
        return results

    def _segment_image(self, image: ImageArray) -> MaskArray:
        """
        Segment a single image using sliding window classification.

        Args:
            image: Input image (512x512).

        Returns:
            Predicted segmentation mask (512x512).

        """
        print(
            "[SlidingWindowSegmentationModel._segment_image] "
            f"Starting segmentation (window_size={self.window_size}, "
            f"stride={self.stride})",
        )

        # Initialize vote accumulator: (512, 512, 3) for 3 classes
        num_classes = 3
        vote_map = np.zeros((512, 512, num_classes), dtype=np.int32)

        # Slide window across image
        window_count = 0
        for y in range(0, 512, self.stride):
            for x in range(0, 512, self.stride):
                # Handle edge boundaries (windows may be smaller at edges)
                actual_width = min(self.window_size, 512 - x)
                actual_height = min(self.window_size, 512 - y)

                # Classify this window region
                label, confidence = self.classifier.classify(
                    image=image,
                    x=x,
                    y=y,
                    width=actual_width,
                    height=actual_height,
                )

                # Add vote for all pixels in this window
                if self.aggregation_method == "majority_vote":
                    vote_map[y : y + actual_height, x : x + actual_width, label] += 1
                else:
                    # Future: support confidence-weighted voting
                    vote_map[y : y + actual_height, x : x + actual_width, label] += 1

                window_count += 1

        print(
            f"[SlidingWindowSegmentationModel._segment_image] "
            f"Processed {window_count} windows",
        )

        # Final mask: argmax across class dimension
        mask = np.argmax(vote_map, axis=2).astype(np.uint8)

        print("[SlidingWindowSegmentationModel._segment_image] Segmentation completed")
        return mask

    def _validate_parameters(self) -> None:
        """
        Validate constructor parameters.

        Ensure stride <= window_size to avoid gaps in coverage.
        Ensure window_size is reasonable (8 <= window_size <= 512).

        Raises:
            ValueError: If parameters are invalid.

        """
        if self.stride > self.window_size:
            raise ValueError(
                f"stride ({self.stride}) must be <= window_size "
                f"({self.window_size}) to ensure full image coverage",
            )

        if not (8 <= self.window_size <= 512):
            raise ValueError(
                f"window_size ({self.window_size}) must be in range [8, 512]",
            )

        if self.stride < 1:
            raise ValueError(f"stride ({self.stride}) must be >= 1")

        if self.aggregation_method not in ["majority_vote"]:
            raise ValueError(
                f"aggregation_method '{self.aggregation_method}' is not supported. "
                "Supported methods: 'majority_vote'",
            )

    def _compute_metrics(
        self,
        predicted_real_pairs: List[MaskPair],
    ) -> MetricsResultInterface:
        """Compute segmentation metrics and return a MetricsResultInterface."""
        num_classes = 3  # brittle, ductile, mixed

        total_pixels = 0
        correct_pixels = 0

        class_correct = np.zeros(num_classes, dtype=int)
        class_total = np.zeros(num_classes, dtype=int)
        intersection = np.zeros(num_classes, dtype=int)
        union = np.zeros(num_classes, dtype=int)
        pred_counts = np.zeros(num_classes, dtype=int)

        for pred, real in predicted_real_pairs:
            pred_flat = pred.flatten()
            real_flat = real.flatten()

            total_pixels += pred_flat.size
            correct_pixels += np.sum(pred_flat == real_flat)

            for cls in range(num_classes):
                pred_cls = pred_flat == cls
                real_cls = real_flat == cls

                class_correct[cls] += np.sum(pred_cls & real_cls)
                class_total[cls] += np.sum(real_cls)
                pred_counts[cls] += np.sum(pred_cls)

                intersection[cls] += np.sum(pred_cls & real_cls)
                union[cls] += np.sum(pred_cls | real_cls)

        # Pixel-level accuracy
        pixel_accuracy = (
            float(correct_pixels) / float(total_pixels) if total_pixels > 0 else 0.0
        )

        # Per-class accuracy (avoid division by zero)
        per_class_accuracy = [
            float(class_correct[c]) / float(class_total[c])
            if class_total[c] > 0
            else 0.0
            for c in range(num_classes)
        ]

        # Mean IoU
        mean_iou = float(
            np.mean(
                [
                    float(intersection[c]) / float(union[c]) if union[c] > 0 else 0.0
                    for c in range(num_classes)
                ],
            ),
        )

        # Precision, recall, F1-score per class
        precision_per_class = [
            float(intersection[c]) / float(pred_counts[c])
            if pred_counts[c] > 0
            else 0.0
            for c in range(num_classes)
        ]
        recall_per_class = [
            float(intersection[c]) / float(class_total[c])
            if class_total[c] > 0
            else 0.0
            for c in range(num_classes)
        ]
        f1_per_class = [
            (2 * precision_per_class[c] * recall_per_class[c])
            / (precision_per_class[c] + recall_per_class[c])
            if (precision_per_class[c] + recall_per_class[c]) > 0
            else 0.0
            for c in range(num_classes)
        ]
        mean_f1 = float(np.mean(f1_per_class))
        mean_precision = float(np.mean(precision_per_class))
        mean_recall = float(np.mean(recall_per_class))

        # Loss fallback
        loss = 1.0 - pixel_accuracy

        return MetricsResultInterface(
            accuracy=pixel_accuracy,
            loss=loss,
            iou=mean_iou,
            precision=mean_precision,
            recall=mean_recall,
            f1_score=mean_f1,
            additional_metrics={
                "per_class_accuracy": per_class_accuracy,
                "precision_per_class": precision_per_class,
                "recall_per_class": recall_per_class,
                "f1_per_class": f1_per_class,
            },
        )
