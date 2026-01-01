"""Quadtree-based segmentation model implementation."""


from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import math
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


class _BestParams(TypedDict):
    threshold: float
    min_region_size: int
    max_depth: Optional[int]


class QuadtreeSegmentationModel(SegmentationModelInterface):
    """
    Quadtree-based image segmentation model.

    The model recursively classifies image regions using an injected
    ClassificationModelInterface. Regions with confidence below a
    threshold are subdivided into four quadrants.
    """

    def __init__(
        self,
        classifier: ClassificationModelInterface,
        classifier_dataset_dir: Optional[Path],
        threshold: float,
        min_region_size: int = 1,
        max_depth: Optional[int] = None,
        optimize_metric: Optional[str] = None,
        search_space: Optional[Dict[str, Tuple[Any, Any]]] = None,
        n_trials: int = 20,
    ) -> None:
        """
        Initialize the quadtree segmentation model.

        Args:
            classifier: Region classifier implementing
                        ClassificationModelInterface.
            classifier_dataset_dir: Directory containing dataset for training
                                    the classifier. If None, assumes classifier
                                    is already trained, will throw error if train()
                                    is called.
            threshold: Minimum confidence required to accept a region.
            min_region_size: Minimum width or height to allow subdivision.
            max_depth: Optional maximum recursion depth.
            optimize_metric: Whether to optimize hyperparameters by maximizing a metric.
            search_space: Optional dictionary defining the search ranges for
                          hyperparameters.
                          - 'threshold': (min, max) float
                          - 'min_region_size': (min, max) int
                          - 'max_depth': (min, max) int
                          If None, default ranges are used.
            n_trials: Number of random trials to perform during hyperparameter
                      tuning. Defaults to 20.
        """
        self.classifier = classifier
        self.threshold = threshold
        self.min_region_size = min_region_size
        self.max_depth = max_depth
        self.classifier_dataset_dir = classifier_dataset_dir
        self.optimize_metric = optimize_metric
        self.search_space = search_space
        self.n_trials = n_trials

    def train(self, dataset: SegmentationDatasetInterface) -> MetricsResultInterface:
        """
        Train the quadtree segmenter. Optionally performs hyperparameter tuning.

        Args:
            dataset: Segmentation dataset for training and tuning.

        Returns:
            MetricsResultInterface containing segmentation quality metrics.

        """
        if self.classifier_dataset_dir is not None:
            classifier_dataset = load_classification_dataset_from_dir(
                self.classifier_dataset_dir,
            )
            self.classifier.train(classifier_dataset)
        # else, assume classifier is already trained,
        # this is needed for tests that check this method not to break

        if not self.optimize_metric:
            predicted_real_pairs = self.evaluate(dataset)
            metrics = self._compute_metrics(predicted_real_pairs)
            return metrics

        # hyperparameter tunning
        if self.search_space:
            threshold_range = self.search_space.get("threshold", (0.5, 0.9))
            min_region_range = self.search_space.get("min_region_size", (4, 16))
            max_depth_range = self.search_space.get("max_depth", (4, 8))
        else:
            # simple default ranges
            threshold_range = (0.5, 0.9)
            min_region_range = (4, 16)
            max_depth_range = (4, 8)

        # Split dataset in train/val just for tunning
        n = len(dataset)
        val_ratio = 0.2
        val_size = int(n * val_ratio)
        indices = np.arange(n)
        np.random.shuffle(indices)
        val_indices = indices[:val_size]
        val_dataset = SegmentationDatasetInterface(
            [dataset.samples[i] for i in val_indices],
        )

        # Simulated Annealing Configuration
        ranges = {
            "threshold": threshold_range,
            "min_region_size": min_region_range,
            "max_depth": max_depth_range,
        }
        
        # Initial Solution (Random Start)
        current_threshold = np.random.uniform(threshold_range[0], threshold_range[1])
        current_min_size = np.random.randint(min_region_range[0], min_region_range[1] + 1)
        current_max_depth = np.random.randint(max_depth_range[0], max_depth_range[1] + 1)

        current_params: _BestParams = {
            "threshold": current_threshold,
            "min_region_size": current_min_size,
            "max_depth": current_max_depth,
        }
        
        # We need an initial metric for SA to compare against
        self.threshold = current_threshold
        self.min_region_size = current_min_size
        self.max_depth = current_max_depth
        
        initial_pairs = self.evaluate(val_dataset)
        initial_metrics = self._compute_metrics(initial_pairs)
        current_metric_val = -1.0
        
        if self.optimize_metric in initial_metrics.to_dict():
             val = initial_metrics.to_dict()[self.optimize_metric]
             if isinstance(val, (int, float)):
                 current_metric_val = val

        best_params = current_params.copy()
        best_metric = current_metric_val

        # SA Hyperparameters
        temp = 1.0
        alpha = 0.90 # Cooling rate
        
        # SA Loop
        for i in range(self.n_trials):
            # Generate Neighbor
            neighbor_params = self._get_neighbor(current_params, ranges)
            
            # Evaluate Neighbor
            self.threshold = neighbor_params["threshold"]
            self.min_region_size = neighbor_params["min_region_size"]
            self.max_depth = neighbor_params["max_depth"]

            predicted_real_pairs = self.evaluate(val_dataset)
            metrics = self._compute_metrics(predicted_real_pairs)
            metrics_dict = metrics.to_dict()

            if self.optimize_metric not in metrics_dict:
                break
                
            neighbor_metric_val = metrics_dict[self.optimize_metric]
            if not isinstance(neighbor_metric_val, (int, float)):
                 break

            # Acceptance Probability (Maximization)
            # if neighbor is better, prob > 1, algorithm accepts
            # if worse, prob < 1, algorithm accepts with probability
            delta = neighbor_metric_val - current_metric_val
            
            if delta > 0:
                acceptance_prob = 2.0 # Always accept
            else:
                # Avoid overflow/underflow if temp is too low or delta too neg
                try:
                    acceptance_prob = math.exp(delta / temp)
                except OverflowError:
                    acceptance_prob = 0.0

            if delta > 0 or np.random.rand() < acceptance_prob:
                current_params = neighbor_params
                current_metric_val = neighbor_metric_val
            
            # Keep track of global best
            if current_metric_val > best_metric:
                best_metric = current_metric_val
                best_params = current_params.copy()
            
            # Cool down
            temp *= alpha

        # update with best configuration
        self.threshold = best_params["threshold"]
        self.min_region_size = best_params["min_region_size"]
        self.max_depth = best_params["max_depth"]

        # evaluate over the whole dataset
        final_pairs = self.evaluate(dataset)
        final_metric = self._compute_metrics(final_pairs)

        return final_metric

    def evaluate(self, dataset: SegmentationDatasetInterface) -> List[MaskPair]:
        """
        Evaluate the model on a dataset.

        For each image, a segmentation mask is produced using recursive
        quadtree decomposition.

        Returns:
            List of (predicted_mask, real_mask) tuples.

        """
        return [(self._segment_image(image), real_mask) for image, real_mask in dataset]

    def _segment_image(
        self,
        image: ImageArray,
    ) -> MaskArray:
        """Segment a single image using recursive quadtree splitting."""
        mask = np.zeros((512, 512), dtype=np.uint8)

        self._segment_region(
            image=image,
            mask=mask,
            x=0,
            y=0,
            width=512,
            height=512,
            depth=0,
        )

        return mask

    def _segment_region(
        self,
        image: ImageArray,
        mask: MaskArray,
        x: int,
        y: int,
        width: int,
        height: int,
        depth: int,
    ) -> None:
        """
        Recursively segment a rectangular region of the image.

        If the classifier confidence is sufficient, the region is
        filled in the mask. Otherwise, the region is subdivided
        into four quadrants and processed recursively.
        """
        label, confidence = self.classifier.classify(
            image=image,
            x=x,
            y=y,
            width=width,
            height=height,
        )

        if self._should_stop_recursion(confidence, width, height, depth):
            mask[y : y + height, x : x + width] = label
            return

        # Subdivide region (integer division allowed)
        w_half = width // 2
        h_half = height // 2

        # Ensure progress (should not happen if min_region_size >= 1)
        if w_half == 0 or h_half == 0:
            mask[y : y + height, x : x + width] = label
            return

        regions: List[Tuple[int, int, int, int]] = [
            (x, y, w_half, h_half),  # top left
            (x + w_half, y, w_half, h_half),  # top right
            (x, y + h_half, w_half, h_half),  # bottom left
            (x + w_half, y + h_half, w_half, h_half),  # bottom right
        ]

        for xr, yr, wr, hr in regions:
            self._segment_region(
                image=image,
                mask=mask,
                x=xr,
                y=yr,
                width=wr,
                height=hr,
                depth=depth + 1,
            )

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def _get_neighbor(
        self,
        params: _BestParams,
        ranges: Dict[str, Tuple[Any, Any]],
    ) -> _BestParams:
        """
        Generate a neighbor configuration by perturbing current parameters.
        """
        new_params = params.copy()

        # Perturb one parameter at a time or all? Let's perturb all slightly.
        
        # Threshold: Perturb by normal noise stride 0.05
        t_range = ranges["threshold"]
        t_current = new_params["threshold"]
        t_new = t_current + np.random.normal(0, 0.05)
        new_params["threshold"] = np.clip(t_new, t_range[0], t_range[1])

        # Min Region: Perturb by +/- 1 or Stay
        mr_range = ranges["min_region_size"]
        mr_current = new_params["min_region_size"]
        mr_step = np.random.randint(-2, 3) # -2, -1, 0, 1, 2
        mr_new = mr_current + mr_step
        new_params["min_region_size"] = int(
            np.clip(mr_new, mr_range[0], mr_range[1])
        )

        # Max Depth: Perturb by +/- 1
        md_range = ranges["max_depth"]
        if new_params["max_depth"] is not None:
             md_current = new_params["max_depth"]
             md_step = np.random.randint(-1, 2)
             md_new = md_current + md_step
             new_params["max_depth"] = int(np.clip(md_new, md_range[0], md_range[1]))
        
        return new_params

    def _should_stop_recursion(
        self,
        confidence: float,
        width: int,
        height: int,
        depth: int,
    ) -> bool:
        """
        Determine whether recursion should stop.

        Determine whether recursion should stop based on region size and maximum
        depth constraints.
        """
        return (
            confidence >= self.threshold
            or width <= self.min_region_size
            or height <= self.min_region_size
            or (self.max_depth is not None and depth >= self.max_depth)
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
