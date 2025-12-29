"""Quadtree-based segmentation model implementation."""

from typing import List, Optional, Tuple

import numpy as np

from auto_ml.interfaces import (
    ClassificationModelInterface,
    ImageArray,
    MaskArray,
    MaskPair,
    MetricsResultInterface,
    SegmentationDatasetInterface,
    SegmentationModelInterface,
)


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
        threshold: float,
        min_region_size: int = 1,
        max_depth: Optional[int] = None,
    ) -> None:
        """
        Initialize the quadtree segmentation model.

        Args:
            classifier: Region classifier implementing
                        ClassificationModelInterface.
            threshold: Minimum confidence required to accept a region.
            min_region_size: Minimum width or height to allow subdivision.
            max_depth: Optional maximum recursion depth.

        """
        self.classifier = classifier
        self.threshold = threshold
        self.min_region_size = min_region_size
        self.max_depth = max_depth

    def train(self, dataset: SegmentationDatasetInterface) -> MetricsResultInterface:
        """
        Train the model on the provided dataset.

        Args:
            dataset: The training dataset.

        Returns:
            MetricsResultInterface containing training metrics.

        """
        return MetricsResultInterface()

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
