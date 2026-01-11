"""Metrics dataclasses for tracking training and validation results."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from auto_ml.implementations.segmentators.swin import SwinModel
    from auto_ml.interfaces import MaskPair


@dataclass
class TrainingHistory:
    """Store validated training history with per-epoch losses."""

    epochs: List[int]
    train_losses: List[float]
    val_losses: List[float]

    @classmethod
    def from_history_dicts(
        cls,
        history: List[Dict[str, float]],
    ) -> "TrainingHistory | None":
        """
        Create TrainingHistory from list of epoch metric dicts.

        Return None and log error if history is invalid (missing train_loss
        or val_loss for any epoch).

        Args:
            history: List of dicts with 'epoch', 'train_loss', and 'val_loss' keys.

        Returns:
            TrainingHistory instance, or None if validation fails.

        """
        if not history:
            print("Error: Training history is empty")
            return None

        epochs = []
        train_losses = []
        val_losses = []

        for i, epoch_dict in enumerate(history):
            epoch_num = int(epoch_dict.get("epoch", i + 1))

            if "train_loss" not in epoch_dict:
                print(f"Error: Missing train_loss for epoch {epoch_num}")
                return None
            if "val_loss" not in epoch_dict:
                print(f"Error: Missing val_loss for epoch {epoch_num}")
                return None

            epochs.append(epoch_num)
            train_losses.append(float(epoch_dict["train_loss"]))
            val_losses.append(float(epoch_dict["val_loss"]))

        return cls(
            epochs=epochs,
            train_losses=train_losses,
            val_losses=val_losses,
        )


@dataclass
class FoldMetrics:
    """Store metrics for a single fold."""

    fold: int
    train_history: List[Dict[str, float]] = field(default_factory=list)

    # All metrics from evaluator
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def dice_macro(self) -> float:
        """
        Return Dice macro metric for backward compatibility.

        Raises:
            KeyError: If Dice_Macro metric not found.

        """
        if "Dice_Macro" not in self.metrics:
            raise KeyError("Metric 'Dice_Macro' not found in fold metrics")
        return self.metrics["Dice_Macro"]

    @property
    def accuracy(self) -> float:
        """
        Return Accuracy metric for backward compatibility.

        Raises:
            KeyError: If Accuracy metric not found.

        """
        if "Accuracy" not in self.metrics:
            raise KeyError("Metric 'Accuracy' not found in fold metrics")
        return self.metrics["Accuracy"]

    @property
    def final_train_loss(self) -> float:
        """Return the final training loss."""
        if not self.train_history:
            return float("inf")
        return self.train_history[-1].get("train_loss", float("inf"))

    @property
    def final_val_loss(self) -> float:
        """Return the final validation loss."""
        if not self.train_history:
            return float("inf")
        return self.train_history[-1].get("val_loss", float("inf"))

    @property
    def train_losses(self) -> List[float]:
        """Return list of training losses per epoch."""
        return [h.get("train_loss", float("inf")) for h in self.train_history]

    @property
    def val_losses(self) -> List[float]:
        """Return list of validation losses per epoch."""
        return [h.get("val_loss", float("inf")) for h in self.train_history]


@dataclass
class PercentageMetrics:
    """Store aggregated metrics for a training percentage."""

    percentage: int
    fold_metrics: List[FoldMetrics] = field(default_factory=list)

    # --- Dynamic Metric Access ---

    def get_metric_mean(self, metric_name: str) -> float:
        """
        Calculate mean of a specific metric across folds.

        Args:
            metric_name: Name of the metric (e.g., "Dice_Macro", "IoU_Class1").

        Returns:
            Mean value across folds.

        Raises:
            ValueError: If no fold metrics available.
            KeyError: If metric not found in any fold.

        """
        if not self.fold_metrics:
            raise ValueError("No fold metrics available")
        for fm in self.fold_metrics:
            if metric_name not in fm.metrics:
                raise KeyError(
                    f"Metric '{metric_name}' not found in fold {fm.fold} metrics",
                )
        values = [fm.metrics[metric_name] for fm in self.fold_metrics]
        return float(np.mean(values))

    def get_metric_std(self, metric_name: str) -> float:
        """
        Calculate std of a specific metric across folds.

        Args:
            metric_name: Name of the metric (e.g., "Dice_Macro", "IoU_Class1").

        Returns:
            Standard deviation across folds.

        Raises:
            ValueError: If no fold metrics available.
            KeyError: If metric not found in any fold.

        """
        if not self.fold_metrics:
            raise ValueError("No fold metrics available")
        for fm in self.fold_metrics:
            if metric_name not in fm.metrics:
                raise KeyError(
                    f"Metric '{metric_name}' not found in fold {fm.fold} metrics",
                )
        values = [fm.metrics[metric_name] for fm in self.fold_metrics]
        return float(np.std(values))

    def get_all_metric_names(self) -> List[str]:
        """
        Return all metric names available in the fold metrics.

        Returns:
            List of metric names.

        Raises:
            ValueError: If no fold metrics available.

        """
        if not self.fold_metrics:
            raise ValueError("No fold metrics available")
        return list(self.fold_metrics[0].metrics.keys())

    # --- Dice Macro (F1) - Backward Compatibility ---

    @property
    def mean_dice_macro(self) -> float:
        """Calculate mean Dice macro across folds."""
        return self.get_metric_mean("Dice_Macro")

    @property
    def std_dice_macro(self) -> float:
        """Calculate std of Dice macro across folds."""
        return self.get_metric_std("Dice_Macro")

    # --- Accuracy - Backward Compatibility ---

    @property
    def mean_accuracy(self) -> float:
        """Calculate mean accuracy across folds."""
        return self.get_metric_mean("Accuracy")

    @property
    def std_accuracy(self) -> float:
        """Calculate std of accuracy across folds."""
        return self.get_metric_std("Accuracy")

    # --- Training Loss ---

    @property
    def mean_final_train_loss(self) -> float:
        """Calculate mean final training loss across folds."""
        if not self.fold_metrics:
            return float("inf")
        return float(np.mean([fm.final_train_loss for fm in self.fold_metrics]))

    @property
    def std_final_train_loss(self) -> float:
        """Calculate std of final training loss across folds."""
        if not self.fold_metrics:
            return 0.0
        return float(np.std([fm.final_train_loss for fm in self.fold_metrics]))

    # --- Validation Loss ---

    @property
    def mean_final_val_loss(self) -> float:
        """Calculate mean final validation loss across folds."""
        if not self.fold_metrics:
            return float("inf")
        return float(np.mean([fm.final_val_loss for fm in self.fold_metrics]))

    @property
    def std_final_val_loss(self) -> float:
        """Calculate std of final validation loss across folds."""
        if not self.fold_metrics:
            return 0.0
        return float(np.std([fm.final_val_loss for fm in self.fold_metrics]))


@dataclass
class SubsetMetrics:
    """Store leave-two-out evaluation metrics."""

    subset_size: int
    num_subsets: int

    # Full test set metrics: {"Accuracy": 0.85, ...}
    full_metrics: Dict[str, float]

    # Mean across subsets: {"Accuracy": 0.84, ...}
    subset_means: Dict[str, float]

    # Std across subsets: {"Accuracy": 0.02, ...}
    subset_stds: Dict[str, float]

    # Full distributions: {"Accuracy": [0.84, 0.86, ...], ...}
    subset_distributions: Dict[str, List[float]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "subset_size": self.subset_size,
            "num_subsets": self.num_subsets,
            "full_metrics": self.full_metrics,
            "subset_means": self.subset_means,
            "subset_stds": self.subset_stds,
            "subset_distributions": self.subset_distributions,
        }


@dataclass
class TrainingResult:
    """Store all outputs from final training."""

    model: "SwinModel"
    test_metrics: SubsetMetrics
    mask_pairs: List["MaskPair"]
    predictions_figure: "Figure"
    loss_curves_figure: "Figure | None"
    histograms: Dict[str, "Figure"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary (excludes non-serializable objects)."""
        return {
            "test_metrics": self.test_metrics.to_dict(),
        }
