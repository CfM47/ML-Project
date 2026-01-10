"""Metrics dataclasses for tracking training and validation results."""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


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

    # Final metrics from evaluator
    dice_macro: float = 0.0
    accuracy: float = 0.0

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

    # --- Dice Macro (F1) ---

    @property
    def mean_dice_macro(self) -> float:
        """Calculate mean Dice macro across folds."""
        if not self.fold_metrics:
            return 0.0
        return float(np.mean([fm.dice_macro for fm in self.fold_metrics]))

    @property
    def std_dice_macro(self) -> float:
        """Calculate std of Dice macro across folds."""
        if not self.fold_metrics:
            return 0.0
        return float(np.std([fm.dice_macro for fm in self.fold_metrics]))

    # --- Accuracy ---

    @property
    def mean_accuracy(self) -> float:
        """Calculate mean accuracy across folds."""
        if not self.fold_metrics:
            return 0.0
        return float(np.mean([fm.accuracy for fm in self.fold_metrics]))

    @property
    def std_accuracy(self) -> float:
        """Calculate std of accuracy across folds."""
        if not self.fold_metrics:
            return 0.0
        return float(np.std([fm.accuracy for fm in self.fold_metrics]))

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
