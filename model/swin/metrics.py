"""Metrics dataclasses for tracking training and validation results."""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


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
