from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class TrainingContext:
    """A dataclass to hold all the stateful objects for a training process."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: nn.Module
    device: torch.device


@dataclass
class ValidationMetrics:
    """Model to hold and print validation metrics."""

    loss: float
    accuracy: float
    f1: float
    auroc: Any  # CANNOT be different (thank you python)

    def print_metrics(self, epoch: int, total_epochs: int, train_loss: float) -> None:
        """Print the validation metrics in a formatted string."""
        print(
            f"Epoch [{epoch}/{total_epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {self.loss:.4f} | "
            f"Val Acc: {self.accuracy:.4f} | "
            f"Val F1: {self.f1:.4f} | "
            f"Val AUC: {self.auroc:.4f}",
        )


@dataclass
class CrossValidationMetrics:
    """Model to hold and print cross-validation metrics."""

    train_percent: float
    train_loss: float
    val_loss: float

    def print_metrics(self) -> None:
        """Print the cross-validation metrics in a formatted string."""
        print(
            f"Train Percent: {self.train_percent:.2f} | "
            f"Train Loss: {self.train_loss:.4f} | "
            f"Val Loss: {self.val_loss:.4f}",
        )

    @staticmethod
    def summarize_results(results: list["CrossValidationMetrics"]) -> None:
        """Print a summary of the experimental results."""
        print("\n" + "=" * 50)
        print("--- Experiment Summary ---")
        print("=" * 50)
        print(f"{'Train %':<10} | {'Final Train Loss':<20} | {'Best Val Loss':<20}")
        print("-" * 50)
        for res in results:
            print(
                f"{res.train_percent:<10.0%} | "
                f"{res.train_loss:<20.4f} | "
                f"{res.val_loss:<20.4f}",
            )
        print("=" * 50)
