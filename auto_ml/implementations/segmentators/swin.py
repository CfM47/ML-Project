"""Swin Transformer model implementation for segmentation."""

import copy
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from auto_ml.implementations.segmentators.base import InMemoryPyTorchDataset
from auto_ml.interfaces import (
    MaskPair,
    MetricsResultInterface,
    SegmentationDatasetInterface,
    SegmentationModelInterface,
)
from auto_ml.models.swin.model import SwinSegmentation


class SwinModel(SegmentationModelInterface):
    """
    Swin Transformer Model implementation for AutoML.

    Wraps the SwinSegmentation model from swin.model.
    """

    def __init__(  # noqa: D107
        self,
        epochs: int = 10,
        batch_size: int = 4,
        lr: float = 1e-4,
        embed_dim: int = 96,
        depths: List[int] | None = None,
        num_heads: List[int] | None = None,
        window_size: List[int] | None = None,
        patience: int | None = None,
        device: str = "auto",
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience

        # Defaults for Swin-T if not provided
        self.embed_dim = embed_dim
        self.depths = depths if depths else [2, 2, 6, 2]
        self.num_heads = num_heads if num_heads else [3, 6, 12, 24]
        self.window_size = window_size if window_size else [7, 7]

        if device == "auto":
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = device

        self.model = SwinSegmentation(
            patch_size=[4, 4],  # Default fixed
            embed_dim=self.embed_dim,
            depths=self.depths,
            num_heads=self.num_heads,
            window_size=self.window_size,
            mlp_ratio=4.0,
            dropout=0.1,
            num_classes=3,
            channels=1,
        ).to(self.device)

    def train(
        self,
        dataset: SegmentationDatasetInterface,
        validation_dataset: SegmentationDatasetInterface | None = None,
    ) -> MetricsResultInterface:
        """Train the model."""
        pytorch_dataset = InMemoryPyTorchDataset(dataset)
        dataloader = DataLoader(
            pytorch_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Log training mode
        self._log_training_mode(validation_dataset)

        # Early stopping state
        best_val_loss = float("inf")
        best_model_state: dict | None = None
        patience_counter = 0

        total_loss: float = 0
        history: List[Dict[str, float]] = []
        epochs_trained = 0

        for epoch in range(self.epochs):
            epoch_loss = 0
            self.model.train()

            for inputs, masks in dataloader:
                inputs = inputs.to(self.device)
                masks = masks.to(self.device)

                # Check for channel mismatch (similar logic to ViTModel)
                if inputs.shape[1] != 1:
                    inputs = (
                        inputs[:, 0:1, :, :] * 0.299
                        + inputs[:, 1:2, :, :] * 0.587
                        + inputs[:, 2:3, :, :] * 0.114
                    )

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0
            total_loss = float(avg_loss)
            epochs_trained = epoch + 1

            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": float(avg_loss),
            }

            # Validation step
            if validation_dataset:
                avg_val_loss = self._compute_validation_loss(
                    validation_dataset,
                    criterion,
                )
                epoch_metrics["val_loss"] = avg_val_loss

                # Track best model (always when validation is provided)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                    status = "improved"
                else:
                    patience_counter += 1
                    if self.patience is not None:
                        status = f"no improvement ({patience_counter}/{self.patience})"
                    else:
                        status = "no improvement"

                print(
                    f"Epoch {epoch + 1}/{self.epochs}, "
                    f"Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
                    f"Status: {status}",
                )

                # Early stopping check (only when patience is set)
                if self.patience is not None and patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")

            history.append(epoch_metrics)

        # Restore best model if early stopping was used
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Restored best model (val_loss: {best_val_loss:.6f})")
            total_loss = best_val_loss

        return MetricsResultInterface(
            loss=total_loss,
            accuracy=0.0,
            additional_metrics={"epochs_trained": epochs_trained},
            history=history,
        )

    def _log_training_mode(
        self,
        validation_dataset: SegmentationDatasetInterface | None,
    ) -> None:
        """Log the training mode based on configuration."""
        if self.patience is not None and validation_dataset is not None:
            print(
                f"Training for up to {self.epochs} epochs "
                f"with early stopping (patience={self.patience})",
            )
        elif self.patience is not None and validation_dataset is None:
            print(
                f"Training for {self.epochs} epochs "
                f"(patience ignored: no validation dataset)",
            )
        else:
            print(f"Training for {self.epochs} epochs (no early stopping)")

    def _compute_validation_loss(
        self,
        validation_dataset: SegmentationDatasetInterface,
        criterion: nn.Module,
    ) -> float:
        """Compute average validation loss."""
        self.model.eval()
        val_dataset_torch = InMemoryPyTorchDataset(validation_dataset)
        val_loader = DataLoader(val_dataset_torch, batch_size=1, shuffle=False)
        val_loss = 0.0

        with torch.no_grad():
            for v_inputs, v_masks in val_loader:
                v_inputs = v_inputs.to(self.device)
                v_masks = v_masks.to(self.device)

                if v_inputs.shape[1] != 1:
                    v_inputs = (
                        v_inputs[:, 0:1, :, :] * 0.299
                        + v_inputs[:, 1:2, :, :] * 0.587
                        + v_inputs[:, 2:3, :, :] * 0.114
                    )

                v_outputs = self.model(v_inputs)
                v_loss = criterion(v_outputs, v_masks)
                val_loss += v_loss.item()

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        return float(avg_val_loss)

    def evaluate(self, dataset: SegmentationDatasetInterface) -> List[MaskPair]:
        """Evaluate the model and return predicted/real mask pairs."""
        pytorch_dataset = InMemoryPyTorchDataset(dataset)
        dataloader = DataLoader(pytorch_dataset, batch_size=1, shuffle=False)

        self.model.eval()
        mask_pairs: List[MaskPair] = []

        with torch.no_grad():
            for inputs, masks in dataloader:
                inputs = inputs.to(self.device)
                masks = masks.to(self.device)

                if inputs.shape[1] != 1:
                    inputs = (
                        inputs[:, 0:1, :, :] * 0.299
                        + inputs[:, 1:2, :, :] * 0.587
                        + inputs[:, 2:3, :, :] * 0.114
                    )

                outputs = self.model(inputs)

                # Get predicted mask
                predictions = torch.argmax(outputs, dim=1)
                predicted_mask = predictions.squeeze().cpu().numpy().astype(np.uint8)
                real_mask = masks.squeeze().cpu().numpy().astype(np.uint8)

                mask_pairs.append((predicted_mask, real_mask))

        return mask_pairs
