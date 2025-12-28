"""Autoencoder-based mask evaluator implementation."""

from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn

from auto_ml.interfaces import EvaluatorInterface, MaskArray, MaskPair
from auto_ml.models.maskautoencoder.maskautoencoder import MaskAutoencoder


class AutoencoderMaskEvaluator(EvaluatorInterface):
    """
    Autoencoder-based Mask Evaluator.

    Uses a convolutional autoencoder to embed masks into 3D space,
    trains One-Class SVM on reference masks, then evaluates how many
    predicted masks match the learned distribution.

    Workflow:
        1. Initialize with reference masks → trains autoencoder + SVM
        2. evaluate() classifies predicted masks against learned distribution

    Returns: float - ratio of predicted masks matching reference distribution (0.0-1.0)
    """

    def __init__(
        self,
        reference_masks: List[MaskArray],
        latent_dim: int = 3,
        epochs: int = 50,
        batch_size: int = 4,
        lr: float = 1e-3,
        nu: float = 0.1,
        kernel: str = "rbf",
        device: str = "auto",
    ) -> None:
        """
        Initialize and train the autoencoder evaluator.

        Args:
            reference_masks: List of 512x512 mask arrays defining the target class.
            latent_dim: Dimension of the latent space (default: 3).
            epochs: Number of training epochs for autoencoder.
            batch_size: Batch size for training.
            lr: Learning rate.
            nu: One-Class SVM nu parameter (proportion of outliers).
            kernel: One-Class SVM kernel type.
            device: Device to use ('auto', 'cpu', 'cuda', 'mps').

        """
        self.reference_masks = reference_masks
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.nu = nu
        self.kernel = kernel

        # Validate reference masks
        if len(reference_masks) < 2:
            raise ValueError("Need at least 2 reference masks for training")

        for i, mask in enumerate(reference_masks):
            if mask.shape != (512, 512):
                raise ValueError(
                    f"Reference mask {i} must be 512x512, got {mask.shape}",
                )

        # Setup device
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

        # Initialize and train autoencoder
        self.autoencoder = MaskAutoencoder(latent_dim=latent_dim).to(self.device)
        self._classifier: Optional[Any] = None

        # Train immediately
        self._train(reference_masks)

    def _mask_to_tensor(self, mask: MaskArray) -> torch.Tensor:
        """Convert mask array to normalized tensor."""
        mask_normalized = (
            mask.astype(np.float32) / mask.max()
            if mask.max() > 0
            else mask.astype(np.float32)
        )
        tensor = torch.from_numpy(mask_normalized).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def _train(self, masks: List[MaskArray]) -> None:
        """Train autoencoder and One-Class SVM on reference masks."""
        from sklearn.svm import OneClassSVM

        print(f"Training AutoencoderMaskEvaluator on {len(masks)} reference masks...")

        tensors = [self._mask_to_tensor(m).squeeze(0) for m in masks]
        dataset_tensor = torch.stack(tensors)

        self.autoencoder.train()
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        for epoch in range(self.epochs):
            total_loss = 0.0
            indices = torch.randperm(len(dataset_tensor))

            for i in range(0, len(dataset_tensor), self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                batch = dataset_tensor[batch_idx].to(self.device)

                optimizer.zero_grad()
                recon, _ = self.autoencoder(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 20 == 0 or epoch == 0:
                avg_loss = total_loss / max(1, len(dataset_tensor) / self.batch_size)
                print(f"  Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")

        # Extract embeddings and fit One-Class SVM
        self.autoencoder.eval()
        embeddings = []
        with torch.no_grad():
            for t in tensors:
                z = self.autoencoder.encode(t.unsqueeze(0).to(self.device))
                embeddings.append(z.cpu().numpy().squeeze())

        self._classifier = OneClassSVM(nu=self.nu, kernel=self.kernel)
        assert self._classifier is not None
        self._classifier.fit(np.array(embeddings))
        print("  Training complete.")

    def encode(self, mask: MaskArray) -> np.ndarray:
        """Encode a single mask to its 3D embedding."""
        self.autoencoder.eval()
        with torch.no_grad():
            tensor = self._mask_to_tensor(mask)
            z = self.autoencoder.encode(tensor)
            return z.cpu().numpy().squeeze().astype(np.float32)

    def evaluate(self, mask_pairs: List[List[MaskPair]]) -> float:
        """
        Evaluate mask pairs against the reference distribution.

        Args:
            mask_pairs: List of mask pair lists from ModelNode,
                       where each pair is (predicted_mask, real_mask).

        Returns:
            Float ratio (0.0-1.0) of predicted masks matching the reference
            distribution.

        """
        predicted_masks = []
        for fold_pairs in mask_pairs:
            for pred_mask, _ in fold_pairs:
                predicted_masks.append(pred_mask)

        if not predicted_masks:
            return 0.0

        # Classify predicted masks against reference distribution
        assert self._classifier is not None
        matches = 0
        for pred_mask in predicted_masks:
            embedding = self.encode(pred_mask)
            prediction = self._classifier.predict(embedding.reshape(1, -1))
            if prediction[0] == 1:  # Inlier
                matches += 1

        return matches / len(predicted_masks)

    def get_embeddings(self, masks: List[MaskArray]) -> np.ndarray:
        """Get 3D embeddings for a list of masks."""
        return np.array([self.encode(m) for m in masks])
