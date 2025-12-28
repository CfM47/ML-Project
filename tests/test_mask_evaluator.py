"""Tests for AutoencoderMaskEvaluator and UMAPMaskEvaluator."""

import numpy as np
import pytest

from auto_ml.implementations import AutoencoderMaskEvaluator
from auto_ml.models.maskautoencoder.maskautoencoder import MaskAutoencoder


class TestMaskAutoencoder:
    """Tests for the MaskAutoencoder class."""

    def test_encoder_output_shape(self) -> None:
        """Test that encoder produces correct 3D output."""
        import torch

        model = MaskAutoencoder(latent_dim=3)

        x = torch.randn(1, 1, 512, 512)
        z = model.encode(x)

        assert z.shape == (1, 3), f"Expected (1, 3), got {z.shape}"

    def test_decoder_output_shape(self) -> None:
        """Test that decoder reconstructs to original dimensions."""
        import torch

        model = MaskAutoencoder(latent_dim=3)

        z = torch.randn(1, 3)
        recon = model.decode(z)

        expected_shape = (1, 1, 512, 512)
        assert recon.shape == expected_shape, (
            f"Expected {expected_shape}, got {recon.shape}"
        )

    def test_forward_pass(self) -> None:
        """Test full forward pass returns reconstruction and latent."""
        import torch

        model = MaskAutoencoder(latent_dim=3)

        x = torch.randn(1, 1, 512, 512)
        recon, z = model(x)

        assert recon.shape == x.shape
        assert z.shape == (1, 3)


class TestAutoencoderMaskEvaluator:
    """Tests for the AutoencoderMaskEvaluator class."""

    @pytest.fixture
    def sample_mask_pairs(self) -> list:
        """Create sample mask pairs for testing."""
        pairs = []
        for i in range(5):
            # Create similar predicted and real masks
            pred_mask = np.zeros((512, 512), dtype=np.uint8)
            real_mask = np.zeros((512, 512), dtype=np.uint8)

            # Draw circles
            center = (256 + i * 5, 256 + i * 5)
            radius = 100 + i * 10
            y, x = np.ogrid[:512, :512]
            dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            pred_mask[dist <= radius] = 1
            real_mask[dist <= radius + 5] = 1  # Slightly larger

            pairs.append((pred_mask, real_mask))
        return [pairs]  # List[List[MaskPair]]

    def test_initialization(self, sample_mask_pairs: list) -> None:
        """Test evaluator initializes correctly."""
        real_masks = [pair[1] for pair in sample_mask_pairs[0]]
        evaluator = AutoencoderMaskEvaluator(
            reference_masks=real_masks,
            device="cpu",
            epochs=2,
        )
        assert evaluator.latent_dim == 3
        assert evaluator.epochs == 2

    def test_evaluate_returns_float(self, sample_mask_pairs: list) -> None:
        """Test that evaluate returns a float ratio."""
        real_masks = [pair[1] for pair in sample_mask_pairs[0]]
        evaluator = AutoencoderMaskEvaluator(
            reference_masks=real_masks,
            device="cpu",
            epochs=5,  # Few epochs for speed
        )

        result = evaluator.evaluate(sample_mask_pairs)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_get_embeddings(self, sample_mask_pairs: list) -> None:
        """Test get_embeddings returns correct shape."""
        real_masks = [pair[1] for pair in sample_mask_pairs[0]]
        evaluator = AutoencoderMaskEvaluator(
            reference_masks=real_masks,
            device="cpu",
            epochs=5,
        )

        # Get embeddings for real masks
        embeddings = evaluator.get_embeddings(real_masks)

        assert embeddings.shape == (5, 3)
