"""Test training history tracking."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from auto_ml.automl import AutoML
from auto_ml.implementations import (
    DataAugmentatorNode,
    IdentityAugmentator,
    ModelNode,
    ViTModel,
)
from auto_ml.interfaces import SegmentationDatasetInterface


class MockModel(ViTModel):
    """Mock model to test history tracking."""

    def __init__(self, epochs: int = 2) -> None:
        """Initialize Mock Model."""
        super().__init__(epochs=epochs, batch_size=2, device="cpu")


def test_automl_history_tracking() -> None:
    """Test that AutoML correctly captures training history."""
    # 1. Create Dummy Dataset
    print("Creating dummy dataset...")
    images = [
        np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8) for _ in range(10)
    ]
    masks = [
        np.random.randint(0, 3, (512, 512), dtype=np.uint8) for _ in range(10)
    ]
    dataset = SegmentationDatasetInterface.from_pairs(list(zip(images, masks)))

    # 2. Setup Nodes
    aug_node = DataAugmentatorNode(
        augmentator=IdentityAugmentator(),
        name="Aug_Test",
        k_folds=2,  # Use 2 folds to test list of histories
        random_seed=42,
    )

    # Use a small epoch count
    epochs = 3
    model = MockModel(epochs=epochs)
    model_node = ModelNode(model=model, name="ViT_Test")

    # 3. Run AutoML
    with TemporaryDirectory() as tmp_dir:
        automl = AutoML(cache_dir=Path(tmp_dir))
        # Verify results
        results = automl.run_experiment(dataset, [aug_node], [model_node])

        # 4. Verify Results
        assert "Aug_Test" in results
        assert "ViT_Test" in results["Aug_Test"]

        model_result = results["Aug_Test"]["ViT_Test"]

        # Check if history exists
        assert "training_history" in model_result
        history = model_result["training_history"]

        print(f"Captured History: {history}")

        # Should have 2 entries (one per fold)
        assert len(history) == 2

        # Check that losses are not identical
        # (suggests identical training path/weights reuse)
        # We initialized random seeds so training *should* be deterministic per fold
        # IF the data is the same, but folds have different data.
        loss_fold_0 = history[0][-1]["train_loss"]
        loss_fold_1 = history[1][-1]["train_loss"]
        # It's highly unlikely they are EXACTLY the same float
        # if trained on different data
        assert loss_fold_0 != loss_fold_1

        # Check content of history
        for fold_history in history:
            assert isinstance(fold_history, list)
            assert len(fold_history) == epochs

            for epoch_entry in fold_history:
                assert "epoch" in epoch_entry
                assert "train_loss" in epoch_entry
                assert "val_loss" in epoch_entry

    print("Test Passed!")


if __name__ == "__main__":
    test_automl_history_tracking()
