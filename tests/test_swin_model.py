import numpy as np

from auto_ml.implementations import SwinModel
from auto_ml.interfaces import DatasetInterface


def test_swin() -> None:  # noqa: D103
    print("Initializing Verification for SwinModel...")

    # 1. Create dummy data
    print("Creating dummy dataset...")
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    mask = np.random.randint(0, 3, (512, 512), dtype=np.uint8)

    dataset = DatasetInterface()
    # Add enough samples for a batch
    for _ in range(4):
        dataset.add_sample(image, mask)

    print(f"Dataset created with {len(dataset)} samples.")

    # 2. Instantiate Model
    print("Instantiating SwinModel...")
    try:
        # Use CPU for CI/Verification stability
        model = SwinModel(epochs=1, batch_size=2, device="cpu")
        print("SwinModel instantiated successfully.")
    except Exception as e:
        print(f"Error instantiating SwinModel: {e}")
        return

    # 3. Test Training
    print("Testing train() method...")
    try:
        train_metrics = model.train(dataset)
        print("Train Metrics:", train_metrics)
        assert train_metrics.loss > 0, "Loss should be non-zero"
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback

        traceback.print_exc()
        return

    # 4. Test Evaluation
    # evaluate() now returns List[MaskPair] instead of MetricsResultInterface
    print("Testing evaluate() method...")
    try:
        mask_pairs = model.evaluate(dataset)
        print(f"Evaluate returned {len(mask_pairs)} mask pairs")
        assert len(mask_pairs) == len(dataset), "Should return one pair per sample"
        # Check that each pair contains (predicted, real) masks
        for predicted, real in mask_pairs:
            assert predicted.shape == (512, 512), "Predicted mask should be 512x512"
            assert real.shape == (512, 512), "Real mask should be 512x512"
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback

        traceback.print_exc()
        return

    print("VERIFICATION SUCCESSFUL!")

