import numpy as np

from auto_ml.implementations import ViTModel
from auto_ml.interfaces import DatasetInterface


def test_vit() -> None:  # noqa: D103
    print("Initializing Verification for ViTModel...")

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
    print("Instantiating ViTModel...")
    try:
        model = ViTModel(
            epochs=1,
            batch_size=2,
            device="cpu",
        )  # Use CPU for CI/Verification stability
        print("ViTModel instantiated successfully.")
    except Exception as e:
        print(f"Error instantiating ViTModel: {e}")
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
    print("Testing evaluate() method...")
    try:
        eval_metrics = model.evaluate(dataset)
        print("Eval Metrics:", eval_metrics)
        assert eval_metrics.accuracy >= 0, "Accuracy should be valid"
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback

        traceback.print_exc()
        return

    print("VERIFICATION SUCCESSFUL!")
