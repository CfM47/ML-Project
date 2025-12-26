from pathlib import Path

from auto_ml.implementations import (
    AccuracyEvaluator,
    DataAugmentatorNode,
    EvaluatorNode,
    IdentityAugmentator,
    ModelNode,
    ViTModel,
    load_dataset_from_directories,
)


def test_pipeline() -> None:  # noqa: D103
    print("=== Starting Full Pipeline Verification ===")

    # Paths
    base_dir = Path(".")
    input_dir = base_dir / "vega_3_tescan_unlabeled_images"
    target_dir = base_dir / "vega_3_tescan_labeled_images"

    # 1. Load Dataset
    print("\n--- Step 1: Loading Dataset ---")
    dataset = load_dataset_from_directories(input_dir, target_dir)

    if len(dataset) == 0:
        print("Error: No data loaded. Check paths and matching logic.")
        return

    # 2. Data Augmentation Node (Splitting)
    print("\n--- Step 2: Data Augmentation Node (Splitting & Augmenting) ---")
    # Using small k=2 for speed verification
    augmentator = IdentityAugmentator()
    data_node = DataAugmentatorNode(augmentator=augmentator, k_folds=2, random_seed=123)

    dataset_pairs = data_node.process(dataset)
    print(f"Generated {len(dataset_pairs)} dataset pairs (folds).")

    # 3. Model Node (Training)
    print("\n--- Step 3: Model Node (Training) ---")
    # ViTModel with small epochs for verification
    # Using 'cpu' or 'mps' if available.
    # Force cpu for CI-like stability if needed, but let's try auto.
    model = ViTModel(epochs=1, batch_size=2, device="cpu")
    model_node = ModelNode(model=model)

    # ModelNode.train() now returns List[List[MaskPair]]
    mask_pairs_result = model_node.train(dataset_pairs)

    print("\n--- Step 4: Verification Results ---")
    print(f"Number of folds: {len(mask_pairs_result)}")
    for i, fold_pairs in enumerate(mask_pairs_result):
        print(f"Fold {i + 1}: {len(fold_pairs)} mask pairs")
        assert len(fold_pairs) > 0, f"Fold {i + 1} should have mask pairs"

    # 4. Optional: Run evaluator on mask pairs
    evaluator_node = EvaluatorNode(
        evaluators={"accuracy": AccuracyEvaluator()},
        name="TestEvaluator",
    )
    eval_results = evaluator_node.evaluate(mask_pairs_result)
    print(f"Evaluation results: {eval_results}")
    assert "accuracy" in eval_results, "Should have accuracy result"

    print("\n=== PIPELINE VERIFICATION SUCCESSFUL! ===")

