from pathlib import Path

from auto_ml.automl import AutoML
from auto_ml.implementations import (
    AccuracyEvaluator,
    AutoencoderMaskEvaluator,
    DataAugmentatorNode,
    EvaluatorNode,
    IdentityAugmentator,
    ModelNode,
    SwinModel,
    ViTModel,
    load_dataset_from_directories,
)
from setup.augmentators.setup import get_augmentator_nodes
from setup.evaluator.setup import get_evaluator_node
from setup.models.setup import get_model_nodes


def _run_automl() -> None:
    print("=== Starting AutoML Verification ===")

    # Paths
    base_dir = Path("pictures")
    input_dir = base_dir / "vega_3_tescan_unlabeled_images"
    target_dir = base_dir / "vega_3_tescan_labeled_images"

    # 1. Load Dataset
    print("\n--- Step 1: Loading Dataset ---")
    dataset = load_dataset_from_directories(input_dir, target_dir)

    if len(dataset) == 0:
        print("Error: No data loaded.")
        return

    # 2. Setup Nodes
    print("\n--- Step 2: Setting up Nodes ---")

    # Augmentators
    aug_node_1 = DataAugmentatorNode(
        augmentator=IdentityAugmentator(),
        name="Aug_Identity_K5",
        k_folds=5,
        random_seed=42,
    )

    aug_node_2 = DataAugmentatorNode(  # noqa: F841
        augmentator=IdentityAugmentator(),  # reusing identity for now
        name="Aug_Identity_K3",
        k_folds=3,
        random_seed=42,
    )

    augmentators = [aug_node_1, aug_node_2]

    # Models
    vit_model = ViTModel(epochs=2, batch_size=2, device="auto")
    swin_model = SwinModel(epochs=2, batch_size=2, device="auto")

    model_node_vit = ModelNode(model=vit_model, name="ViT_Model_Node")
    model_node_swin = ModelNode(model=swin_model, name="Swin_Model_Node")  # noqa: F841

    models = [model_node_vit]

    # Get reference masks from dataset for autoencoder training
    reference_masks = dataset.masks

    # Evaluator Node with named evaluators (including Autoencoder)
    evaluator_node = EvaluatorNode(
        evaluators={
            "accuracy": AccuracyEvaluator(),
            "mask_cohesion": AutoencoderMaskEvaluator(
                reference_masks=reference_masks,
                latent_dim=8,
                epochs=20,
                nu=0.5,
                device="auto",
            ),
        },
        name="MainEvaluator",
    )

    # 3. Run AutoML
    print("\n--- Step 3: Running AutoML Experiment ---")
    automl = AutoML()
    automl.run_experiment(dataset, augmentators, models, evaluator_node=evaluator_node)

    # 4. Results
    print("\n--- Step 4: Summary ---")
    print(automl.get_summary())

    print("\n=== AUTOML VERIFICATION SUCCESSFUL! ===")


def _run_with_setup(
    unlabeled_dir: str | Path,
    labeled_dir: str | Path,
    classification_dataset_dir: str | Path,
    auto_ml_cache_dir: str | Path,
    augmentator_indices: list[int] | None = None,
    model_indices: list[int] | None = None,
) -> None:
    """
    Use setup to run Auto-ML.

    Args:
        unlabeled_dir: Directory containing unlabeled images.
        labeled_dir: Directory containing labeled images.
        classification_dataset_dir: Directory for classification dataset.
        auto_ml_cache_dir: Directory for AutoML cache.
        augmentator_indices: Optional list of indices to filter augmentator nodes.
            If None, all augmentators are used.
        model_indices: Optional list of indices to filter model nodes.
            If None, all models are used.

    """
    unlabeled_dir = Path(unlabeled_dir)
    labeled_dir = Path(labeled_dir)
    classification_dataset_dir = Path(classification_dataset_dir)
    auto_ml_cache_dir = Path(auto_ml_cache_dir)

    dataset = load_dataset_from_directories(Path(unlabeled_dir), labeled_dir)

    augmentators = get_augmentator_nodes()
    if augmentator_indices is not None:
        augmentators = [augmentators[i] for i in augmentator_indices]

    models = get_model_nodes(classification_dataset_dir)
    if model_indices is not None:
        models = [models[i] for i in model_indices]

    evaluator = get_evaluator_node(dataset)

    automl = AutoML(cache_dir=auto_ml_cache_dir)
    automl.run_experiment(dataset, augmentators, models, evaluator_node=evaluator)


if __name__ == "__main__":
    _run_automl()
