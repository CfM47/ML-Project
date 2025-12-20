
from pathlib import Path
from auto_ml.implementations import (
    load_dataset_from_directories,
    DataAugmentatorNode,
    IdentityAugmentator,
    ViTModel,
    SwinModel,
    ModelNode
)
from auto_ml.automl import AutoML

def run_automl_verification():
    print("=== Starting AutoML Verification ===")
    
    # Paths
    base_dir = Path(".")
    input_dir = base_dir / "vega_3_tescan_unlabeled_images"
    target_dir = base_dir / "vega_3_tescan_labeled_images"
    
    # 1. Load Dataset
    print("\n--- Step 1: Loading Dataset ---")
    dataset = load_dataset_from_directories(str(input_dir), str(target_dir))
    
    if len(dataset) == 0:
        print("Error: No data loaded.")
        return

    # 2. Setup Nodes
    print("\n--- Step 2: Setting up Nodes ---")
    
    # Augmentators
    # We can create two identical augmentators just to test the graph logic
    # In reality one would be Rotated, one Scaled etc.
    aug_node_1 = DataAugmentatorNode(
        augmentator=IdentityAugmentator(), 
        name="Aug_Identity_K5",
        k_folds=5,
        random_seed=42
    )
    
    aug_node_2 = DataAugmentatorNode(
        augmentator=IdentityAugmentator(), # reusing identity for now
        name="Aug_Identity_K3",
        k_folds=3,
        random_seed=42
    )
    
    augmentators = [aug_node_1, aug_node_2]
    
    # Models
    # Swin and ViT
    vit_model = ViTModel(epochs=2, batch_size=2, device="auto")
    swin_model = SwinModel(epochs=2, batch_size=2, device="auto")
    
    model_node_vit = ModelNode(model=vit_model, name="ViT_Model_Node")
    model_node_swin = ModelNode(model=swin_model, name="Swin_Model_Node")
    
    models = [model_node_vit, model_node_swin]
    
    # 3. Run AutoML
    print("\n--- Step 3: Running AutoML Experiment ---")
    automl = AutoML()
    results = automl.run_experiment(dataset, augmentators, models)
    
    # 4. Results
    print("\n--- Step 4: Summary ---")
    print(automl.get_summary())
    
    print("\n=== AUTOML VERIFICATION SUCCESSFUL! ===")

if __name__ == "__main__":
    run_automl_verification()
