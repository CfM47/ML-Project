
import os
import glob
from pathlib import Path
from PIL import Image
import numpy as np
import torch

from auto_ml.interfaces import DatasetInterface
from auto_ml.implementations import (
    load_dataset_from_directories,
    DataAugmentatorNode, 
    IdentityAugmentator, 
    ViTModel, 
    ModelNode
)

def run_pipeline_verification():
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
    model = ViTModel(epochs=10, batch_size=2, device="auto")
    model_node = ModelNode(model=model)
    
    results = model_node.train(dataset_pairs)
    
    print("\n--- Step 4: Verification Results ---")
    print(f"Mean Loss: {results['mean_loss']}")
    for i, res in enumerate(results['results']):
        print(f"Fold {i+1} Metrics: Loss={res.loss:.4f}, Accuracy={res.accuracy:.4f}")
        assert res.loss > 0, f"Fold {i+1} loss should be > 0"
    
    print("\n=== PIPELINE VERIFICATION SUCCESSFUL! ===")

if __name__ == "__main__":
    run_pipeline_verification()
