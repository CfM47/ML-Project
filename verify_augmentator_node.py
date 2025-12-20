
import numpy as np
from auto_ml.interfaces import DatasetInterface
from auto_ml.implementations import DataAugmentatorNode, IdentityAugmentator

def run_verification():
    print("Initializing Verification for DataAugmentatorNode...")
    
    # 1. Create dummy dataset (100 samples)
    print("Creating dummy dataset...")
    image = np.zeros((512, 512, 3), dtype=np.uint8) # Dummy image
    mask = np.zeros((512, 512), dtype=np.uint8)     # Dummy mask
    
    dataset = DatasetInterface()
    for _ in range(100):
        dataset.add_sample(image, mask)
        
    print(f"Dataset created with {len(dataset)} samples.")
    
    # 2. Test K-Folds = 5 (default)
    print("\nTesting K=5 Folds...")
    node_k5 = DataAugmentatorNode(IdentityAugmentator(), k_folds=5)
    splits_k5 = node_k5.process(dataset)
    
    print(f"Number of splits: {len(splits_k5)}")
    assert len(splits_k5) == 5, f"Expected 5 splits, got {len(splits_k5)}"
    
    for i, (train, val) in enumerate(splits_k5):
        print(f"  Split {i+1}: Train={len(train)}, Val={len(val)}")
        assert len(train) == 80, f"Expected 80 train samples, got {len(train)}"
        assert len(val) == 20, f"Expected 20 val samples, got {len(val)}"
        
    # 3. Test K-Folds = 1 (Single Split)
    print("\nTesting K=1 Fold (Single Split)...")
    node_k1 = DataAugmentatorNode(IdentityAugmentator(), k_folds=1, test_size=0.3)
    splits_k1 = node_k1.process(dataset)
    
    print(f"Number of splits: {len(splits_k1)}")
    assert len(splits_k1) == 1, f"Expected 1 split, got {len(splits_k1)}"
    
    train, val = splits_k1[0]
    print(f"  Split 1: Train={len(train)}, Val={len(val)}")
    assert len(train) == 70, f"Expected 70 train samples, got {len(train)}"
    assert len(val) == 30, f"Expected 30 val samples, got {len(val)}"
    
    print("\nVERIFICATION SUCCESSFUL!")

if __name__ == "__main__":
    run_verification()
