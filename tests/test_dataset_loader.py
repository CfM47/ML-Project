import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from auto_ml.implementations import load_dataset_from_directories


def test_dataset_loader() -> None:  # noqa: D103
    print("Initializing Verification for Dataset Loader...")

    # 1. Setup Temporary Directories
    input_dir = Path("temp_input")
    target_dir = Path("temp_target")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    try:
        # 2. Create Dummy Images
        print("Creating dummy images...")
        size = (512, 512)

        # Sample 1: Red Dominant (Class 0)
        img1 = Image.new("L", size, color=100)
        img1.save(os.path.join(input_dir, "sample1.png"))

        tgt1 = Image.new("RGB", size, color=(200, 50, 50))  # Red
        tgt1.save(os.path.join(target_dir, "sample1.png"))

        # Sample 2: Green Dominant (Class 1)
        img2 = Image.new("L", size, color=150)
        img2.save(os.path.join(input_dir, "sample2.jpg"))

        tgt2 = Image.new("RGB", size, color=(50, 200, 50))  # Green
        tgt2.save(os.path.join(target_dir, "sample2.jpg"))

        # Sample 3: Misc/Background (Class 2)
        img3 = Image.new("L", size, color=200)
        img3.save(os.path.join(input_dir, "sample3.tiff"))

        tgt3 = Image.new("RGB", size, color=(50, 50, 50))  # Dark
        tgt3.save(os.path.join(target_dir, "sample3.tiff"))

        # Unmatched Sample
        img4 = Image.new("L", size, color=200)
        img4.save(os.path.join(input_dir, "sample4.png"))

        # 3. Test Loading
        print("Loading dataset...")
        dataset = load_dataset_from_directories(input_dir, target_dir)

        print(f"Loaded {len(dataset)} samples.")
        assert len(dataset) == 3, f"Expected 3 samples, got {len(dataset)}"

        # 4. Verify Content
        print("Verifying content...")

        # Check Sample 1 (Red -> 0)
        _, out1 = dataset[0]
        unique1 = np.unique(out1.mask)
        print(f"Sample 1 Unique Mask Values: {unique1}")
        assert 0 in unique1, "Sample 1 should contain class 0 (Red)"

        # Check Sample 2 (Green -> 1)
        _, out2 = dataset[1]
        unique2 = np.unique(out2.mask)
        print(f"Sample 2 Unique Mask Values: {unique2}")
        assert 1 in unique2, "Sample 2 should contain class 1 (Green)"

        # Check Sample 3 (Background -> 2)
        _, out3 = dataset[2]
        unique3 = np.unique(out3.mask)
        print(f"Sample 3 Unique Mask Values: {unique3}")
        assert 2 in unique3, "Sample 3 should contain class 2 (Background)"

        print("VERIFICATION SUCCESSFUL!")

    finally:
        # Cleanup
        if os.path.exists(input_dir):
            shutil.rmtree(input_dir)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
