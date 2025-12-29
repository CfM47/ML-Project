"""Dataset loading utilities."""

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from auto_ml.interfaces import SegmentationDatasetInterface


def load_dataset_from_directories(
    input_dir: Path,
    target_dir: Path,
    target_size: Tuple[int, int] = (512, 512),
) -> SegmentationDatasetInterface:
    """
    Load dataset from input and target directories.

    Args:
        input_dir: Directory containing input images.
        target_dir: Directory containing target (labeled) images.
        target_size: tuple (height, width) to resize images to.

    Returns:
        Populated DatasetInterface.

    """
    print(f"Loading from:\n  Input: {input_dir}\n  Target: {target_dir}")

    input_path = Path(input_dir)
    target_path = Path(target_dir)

    # Get all input files
    input_files = sorted(
        [
            f
            for f in input_path.glob("*")
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
        ],
    )

    # Get all target files
    target_files = sorted(
        [
            f
            for f in target_path.glob("*")
            if f.suffix.lower() in [".png"] and "_labeled" in f.name
        ],
    )

    dataset = SegmentationDatasetInterface()

    # Create valid pairs
    # Heuristic: Target stem should start with Input stem
    # Or more robust: Target name is Input stem + "_labeled"

    # Let's map normalized stems to consistency
    target_map = {}
    for t in target_files:
        # standard format: "name_labeled.png" -> key: "name"
        # Handle "  _labeled" or "_labeled"
        stem = t.stem
        if stem.endswith("_labeled"):
            key = stem[:-8].strip()  # remove _labeled and strip spaces
            target_map[key] = t

    matched_count = 0

    for inp in input_files:
        key = inp.stem.strip()

        if key in target_map:
            target_file = target_map[key]
            try:
                # 1. Load Input (Grayscale L)
                input_img = Image.open(inp).convert("L")
                input_img = input_img.resize((512, 512))
                input_np = np.array(input_img)

                # 2. Load Target (RGB)
                target_img = Image.open(target_file).convert("RGB")
                target_img = target_img.resize(
                    (512, 512),
                    resample=Image.Resampling.NEAREST,
                )
                target_np = np.array(target_img)

                # 3. Process Mask (Red/Green logic from verify logic or implementations)
                # Reusing logic from implementations.py load_dataset_from_directories
                r = target_np[:, :, 0]
                g = target_np[:, :, 1]
                b = target_np[:, :, 2]

                is_red = (r > 100) & (r > g + 20) & (r > b + 20)
                is_green = (g > 100) & (g > r + 20) & (g > b + 20)

                mask = np.full_like(r, 2, dtype=np.uint8)  # Default 2 (background?)
                mask[is_red] = 0
                mask[is_green] = 1

                dataset.add_sample(input_np, mask)
                matched_count += 1

            except Exception as e:
                print(f"Error loading {inp.name}: {e}")
        else:
            pass

    print(f"Loaded {len(dataset)} pairs out of {len(input_files)} input files.")
    return dataset
