# Proposed Project Structure

This document outlines the final proposed structure for this machine learning project, designed for maximum experimental encapsulation and reproducibility.

The core philosophy is to decouple data processing, model architecture, and experimental training logic into distinct, modular components.

## Directory Tree

```
/
├── main.py
│
├── .data/
│   ├── augment_crop5_rotate/
│   │   ├── brittle/
│   │   └── ductile/
│   └── cnn_v1_approach/
│       └── logs/
│
├── data/
│   └── sem_images/
│       └── raw/
│
├── notebooks/
│   └── ...
│
├── saved_models/
│   └── cnn_v1_approach/
│
└── src/
    ├── __init__.py
    │
    ├── data_processing/
    │   └── sem_images/
    │       └── create_crop5_rotate.py
    │
    ├── models/
    │   └── binary_classification/
    │       └── cnn_v1.py
    │
    ├── utils/
    │   └── config.py
    │
    └── approaches/
        ├── __init__.py
        │
        └── cnn_v1_approach/
            ├── __init__.py
            ├── train.py
            ├── evaluate.py
            │
            ├── data/
            │   ├── __init__.py
            │   └── datasets.py
            │
            └── configs/
                ├── __init__.py
                ├── schemas.py
                └── base.yaml
```

## Directory Explanations

*   **`main.py`**: The main entry point that dispatches commands to the appropriate scripts within `src/approaches/`.

*   **`.data/`**: A git-ignored directory for storing all **generated artifacts**.
    *   **Processed Datasets**: Subdirectories containing the output of scripts from `src/data_processing/`. Convention is to name them after the transformation applied (e.g., `augment_crop5_rotate`).
    *   **Approach-Specific Artifacts**: Subdirectories named after an approach (e.g., `cnn_v1_approach/`) can store outputs like training logs, evaluation metrics, or model checkpoints.

*   **`data/`**: The top-level directory for storing **raw, immutable source data**. This data should be treated as read-only.

*   **`notebooks/`**: For exploratory data analysis and result visualization.

*   **`saved_models/`**: Stores final, trained model weights, organized by the approach that generated them.

*   **`src/`**: The main source code package.

    *   **`src/data_processing/`**: Contains scripts dedicated to transforming raw data into processed datasets.
        *   **Purpose**: This decouples the data engineering pipeline from the modeling pipeline. Scripts here take data from `data/` and save the output to `.data/`.
        *   **Organization**: It's good practice to mirror the structure of `data/`, so a script that processes `data/sem_images` would live in `src/data_processing/sem_images`.

    *   **`src/models/`**: A shared library for reusable model **architectures** (e.g., `CNNBinaryClassifierV1`).

    *   **`src/utils/`**: A collection of helper methods and utilities that can be used throughout the entire project.

    *   **`src/approaches/`**: The central hub for experiments. Each subdirectory is a complete, end-to-end approach to solving a problem.
        *   **`data/`**: Contains approach-specific `Dataset` classes (e.g., using PyTorch) that know how to load data from a specific directory in `.data/`.
        *   `train.py`, `evaluate.py`: The scripts containing the training and evaluation loops for the experiment.
        *   `configs/`: Manages the configuration for this approach.

---

### In-Depth: The Configuration System

Each approach uses a config file to define all its parameters, from data paths to model hyperparameters.

*Example: `base.yaml` for `cnn_v1_approach`*
```yaml
# Path to the processed dataset this approach will use
data:
  path: "data/sem_images/raw"
  batch_size: 32

# Training parameters
training:
  learning_rate: 0.001
  epochs: 50

# Training parameters
training:
  optimizer: "Adam" # Options: "Adam", "SGD"
  epochs: 50
  learning_rate: 0.001
  early_stopping:
    patience: 5
    metric: "val_loss" # Options: "val_loss", "val_f1", "val_auc"
    mode: "min" # Options: "min", "max"
```

This configuration is loaded and validated at runtime by the `utils.config.load_config` function, ensuring that experiments are reproducible and robust.
