# ML-Project

Machine Learning Project for senior year Computer Science ML course.

![Pull Request Checks](https://github.com/CfM47/ML-Project/actions/workflows/pr-checks.yml/badge.svg)

## Overview

This project implements an AutoML pipeline for **SEM (Scanning Electron Microscopy) image segmentation**. It provides two main workflows:

1. **AutoML Exploration** - Automatically explore combinations of augmentations and models to find optimal configurations
2. **Swin Model Training** - Dedicated training pipeline for Swin Transformer segmentation with learning curve analysis

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Clone the repository
git clone https://github.com/CfM47/ML-Project.git
cd ML-Project

# Install dependencies
uv sync

# Install dev dependencies (for testing/linting)
uv sync --group dev
```

## Data Setup

### SEM Images

Place the SEM Images dataset in the `data/sem_images/raw/` directory:

```
data/sem_images/raw/Brittle/{images.png...}
data/sem_images/raw/Ductile/{images.png...}
```

For training/test splits:
```
data/train/unlabeled/{images.png...}
data/train/labeled/{masks.png...}
data/test/unlabeled/{images.png...}
data/test/labeled/{masks.png...}
```

## Usage

### Workflow 1: AutoML Exploration

The AutoML system explores combinations of augmentation strategies and segmentation models using k-fold cross-validation.

```python
from main import _run_with_setup

_run_with_setup(
    unlabeled_dir="data/train/unlabeled",
    labeled_dir="data/train/labeled",
    classification_dataset_dir="data/classification",
    auto_ml_cache_dir="cache/automl",
    augmentator_indices=[0, 1],  # Optional: filter augmentators
    model_indices=[0, 1],        # Optional: filter models
)
```

**Available Models** (via `setup/models/setup.py`):
- ViT Segmentation Model
- Swin Segmentation Model
- QuadTree + CNN/ViT classifiers
- SlidingWindow + CNN/ViT classifiers

**Available Augmentations** (via `setup/augmentators/setup.py`):
- Identity (no augmentation)
- Combined 2Geo + 2Photo + 1SEM
- Combined 3Geo + 1Photo + 1SEM

### Workflow 2: Swin Model Training

Dedicated training pipeline with learning curve analysis and early stopping support.

#### Learning Curve Validation

Run k-fold cross-validation at varying training percentages:

```python
from model.swin.train import run_percentage_validation
from model.swin.config import SwinTrainingConfig

config = SwinTrainingConfig(
    train_percentages=[10, 20, 30, 40, 50, 60, 70, 80],
    n_folds=5,
    epochs=40,
    patience=5,  # Early stopping (None to disable)
)

metrics, fig = run_percentage_validation(
    train_unlabeled_dir="data/train/unlabeled",
    train_labeled_dir="data/train/labeled",
    config=config,
)
```

#### Final Model Training

Train on full dataset with 80/20 validation split:

```python
from model.swin.train import run_final_training
from model.swin.config import SwinTrainingConfig

config = SwinTrainingConfig(
    epochs=40,
    patience=5,
    output_dir="results/swin",
)

model, test_metrics, mask_pairs, fig = run_final_training(
    train_unlabeled_dir="data/train/unlabeled",
    train_labeled_dir="data/train/labeled",
    test_unlabeled_dir="data/test/unlabeled",
    test_labeled_dir="data/test/labeled",
    config=config,
)
```

#### CLI Usage

```bash
# Learning curve validation
python -m model.swin.train validate \
    --train-unlabeled data/train/unlabeled \
    --train-labeled data/train/labeled \
    --output-dir results/swin

# Final training with test evaluation
python -m model.swin.train train \
    --train-unlabeled data/train/unlabeled \
    --train-labeled data/train/labeled \
    --test-unlabeled data/test/unlabeled \
    --test-labeled data/test/labeled \
    --output-dir results/swin
```

### Kaggle Notebooks

Pre-configured notebooks for running on Kaggle are available in `kaggle/`:

- `run-automl.ipynb` - AutoML exploration
- `run-training.ipynb` - Swin final model training
- `run-validation.ipynb` - Swin learning curve validation

## Configuration

### SwinTrainingConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_percentages` | `List[int]` | `[10, 20, ..., 80]` | Percentages for learning curve |
| `n_folds` | `int` | `5` | Number of cross-validation folds |
| `epochs` | `int` | `40` | Training epochs |
| `batch_size` | `int` | `2` | Batch size |
| `learning_rate` | `float` | `1e-4` | Learning rate |
| `embed_dim` | `int` | `128` | Swin embedding dimension |
| `depths` | `List[int]` | `[2, 2, 18, 2]` | Swin layer depths |
| `num_heads` | `List[int]` | `[8, 16, 32, 64]` | Swin attention heads |
| `patience` | `int \| None` | `None` | Early stopping patience (None = disabled) |
| `augmentation_copies` | `int` | `2` | Augmentation copies per sample |
| `num_test_visualizations` | `int` | `10` | Samples to visualize |
| `output_dir` | `Path` | `results/swin` | Output directory |
| `seed` | `int` | `42` | Random seed |
| `device` | `str` | `"auto"` | Device: "auto", "cuda", "mps", "cpu" |

## Project Structure

```
ML-Project/
├── auto_ml/                    # Core AutoML framework
│   ├── implementations/        # Concrete implementations
│   │   ├── augmentators/       # Data augmentation strategies
│   │   ├── classifiers/        # CNN, ViT classifiers
│   │   ├── evaluators/         # Metrics (Dice, IoU, Accuracy, etc.)
│   │   ├── segmentators/       # Swin, ViT, QuadTree, SlidingWindow
│   │   ├── datasets.py         # Dataset loading utilities
│   │   └── nodes.py            # AutoML pipeline nodes
│   ├── interfaces.py           # Abstract interfaces
│   └── automl.py               # AutoML orchestration
├── model/                      # Swin training pipeline
│   └── swin/
│       ├── config.py           # SwinTrainingConfig
│       ├── train.py            # Training entry points
│       ├── data.py             # Data utilities
│       ├── evaluation.py       # Evaluation helpers
│       ├── metrics.py          # Metrics dataclasses
│       └── visualization.py    # Plotting utilities
├── setup/                      # Pre-configured setups for AutoML
│   ├── augmentators/           # Augmentation node configurations
│   ├── evaluator/              # Evaluator configurations
│   └── models/                 # Model node configurations
├── kaggle/                     # Kaggle notebook templates
├── tests/                      # Unit tests
├── main.py                     # AutoML entry point
└── pyproject.toml              # Project configuration
```

## Development

```bash
# Run tests
make test

# Type checking
make typecheck

# Linting
make lint

# Format code
make format
```

## License

This project is for educational purposes as part of a senior year ML course.
