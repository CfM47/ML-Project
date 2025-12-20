"""
Concrete implementations of the AutoML interfaces.

This module provides ready-to-use implementations of the abstract
interfaces defined in interfaces.py.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import os
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from auto_ml.models.vit.model import ViTSegmentation
from auto_ml.models.swin.model import SwinSegmentation

from auto_ml.interfaces import (
    DataAugmentatorInterface,
    DataAugmentatorNodeInterface,
    DatasetInterface,
    ImageArray,
    MaskArray,
    ModelNodeInterface,
    ModelInterface,
    MetricsResultInterface
)


# ==============================================================================
# Data Augmentator Implementations
# ==============================================================================

class IdentityAugmentator(DataAugmentatorInterface):
    """
    Identity augmentator that returns the dataset unchanged.
    
    Useful as a baseline or when no augmentation is desired.
    """
    
    def augment(self, dataset: DatasetInterface) -> DatasetInterface:
        """Return the dataset unchanged."""
        return DatasetInterface(
            samples=list(dataset.samples),
            metadata={**dataset.metadata, "augmentation": "identity"}
        )

# ==============================================================================
# Data Processing Implementations
# ==============================================================================

def load_dataset_from_directories(
    input_dir: str, 
    target_dir: str, 
    target_size: Tuple[int, int] = (512, 512)
) -> DatasetInterface:
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
    input_files = sorted([
        f for f in input_path.glob('*') 
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    ])
    
    # Get all target files
    target_files = sorted([
        f for f in target_path.glob('*') 
        if f.suffix.lower() in ['.png'] and '_labeled' in f.name
    ])
    
    dataset = DatasetInterface()
    
    # Create valid pairs
    # Heuristic: Target stem should start with Input stem
    # Or more robust: Target name is Input stem + "_labeled"
    
    # Let's map normalized stems to consistency
    target_map = {}
    for t in target_files:
        # standard format: "name_labeled.png" -> key: "name"
        # Handle "  _labeled" or "_labeled"
        stem = t.stem
        if stem.endswith('_labeled'):
             key = stem[:-8].strip() # remove _labeled and strip spaces
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
                target_img = target_img.resize((512, 512), resample=Image.NEAREST)
                target_np = np.array(target_img)
                
                # 3. Process Mask (Red/Green logic from verify logic or implementations)
                # Reusing logic from implementations.py load_dataset_from_directories
                r = target_np[:, :, 0]
                g = target_np[:, :, 1]
                b = target_np[:, :, 2]
                
                is_red = (r > 100) & (r > g + 20) & (r > b + 20)
                is_green = (g > 100) & (g > r + 20) & (g > b + 20)
                
                mask = np.full_like(r, 2, dtype=np.uint8) # Default 2 (background?)
                mask[is_red] = 0
                mask[is_green] = 1
                
                dataset.add_sample(input_np, mask)
                matched_count += 1
                
            except Exception as e:
                print(f"Error loading {inp.name}: {e}")
        else:
            # print(f"Skipping {inp.name} (No match found for key '{key}')")
            pass
            
    print(f"Loaded {len(dataset)} pairs out of {len(input_files)} input files.")
    return dataset

# ==============================================================================
# Data Augmentator Node Implementation
# ==============================================================================



# ==============================================================================
# Model Implementations
# ==============================================================================

class DataAugmentatorNode(DataAugmentatorNodeInterface):
    """
    Data Augmentator Node implementation.
    
    Splits the dataset into K folds (or single split) and applies augmentation
    to the training set of each fold.
    """
    
    def __init__(
        self, 
        augmentator: DataAugmentatorInterface, 
        name: str = "DataAugmentatorNode",
        k_folds: int = 5,
        test_size: float = 0.2,
        random_seed: int = 42
    ):
        """
        Initialize the node.
        
        Args:
            augmentator: The augmentator to apply to training sets.
            name: Name of this node instance.
            k_folds: Number of folds for Cross Validation. 
                     If 1, performs a single random split based on test_size.
            test_size: Fraction of data for validation if k_folds=1.
            random_seed: Seed for reproducibility.
        """
        self.augmentator = augmentator
        self.name = name
        self.k_folds = k_folds
        self.test_size = test_size
        self.random_seed = random_seed
        
    def process(
        self, dataset: DatasetInterface
    ) -> List[Tuple[DatasetInterface, DatasetInterface]]:
        """
        Process the dataset.
        
        Returns:
            List of (augmented_train_dataset, val_dataset) tuples.
        """
        n_samples = len(dataset)
        indices = np.arange(n_samples)
        rng = np.random.default_rng(self.random_seed)
        
        # Shuffle indices once
        rng.shuffle(indices)
        
        results = []
        
        if self.k_folds <= 1:
            # Single Split
            split_idx = int(n_samples * (1 - self.test_size))
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]
            
            # Create Datasets
            train_dataset = DatasetInterface.from_pairs(
                [dataset.samples[i] for i in train_indices],
                metadata={**dataset.metadata, "split": "train", "fold": 0}
            )
            val_dataset = DatasetInterface.from_pairs(
                [dataset.samples[i] for i in val_indices],
                metadata={**dataset.metadata, "split": "val", "fold": 0}
            )
            
            # Augment Train
            aug_train = self.augmentator.augment(train_dataset)
            
            results.append((aug_train, val_dataset))
            
        else:
            # K-Fold Split
            fold_sizes = np.full(self.k_folds, n_samples // self.k_folds, dtype=int)
            fold_sizes[:n_samples % self.k_folds] += 1
            current = 0
            
            for i in range(self.k_folds):
                start, stop = current, current + fold_sizes[i]
                val_mask = np.zeros(n_samples, dtype=bool)
                val_mask[start:stop] = True
                
                val_indices_fold = indices[val_mask]
                train_indices_fold = indices[~val_mask]
                
                # Create Datasets
                train_dataset = DatasetInterface.from_pairs(
                    [dataset.samples[j] for j in train_indices_fold],
                    metadata={**dataset.metadata, "split": "train", "fold": i}
                )
                val_dataset = DatasetInterface.from_pairs(
                    [dataset.samples[j] for j in val_indices_fold],
                    metadata={**dataset.metadata, "split": "val", "fold": i}
                )
                
                # Augment Train
                aug_train = self.augmentator.augment(train_dataset)
                
                results.append((aug_train, val_dataset))
                
                current = stop
                
        return results


class InMemoryPyTorchDataset(Dataset):
    """
    Bridge between AutoML DatasetInterface and PyTorch Dataset.
    """
    
    def __init__(self, dataset: DatasetInterface):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # reuse logic from vit/dataset.py but adapted for in-memory data
        input_interface, output_interface = self.dataset[idx]
        
        # Prepare inputs
        # Assuming images are numpy arrays (H, W, 3) or (H, W)
        img = input_interface.image
        if input_interface.is_grayscale:
             # Convert to tensor (1, H, W)
             img_tensor = torch.from_numpy(img).float() / 255.0
             img_tensor = img_tensor.unsqueeze(0)
        else:
            # Convert RGB (H, W, 3) -> (3, H, W)
            img_tensor = torch.from_numpy(img).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1)
            
        # Prepare targets
        mask = output_interface.mask
        mask_tensor = torch.from_numpy(mask).long()
        
        return img_tensor, mask_tensor


class ViTModel(ModelInterface):
    """
    ViT Model implementation for AutoML.
    Wraps the ViTSegmentation model from vit.model.
    """
    
    def __init__(
        self, 
        epochs: int = 10, 
        batch_size: int = 4, 
        lr: float = 1e-4,
        device: str = "auto"
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = ViTSegmentation(
            image_size=512,
            patch_size=16,
            num_classes=3,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=1024,
            channels=1, # Warning: Hardcoded for grayscale, should be dynamic if needed
            dropout=0.1,
            emb_dropout=0.1
        ).to(self.device)

    def train(self, dataset: DatasetInterface) -> MetricsResultInterface:
        """Train the model."""
        pytorch_dataset = InMemoryPyTorchDataset(dataset)
        dataloader = DataLoader(pytorch_dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.model.train()
        
        total_loss = 0
        
        # Simplified training loop for AutoML context
        for epoch in range(self.epochs):
            epoch_loss = 0
            # iterate over batches
            for inputs, masks in dataloader:
                inputs = inputs.to(self.device)
                masks = masks.to(self.device)
                
                # Check for channel mismatch if we hardcoded channels=1
                if inputs.shape[1] != 1 and self.model.patch_embed.in_channels == 1:
                     # Force grayscale conversion if model expects 1 channel
                     # (B, 3, H, W) -> (B, 1, H, W) using luminosity method
                     inputs = inputs[:, 0:1, :, :] * 0.299 + inputs[:, 1:2, :, :] * 0.587 + inputs[:, 2:3, :, :] * 0.114

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0
            total_loss = avg_loss # Report the last epoch's loss
            print(f"Epoch {epoch+1}/{self.epochs} Loss: {avg_loss:.4f}")

        return MetricsResultInterface(
            loss=total_loss,
            accuracy=0.0, # Placeholder
            additional_metrics={"epochs_trained": self.epochs}
        )

    def evaluate(self, dataset: DatasetInterface) -> MetricsResultInterface:
        """Evaluate the model."""
        pytorch_dataset = InMemoryPyTorchDataset(dataset)
        dataloader = DataLoader(pytorch_dataset, batch_size=1, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_pixels = 0
        
        with torch.no_grad():
            for inputs, masks in dataloader:
                inputs = inputs.to(self.device)
                masks = masks.to(self.device)
                
                if inputs.shape[1] != 1 and self.model.patch_embed.in_channels == 1:
                     inputs = inputs[:, 0:1, :, :] * 0.299 + inputs[:, 1:2, :, :] * 0.587 + inputs[:, 2:3, :, :] * 0.114
                
                outputs = self.model(inputs)
                loss = criterion(outputs, masks)
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(outputs, dim=1)
                correct = (predictions == masks).sum().item()
                total_correct += correct
                total_pixels += masks.numel()
        
        count = len(dataloader)
        avg_loss = total_loss / count if count > 0 else 0.0
        accuracy = total_correct / total_pixels if total_pixels > 0 else 0.0
        
        return MetricsResultInterface(
            loss=avg_loss,
            accuracy=accuracy
        )

class SwinModel(ModelInterface):
    """
    Swin Transformer Model implementation for AutoML.
    Wraps the SwinSegmentation model from swin.model.
    """
    
    def __init__(
        self, 
        epochs: int = 10, 
        batch_size: int = 4, 
        lr: float = 1e-4,
        embed_dim: int = 96,
        depths: List[int] = None,
        num_heads: List[int] = None,
        window_size: List[int] = None,
        device: str = "auto"
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        
        # Defaults for Swin-T if not provided
        self.embed_dim = embed_dim
        self.depths = depths if depths else [2, 2, 6, 2]
        self.num_heads = num_heads if num_heads else [3, 6, 12, 24]
        self.window_size = window_size if window_size else [7, 7]
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = SwinSegmentation(
            patch_size=[4, 4], # Default fixed
            embed_dim=self.embed_dim,
            depths=self.depths,
            num_heads=self.num_heads,
            window_size=self.window_size,
            mlp_ratio=4.0,
            dropout=0.1,
            num_classes=3,
            channels=1
        ).to(self.device)

    def train(self, dataset: DatasetInterface) -> MetricsResultInterface:
        """Train the model."""
        pytorch_dataset = InMemoryPyTorchDataset(dataset)
        dataloader = DataLoader(pytorch_dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.model.train()
        
        total_loss = 0
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            for inputs, masks in dataloader:
                inputs = inputs.to(self.device)
                masks = masks.to(self.device)
                
                # Check for channel mismatch (similar logic to ViTModel)
                if inputs.shape[1] != 1:
                     inputs = inputs[:, 0:1, :, :] * 0.299 + inputs[:, 1:2, :, :] * 0.587 + inputs[:, 2:3, :, :] * 0.114

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0
            total_loss = avg_loss
            print(f"Swin Epoch {epoch+1}/{self.epochs} Loss: {avg_loss:.4f}")

        return MetricsResultInterface(
            loss=total_loss,
            accuracy=0.0,
            additional_metrics={"epochs_trained": self.epochs}
        )

    def evaluate(self, dataset: DatasetInterface) -> MetricsResultInterface:
        """Evaluate the model."""
        pytorch_dataset = InMemoryPyTorchDataset(dataset)
        dataloader = DataLoader(pytorch_dataset, batch_size=1, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_pixels = 0
        
        with torch.no_grad():
            for inputs, masks in dataloader:
                inputs = inputs.to(self.device)
                masks = masks.to(self.device)
                
                if inputs.shape[1] != 1:
                     inputs = inputs[:, 0:1, :, :] * 0.299 + inputs[:, 1:2, :, :] * 0.587 + inputs[:, 2:3, :, :] * 0.114
                
                outputs = self.model(inputs)
                loss = criterion(outputs, masks)
                total_loss += loss.item()
                
                predictions = torch.argmax(outputs, dim=1)
                correct = (predictions == masks).sum().item()
                total_correct += correct
                total_pixels += masks.numel()
        
        count = len(dataloader)
        avg_loss = total_loss / count if count > 0 else 0.0
        accuracy = total_correct / total_pixels if total_pixels > 0 else 0.0
        
        return MetricsResultInterface(
            loss=avg_loss,
            accuracy=accuracy
        )

# ==============================================================================
# Model Node Implementations
# ==============================================================================

class ModelNode(ModelNodeInterface):
    """
    Generic Model Node implementation.
    
    Manages the training and evaluation of a model across multiple dataset pairs
    (e.g., cross-validation folds).
    """
    
    def __init__(self, model: ModelInterface, name: str = "ModelNode"):
        """
        Initialize the Model Node.
        
        Args:
            model: The model to train and evaluate.
            name: Name of this node instance.
        """
        self.model = model
        self.name = name
        
    def train(
        self, 
        dataset_pairs: List[Tuple[DatasetInterface, DatasetInterface]]
    ) -> Dict[str, Any]:
        """
        Train the model on the provided dataset pairs.
        
        Args:
            dataset_pairs: List of (train_dataset, val_dataset) tuples.
            
        Returns:
            Dictionary containing:
                - results: List of MetricsResultInterface (one per pair)
                - mean_loss: Average loss across all validation sets
        """
        results = []
        total_loss = 0.0
        
        for i, (train_dataset, val_dataset) in enumerate(dataset_pairs):
            print(f"Processing split {i+1}/{len(dataset_pairs)}...")
            
            # Train model
            print(f"  Training on {len(train_dataset)} samples...")
            _ = self.model.train(train_dataset)
            
            # Evaluate on validation set
            print(f"  Evaluating on {len(val_dataset)} samples...")
            metrics = self.model.evaluate(val_dataset)
            
            results.append(metrics)
            total_loss += metrics.loss
            
            print(f"  Split {i+1} Results: {metrics}")
            
        mean_loss = total_loss / len(dataset_pairs) if dataset_pairs else 0.0
        
        return {
            "results": results,
            "mean_loss": mean_loss
        }
