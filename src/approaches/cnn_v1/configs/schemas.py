from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, PositiveInt


class EarlyStoppingConfig(BaseModel):
    """Configuration for early stopping."""

    patience: PositiveInt = Field(
        5,
        description="Number of epochs to wait for improvement before stopping",
    )
    metric: Literal["val_loss", "val_f1", "val_auc"] = Field(
        "val_loss",
        description="Metric to monitor for early stopping",
    )
    mode: Literal["min", "max"] = Field(
        "min",
        description="Mode for the monitored metric (min for loss, max for F1/AUC)",
    )


class TrainingConfig(BaseModel):
    """Configuration for the training process."""

    optimizer: Literal["Adam", "SGD"] = Field(
        "Adam",
        description="The optimizer to use for training.",
    )
    epochs: PositiveInt = Field(50, description="Maximum number of training epochs")
    learning_rate: float = Field(
        0.001,
        gt=0.0,
        description="Learning rate for the optimizer",
    )
    early_stopping: EarlyStoppingConfig = Field(
        default_factory=EarlyStoppingConfig.model_construct,
        description="Early stopping configuration",
    )


class ValidationConfig(BaseModel):
    """Configuration for the validation process."""

    folds: PositiveInt = Field(4, description="Number of folds for cross-validation")
    start_train_percentage: PositiveInt = Field(
        20,
        description="Starting percentage of training data",
    )
    end_train_percentage: PositiveInt = Field(
        80,
        description="Ending percentage of training data",
    )
    train_percentage_step: PositiveInt = Field(
        5,
        description="Step size for training data percentage",
    )


class ApproachConfig(BaseModel):
    """Overall configuration for the cnn_v1 approach."""

    data_path: Path = Field(
        ...,
        description="Path to the processed dataset (e.g., .data/augment_crop5_rotate)",
    )
    batch_size: PositiveInt = Field(32, description="Batch size for data loaders")
    train_test_split: float = Field(
        0.8,
        gt=0.0,
        lt=1.0,
        description="Fraction of data to use for training",
    )

    training: TrainingConfig = Field(default_factory=TrainingConfig.model_construct)
    validation: ValidationConfig = Field(
        default_factory=ValidationConfig.model_construct,
    )
