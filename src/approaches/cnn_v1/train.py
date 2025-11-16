import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim import SGD, Adam

from src.utils.config import load_config

from ...models.binary_classification.cnn_v1 import CNNBinaryClassifierV1
from .configs.schemas import ApproachConfig
from .data.datasets import get_data, get_dataloader, split_exclusive
from .schemas import TrainingContext, ValidationMetrics


def train(config_name: str = "base") -> None:
    """Run the main training experiment."""
    config_path = Path(__file__).parent / "configs" / f"{config_name}.yaml"
    saved_model_path = (
        Path(__file__).parent.parent.parent.parent
        / ".data"
        / "cnn_v1_approach"
        / "saved_models"
    )
    config = load_config(config_path, ApproachConfig)

    ctx = setup_training(config)
    train, val = get_data(config)
    train_subset, val_subset = split_exclusive(train, val, config)
    train_loader = get_dataloader(config, train_subset, train=True)
    val_loader = get_dataloader(config, val_subset, train=False)

    run_training_loop(
        ctx,
        config,
        train_loader,
        val_loader,
        save_model_dir=saved_model_path,
    )


def setup_training(config: ApproachConfig) -> TrainingContext:
    """Create and return the training context."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNBinaryClassifierV1().to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer: Adam | SGD

    if config.training.optimizer == "Adam":
        optimizer = Adam(model.parameters(), lr=config.training.learning_rate)
    elif config.training.optimizer == "SGD":
        optimizer = SGD(model.parameters(), lr=config.training.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {config.training.optimizer}")

    return TrainingContext(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
    )


def run_training_loop(
    ctx: TrainingContext,
    config: ApproachConfig,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    save_model_dir: Path | None = None,
) -> Tuple[float, float]:
    """Run the main training loop and return the best validation loss."""
    best_val_loss = float("inf")
    train_loss_for_best_val = float("inf")
    epochs_no_improve = 0
    patience = config.training.early_stopping.patience

    print("--- Starting Training ---")
    start_time = time.time()

    for epoch in range(config.training.epochs):
        train_loss = train_one_epoch(ctx, train_loader)
        val_metrics = validate(ctx, val_loader)

        # Quieter logging for the experiment loop
        print(
            f"Epoch [{epoch + 1}/{config.training.epochs}] - "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics.loss:.4f}",
        )

        if val_metrics.loss < best_val_loss:
            print(
                f"Best validation loss improved from {best_val_loss:.4f} to ",
                f"{val_metrics.loss:.4f}",
            )
            train_loss_for_best_val = train_loss
            best_val_loss = val_metrics.loss
            epochs_no_improve = 0

            if save_model_dir:
                save_model_dir.mkdir(parents=True, exist_ok=True)
                model_path = save_model_dir / "best_model.pth"
                torch.save(ctx.model.state_dict(), model_path)
                print(f"Saved best model to {model_path}")
        else:
            epochs_no_improve += 1
            print(
                "Validation loss did not improve, patience:",
                patience - epochs_no_improve,
            )

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    end_time = time.time()
    print(f"--- Training Finished in {end_time - start_time:.2f} seconds ---")
    return best_val_loss, train_loss_for_best_val


def train_one_epoch(
    ctx: TrainingContext,
    dataloader: torch.utils.data.DataLoader,
) -> float:
    """Train the model for one epoch."""
    ctx.model.train()
    total_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(ctx.device), labels.to(ctx.device)

        outputs = ctx.model(images)
        loss = ctx.loss_fn(outputs, labels.float().unsqueeze(1))

        ctx.optimizer.zero_grad()
        loss.backward()
        ctx.optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(
    ctx: TrainingContext,
    dataloader: torch.utils.data.DataLoader,
) -> ValidationMetrics:
    """Validate the model on the given dataloader."""
    ctx.model.eval()
    total_loss = 0.0
    all_preds_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(ctx.device), labels.to(ctx.device)
            outputs = ctx.model(images)
            loss = ctx.loss_fn(outputs, labels.float().unsqueeze(1))
            total_loss += loss.item()

            all_preds_probs.append(outputs.sigmoid().cpu())
            all_labels.append(labels.cpu())

    all_preds_probs_np = torch.cat(all_preds_probs).numpy()
    all_labels_np = torch.cat(all_labels).numpy()
    all_preds_binary = (all_preds_probs_np >= 0.5).astype(int)

    accuracy = accuracy_score(all_labels_np, all_preds_binary)
    f1 = f1_score(all_labels_np, all_preds_binary)
    auroc = roc_auc_score(all_labels_np, all_preds_probs_np)

    return ValidationMetrics(
        loss=total_loss / len(dataloader),
        accuracy=accuracy,
        f1=f1,
        auroc=auroc,
    )


if __name__ == "__main__":
    train()
