"""
Training, prediction, and checkpointing for the brain-age regression models.
"""

import copy
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

CRITERIONS = {"MSE": nn.MSELoss, "L1": nn.L1Loss, "Huber": nn.HuberLoss, "SmoothL1": nn.SmoothL1Loss}
OPTIMIZERS = {
    "Adam": lambda p, lr, wd: optim.Adam(p, lr=lr, weight_decay=wd),
    "AdamW": lambda p, lr, wd: optim.AdamW(p, lr=lr, weight_decay=wd),
    "SGD": lambda p, lr, wd: optim.SGD(p, lr=lr, momentum=0.9, weight_decay=wd),
    "RMSprop": lambda p, lr, wd: optim.RMSprop(p, lr=lr, alpha=0.99, weight_decay=wd),
    "Adagrad": lambda p, lr, wd: optim.Adagrad(p, lr=lr, weight_decay=wd),
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_losses(train_losses, val_losses, title="Training and Validation Loss"):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def train(model, train_dataset, val_dataset, holdout_dataset, num_epochs=100, device="cuda",
          lr=1e-3, criterion_type="MSE", optimizer_type="SGD", weight_decay=1e-5, batch_size=32,
          seed=42, patience=40, clip_value=1.0, use_amp=True, num_workers=4, verbose=True):
    """
    Train with early stopping on validation loss, then report loss on a
    held-out dataset for a quick sanity check (this is not the final
    evaluation — that is done separately for every trained model in the
    evaluation stage, on both the healthy and ADHD test sets).
    """
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'=' * 70}\nTRAINING CONFIGURATION\n{'=' * 70}")
    print(f"Device: {device}   Model: {model.__class__.__name__}   "
          f"Trainable parameters: {n_params:,}")
    print(f"Batch size: {batch_size}   Learning rate: {lr}   "
          f"Mixed precision: {use_amp and device.type == 'cuda'}")
    print(f"{'=' * 70}\n")

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                          pin_memory=(device.type == "cuda"), persistent_workers=(num_workers > 0))
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    holdout_loader = DataLoader(holdout_dataset, shuffle=False, **loader_kwargs)

    # Re-initialize parameters right before training so every seed starts from
    # a fresh, independently-initialized model even if the builder function
    # that constructed it is reused.
    model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)

    criterion = CRITERIONS[criterion_type]()
    optimizer = OPTIMIZERS[optimizer_type](model.parameters(), lr, weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
    scaler = GradScaler() if (use_amp and device.type == "cuda") else None

    def run_batches(loader, training):
        model.train() if training else model.eval()
        total_loss = 0.0
        with torch.set_grad_enabled(training):
            for images, labels in loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                if training:
                    optimizer.zero_grad()
                with autocast(enabled=scaler is not None):
                    outputs = model(images)
                    loss = criterion(outputs.squeeze(), labels)
                if training:
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                        optimizer.step()
                total_loss += loss.item() * images.size(0)
        return total_loss / len(loader.dataset)

    train_losses, val_losses = [], []
    best_state = copy.deepcopy(model.state_dict())
    best_loss, epochs_without_improvement = float("inf"), 0

    print("Starting training...\n")
    for epoch in range(num_epochs):
        start = time.time()
        train_loss = run_batches(train_loader, training=True)
        val_loss = run_batches(val_loader, training=False)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs} | Train loss: {train_loss:.4f} | "
                  f"Val loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | "
                  f"Time: {time.time() - start:.1f}s")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping: no improvement for {patience} epochs.")
            break

        if epoch % 5 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    model.load_state_dict(best_state)
    holdout_loss = run_batches(holdout_loader, training=False)

    print(f"\n{'=' * 70}\nTRAINING COMPLETE\n{'=' * 70}")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Holdout-set loss (sanity check, not final evaluation): {holdout_loss:.4f}")
    print(f"{'=' * 70}\n")

    plot_losses(train_losses, val_losses)
    return model, holdout_loss, train_losses, val_losses


def predict(model, dataset, device="cuda", batch_size=32, num_workers=4):
    """Run inference over a dataset and return predictions, labels, and regression metrics."""
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=(device.type == "cuda"))

    predictions, labels = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            outputs = model(images.to(device)).squeeze().cpu().numpy()
            predictions.extend(np.atleast_1d(outputs))
            labels.extend(batch_labels.numpy())

    predictions, labels = np.array(predictions), np.array(labels)
    metrics = calculate_metrics(predictions, labels)
    return predictions, labels, metrics


def calculate_metrics(predictions, labels):
    """MSE, MAE, Pearson correlation, and R-squared between predictions and labels."""
    mse = float(np.mean((predictions - labels) ** 2))
    mae = float(np.mean(np.abs(predictions - labels)))
    pearson_r, _ = pearsonr(predictions, labels)
    ss_res = np.sum((labels - predictions) ** 2)
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    r_squared = float(1 - ss_res / ss_tot)
    return {"MSE": mse, "MAE": mae, "Pearson": float(pearson_r), "R_squared": r_squared}


def scatter_plot(predictions, labels, title="Predicted vs. actual age", save_path=None):
    """Scatter predictions against ground truth with an identity line."""
    plt.figure(figsize=(6, 6))
    plt.scatter(labels, predictions, alpha=0.5, edgecolors="k")
    lo, hi = min(labels.min(), predictions.min()), max(labels.max(), predictions.max())
    plt.plot([lo, hi], [lo, hi], "r--", lw=2, label="Identity")
    plt.xlabel("Actual age")
    plt.ylabel("Predicted age")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.axis("equal")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


def save_model_with_metrics(model, holdout_loss, train_losses, val_losses,
                             save_dir="saved_models", model_name=None, additional_info=None):
    """Save model weights plus training history to `{save_dir}/{model_name}.pth`, and a
    companion `{model_name}_metrics.json` for inspecting results without loading PyTorch."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if model_name is None:
        model_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "holdout_loss": holdout_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": min(val_losses),
        "best_epoch": val_losses.index(min(val_losses)) + 1,
        "num_epochs": len(train_losses),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_config": model.get_model_info() if hasattr(model, "get_model_info") else None,
        "additional_info": additional_info,
    }
    checkpoint_path = save_dir / f"{model_name}.pth"
    torch.save(checkpoint, checkpoint_path)

    metrics = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    with open(save_dir / f"{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"Model saved to {checkpoint_path}")
    print(f"  Best validation loss: {checkpoint['best_val_loss']:.4f} at epoch {checkpoint['best_epoch']}")
    return checkpoint_path


def load_model_with_metrics(checkpoint_path, model):
    """Load weights into `model` and return it with the saved training metrics."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    print(f"Model loaded from {checkpoint_path} "
          f"(best val loss {metrics['best_val_loss']:.4f} at epoch {metrics['best_epoch']})")
    return model, metrics
