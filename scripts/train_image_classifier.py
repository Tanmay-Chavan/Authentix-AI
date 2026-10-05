"""Training script for the EfficientNet-B4 image authenticity classifier.

Supports:
- Transfer learning with frozen backbone
- Early stopping on validation AUC-ROC
- Mixed precision training (when GPU available)
- Checkpoint saving of best model
- Detailed metrics logging (loss, accuracy, AUC, precision, recall, F1)

Usage:
    python -m scripts.train_image_classifier --dataset cifake --epochs 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

import structlog

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from authentix.models.image_classifier import ImageClassifier
from authentix.pipelines.data_registry import DataRegistry
from authentix.pipelines.image.transforms import (
    get_train_transforms,
    get_val_transforms,
)

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
# Training Engine
# ═══════════════════════════════════════════════════════════════

class Trainer:
    """Training engine with validation, early stopping, and metrics."""

    def __init__(
        self,
        model: ImageClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cpu",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        patience: int = 3,
        output_dir: str = "models_store",
        model_name: str = "image_classifier",
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.patience = patience
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        # Loss function — BCE with logits (numerically stable)
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer — AdamW with separate LR for backbone vs head
        backbone_params = [
            p for p in model.backbone.parameters() if p.requires_grad
        ]
        head_params = list(model.classifier.parameters())

        self.optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": lr * 0.1},  # Lower LR for backbone
            {"params": head_params, "lr": lr},
        ], weight_decay=weight_decay)

        # Learning rate scheduler — reduce on plateau
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2, verbose=True
        )

        # Mixed precision (GPU only)
        self.scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
        self.use_amp = device == "cuda"

        # Early stopping state
        self.best_auc = 0.0
        self.epochs_without_improvement = 0
        self.history: list[dict] = []

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            labels = batch["label"].float().to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(images).squeeze(-1)
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images).squeeze(-1)
                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            running_loss += loss.item()
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_labels.extend(labels.cpu().numpy().astype(int))
            all_preds.extend(preds)
            all_probs.extend(probs)

            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    "train.batch",
                    epoch=epoch,
                    batch=f"{batch_idx + 1}/{len(self.train_loader)}",
                    loss=f"{loss.item():.4f}",
                )

        metrics = self._compute_metrics(all_labels, all_preds, all_probs)
        metrics["loss"] = running_loss / len(self.train_loader)
        return metrics

    @torch.no_grad()
    def validate(self) -> dict:
        """Run validation and compute all metrics."""
        self.model.eval()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].float().to(self.device)

            logits = self.model(images).squeeze(-1)
            loss = self.criterion(logits, labels)

            running_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_labels.extend(labels.cpu().numpy().astype(int))
            all_preds.extend(preds)
            all_probs.extend(probs)

        metrics = self._compute_metrics(all_labels, all_preds, all_probs)
        metrics["loss"] = running_loss / max(len(self.val_loader), 1)
        return metrics

    def train(self, num_epochs: int) -> dict:
        """Full training loop with early stopping.

        Returns
        -------
        dict
            Final training summary with best metrics.
        """
        logger.info(
            "training.start",
            epochs=num_epochs,
            device=self.device,
            train_batches=len(self.train_loader),
            val_batches=len(self.val_loader),
        )
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Update LR scheduler
            self.scheduler.step(val_metrics["auc_roc"])

            epoch_time = time.time() - epoch_start

            # Log epoch results
            logger.info(
                "training.epoch",
                epoch=f"{epoch}/{num_epochs}",
                train_loss=f"{train_metrics['loss']:.4f}",
                train_acc=f"{train_metrics['accuracy']:.4f}",
                val_loss=f"{val_metrics['loss']:.4f}",
                val_acc=f"{val_metrics['accuracy']:.4f}",
                val_auc=f"{val_metrics['auc_roc']:.4f}",
                val_f1=f"{val_metrics['f1']:.4f}",
                time=f"{epoch_time:.1f}s",
            )

            # Save history
            self.history.append({
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "time": epoch_time,
            })

            # Early stopping check
            if val_metrics["auc_roc"] > self.best_auc:
                self.best_auc = val_metrics["auc_roc"]
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_metrics)
                logger.info(
                    "training.new_best",
                    auc=f"{self.best_auc:.4f}",
                )
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    logger.info(
                        "training.early_stop",
                        epoch=epoch,
                        best_auc=f"{self.best_auc:.4f}",
                    )
                    break

        total_time = time.time() - start_time
        summary = {
            "total_time": f"{total_time:.1f}s",
            "best_auc": self.best_auc,
            "epochs_trained": len(self.history),
            "history": self.history,
        }

        # Save training history
        history_path = self.output_dir / f"{self.model_name}_history.json"
        with open(history_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(
            "training.complete",
            total_time=summary["total_time"],
            best_auc=f"{self.best_auc:.4f}",
            epochs=summary["epochs_trained"],
        )

        return summary

    def _save_checkpoint(self, epoch: int, metrics: dict) -> None:
        """Save the best model checkpoint."""
        path = self.output_dir / f"{self.model_name}_best.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }, path)
        logger.info("training.checkpoint_saved", path=str(path))

    @staticmethod
    def _compute_metrics(
        labels: list,
        preds: list,
        probs: list,
    ) -> dict:
        """Compute all evaluation metrics."""
        labels_arr = np.array(labels)
        preds_arr = np.array(preds)
        probs_arr = np.array(probs)

        try:
            auc = roc_auc_score(labels_arr, probs_arr)
        except ValueError:
            auc = 0.0  # All same class

        return {
            "accuracy": float(accuracy_score(labels_arr, preds_arr)),
            "precision": float(precision_score(labels_arr, preds_arr, zero_division=0)),
            "recall": float(recall_score(labels_arr, preds_arr, zero_division=0)),
            "f1": float(f1_score(labels_arr, preds_arr, zero_division=0)),
            "auc_roc": float(auc),
            "confusion_matrix": confusion_matrix(labels_arr, preds_arr).tolist(),
        }


# ═══════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train Authentix image classifier"
    )
    parser.add_argument(
        "--dataset", choices=["cifake", "genimage"], default="cifake",
        help="Dataset to train on",
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size (reduce for low VRAM)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max samples per class (for quick testing)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto, cpu, or cuda",
    )
    parser.add_argument(
        "--data-dir", type=str, default="d:/Files/DMCE/Authentix/data",
        help="Path to data directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default="models_store",
        help="Path to save checkpoints",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2,
        help="Validation split ratio (for CIFAKE which has no val set)",
    )
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("training.device", device=device)

    # Load data
    registry = DataRegistry(args.data_dir)

    if args.dataset == "cifake":
        train_transform = get_train_transforms()
        val_transform = get_val_transforms()

        full_train = registry.get_cifake(
            split="train",
            transform=train_transform,
            max_samples_per_class=args.max_samples,
        )
        # Split train into train + val
        val_size = int(len(full_train) * args.val_split)
        train_size = len(full_train) - val_size
        train_ds, val_ds = random_split(full_train, [train_size, val_size])

        # Also prepare the test set for final evaluation
        test_ds = registry.get_cifake(
            split="test",
            transform=val_transform,
            max_samples_per_class=args.max_samples,
        )

    elif args.dataset == "genimage":
        train_transform = get_train_transforms()
        val_transform = get_val_transforms()

        train_ds = registry.get_genimage(
            split="train",
            transform=train_transform,
            max_samples_per_class=args.max_samples,
        )
        val_ds = registry.get_genimage(
            split="val",
            transform=val_transform,
            max_samples_per_class=args.max_samples,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    logger.info(
        "training.data_loaded",
        dataset=args.dataset,
        train_samples=len(train_ds),
        val_samples=len(val_ds),
        train_batches=len(train_loader),
        val_batches=len(val_loader),
    )

    # Create model
    model = ImageClassifier(
        num_classes=1,
        dropout=0.4,
        freeze_backbone=True,
        unfreeze_last_n_blocks=2,
        pretrained=True,
    )

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        patience=3,
        output_dir=args.output_dir,
        model_name=f"{args.dataset}_classifier",
    )

    summary = trainer.train(args.epochs)

    # Print final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Dataset:       {args.dataset}")
    print(f"  Device:        {device}")
    print(f"  Epochs:        {summary['epochs_trained']}")
    print(f"  Best AUC-ROC:  {summary['best_auc']:.4f}")
    print(f"  Total Time:    {summary['total_time']}")
    print(f"  Checkpoint:    {args.output_dir}/{args.dataset}_classifier_best.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
