import time
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils import ExperimentTracker, setup_logger


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        device: str,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = device

        # TODO: Define Loss Function (Criterion)
        self.criterion = nn.CrossEntropyLoss()

        # TODO: Initialize ExperimentTracker
        self.tracker = ExperimentTracker(experiment_name=config.get("experiment_name", "mlops_experiment"), config=config)

        # TODO: Initialize metric calculation (like accuracy/f1-score) if needed
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, dataloader: DataLoader, epoch_idx: int) -> Tuple[float, float, float]:
        self.model.train()
        
        # TODO: Implement Training Loop
        # 1. Iterate over dataloader
        # 2. Move data to device
        # 3. Forward pass, Calculate Loss
        # 4. Backward pass, Optimizer step
        # 5. Track metrics (Loss, Accuracy, F1)
        
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch_idx} [Train]")):
            images = images.to(self.device)
            labels = labels.to(self.device).squeeze()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        self.train_losses.append(epoch_loss)
        return epoch_loss, epoch_acc, 0.0

    def validate(self, dataloader: DataLoader, epoch_idx: int) -> Tuple[float, float, float]:
        self.model.eval()
        
        # TODO: Implement Validation Loop
        # Remember: No gradients needed here

        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch_idx} [Val]")):
                images = images.to(self.device)
                labels = labels.to(self.device).squeeze()

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        self.val_losses.append(epoch_loss)
        return epoch_loss, epoch_acc, 0.0

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        # TODO: Save model state, optimizer state, and config
        save_dir = self.config.get("save_dir", "checkpoints")
        os.makedirs(save_dir, exist_ok=True)

        checkpoint_path = os.path.join(save_dir, f"model_epoch{epoch}_valloss{val_loss:.4f}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        epochs = self.config["training"]["epochs"]
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            train_loss, train_acc, _ = self.train_epoch(train_loader, epoch)
            val_loss, val_acc, _ = self.validate(val_loader, epoch)

            self.tracker.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc
            )

            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

            self.save_checkpoint(epoch, val_loss)

        print("Training complete.")
