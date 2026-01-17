import csv
from pathlib import Path
from typing import Any, Dict
import yaml
from torch.utils.tensorboard import SummaryWriter

class ExperimentTracker:
    def __init__(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        base_dir: str = "experiments/results",
    ):
        self.run_dir = Path(base_dir) / experiment_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # TODO: Save config to yaml in run_dir
        if config is not None:
            config_path = self.run_dir / "config.yaml"
            with open(config_path, "w") as f:
                yaml.safe_dump(config, f)
        # Initialize CSV
        self.csv_path = self.run_dir / "metrics.csv"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        self.csv_writer.writerow([
            "epoch",
            "train_loss",
            "train_acc",
            "train_f1",
            "val_loss",
            "val_acc",
            "val_f1",
            "grad_norm",
        ])

        # tensorboard
        self.writer = SummaryWriter(log_dir=str(self.run_dir / "tb_logs"))

        # Header (TODO: add the rest of things we want to track, loss, gradients, accuracy etc.)
        self.csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]) 

    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """
        Writes metrics to CSV (and TensorBoard).
        """
        # TODO: Write other useful metrics to CSV
        self.csv_writer.writerow([
            epoch,
            metrics.get("train_loss"),
            metrics.get("train_acc"),
            metrics.get("train_f1"),
            metrics.get("val_loss"),
            metrics.get("val_acc"),
            metrics.get("val_f1"),
            metrics.get("grad_norm"),
        ])
        self.csv_file.flush()

        # TODO: Log to TensorBoard
        for key, value in metrics.items():
            if value is not None:
                self.writer.add_scalar(key.replace("_", "/"), value, epoch)

    def get_checkpoint_path(self, filename: str) -> str:
        return str(self.run_dir / filename)

    def close(self):
        self.csv_file.close()
        self.writer.close()
