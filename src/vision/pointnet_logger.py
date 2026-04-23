"""
PointNet training metrics logger.

Logs training metrics to CSV file for easy analysis and Kaggle download.
"""

import csv
from pathlib import Path


class PointNetLogger:
    """Logger for PointNet training metrics."""

    def __init__(self, output_dir):
        """Initialize logger.

        Args:
            output_dir: Directory to save CSV file
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "pointnet_metrics.csv"

        # Create CSV with header if doesn't exist
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'learning_rate'])
            print(f"Created metrics CSV: {self.csv_path}")

    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        """Log metrics for one epoch.

        Args:
            epoch: Epoch number
            train_loss: Training loss
            train_acc: Training accuracy (%)
            val_loss: Validation loss
            val_acc: Validation accuracy (%)
            lr: Current learning rate
        """
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, lr])
