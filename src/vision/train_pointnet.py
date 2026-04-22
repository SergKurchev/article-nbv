"""Train PointNet on point cloud dataset.

Usage:
    uv run python src/vision/train_pointnet.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import config
from tqdm import tqdm
from src.vision.models import PointNet


class PointCloudDataset(Dataset):
    """Dataset for PointNet training."""

    def __init__(self, dataset_dir, split='train', val_split=0.2):
        """Initialize dataset.

        Args:
            dataset_dir: Path to dataset directory
            split: 'train' or 'val'
            val_split: Fraction of data for validation
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split

        # Get all samples
        all_samples = sorted([d for d in self.dataset_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")])

        # Split into train/val
        num_val = int(len(all_samples) * val_split)
        if split == 'val':
            self.samples = all_samples[:num_val]
        else:
            self.samples = all_samples[num_val:]

        print(f"{split.upper()} dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]

        # Load point clouds [num_views, num_points, 3]
        point_clouds = np.load(sample_dir / "point_clouds.npy")

        # Load vectors [num_views, 15]
        vectors = np.load(sample_dir / "vectors.npy")

        # Load metadata
        with open(sample_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)

        class_id = metadata['class_id']

        # Randomly select one view
        view_idx = np.random.randint(0, len(point_clouds))
        points = point_clouds[view_idx]  # [num_points, 3]
        vector = vectors[view_idx]  # [15]

        # Transpose to [3, num_points] for PointNet
        points = points.T

        return {
            'points': torch.from_numpy(points).float(),
            'vector': torch.from_numpy(vector).float(),
            'label': torch.tensor(class_id, dtype=torch.long)
        }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        points = batch['points'].to(device)
        vectors = batch['vector'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(points, vectors)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            points = batch['points'].to(device)
            vectors = batch['vector'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            logits = model(points, vectors)
            loss = criterion(logits, labels)

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def train_pointnet():
    """Train PointNet model."""
    print("="*60)
    print("PointNet Training")
    print("="*60)

    # Check dataset
    dataset_dir = config.BASE_DIR / "dataset_pointnet" / config.OBJECT_MODE
    if not dataset_dir.exists():
        print(f"Error: PointNet dataset not found: {dataset_dir}")
        print("Please run 'uv run python src/vision/dataset_pointnet.py' first.")
        return

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create datasets
    train_dataset = PointCloudDataset(dataset_dir, split='train', val_split=config.CNN_VAL_SPLIT)
    val_dataset = PointCloudDataset(dataset_dir, split='val', val_split=config.CNN_VAL_SPLIT)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.POINTNET_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.POINTNET_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Create model
    model = PointNet(
        num_classes=config.NUM_CLASSES,
        vector_dim=15,
        num_points=config.POINTNET_NUM_POINTS,
        use_vector=True
    ).to(device)

    print(f"\nModel: PointNet")
    print(f"Number of classes: {config.NUM_CLASSES}")
    print(f"Points per sample: {config.POINTNET_NUM_POINTS}")
    print(f"Use vector: True")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.POINTNET_LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Training loop
    best_val_acc = 0.0
    output_dir = config.CNN_WEIGHTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for {config.POINTNET_EPOCHS} epochs...")
    print(f"Batch size: {config.POINTNET_BATCH_SIZE}")
    print(f"Learning rate: {config.POINTNET_LR}")
    print(f"Output directory: {output_dir}")

    for epoch in range(config.POINTNET_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.POINTNET_EPOCHS}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Step scheduler
        scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "pointnet_best.pt")
            print(f"  Saved best model (val_acc: {val_acc:.2f}%)")

        # Save last model
        torch.save(model.state_dict(), output_dir / "pointnet_last.pt")

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train_pointnet()
