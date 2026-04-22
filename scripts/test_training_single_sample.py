"""
Test training on a single sample to verify the pipeline works.

This script:
1. Creates a minimal dataset (1 sample per class)
2. Trains LightweightODIN for 1 epoch
3. Verifies the model can forward pass
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import json

import config
from src.vision.models import NetLoader

class MinimalDataset(torch.utils.data.Dataset):
    """Minimal dataset for testing."""
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.samples = []

        # Find all samples
        for sample_dir in sorted(self.data_dir.glob("sample_*")):
            color_map_file = sample_dir / "color_map.json"
            if not color_map_file.exists():
                continue

            with open(color_map_file) as f:
                color_map = json.load(f)

            # Get class ID from color_map (target object is first entry)
            class_id = color_map[0]["category_id"] - 1  # Convert to 0-indexed

            # Get first view
            rgb_files = sorted((sample_dir / "rgb").glob("*.png"))
            if len(rgb_files) == 0:
                continue

            view_id = rgb_files[0].stem

            self.samples.append({
                "rgb": sample_dir / "rgb" / f"{view_id}.png",
                "depth": sample_dir / "depth" / f"{view_id}.npy",
                "label": class_id,
                "sample_dir": sample_dir
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Load RGB
        rgb = np.array(Image.open(s["rgb"]).convert('RGB')).astype(np.float32) / 255.0

        # Load depth
        depth = np.load(s["depth"])
        depth_normalized = np.clip(depth / 10.0, 0, 1)

        # Stack RGB + D -> [4, H, W]
        image_4ch = np.concatenate([rgb.transpose(2, 0, 1), depth_normalized[np.newaxis, :, :]], axis=0)

        # Dummy vector (15D)
        vector = np.zeros(15, dtype=np.float32)

        return torch.FloatTensor(image_4ch), torch.FloatTensor(vector), torch.LongTensor([s["label"]]).squeeze()

def test_training(dataset_dir, stage):
    """Test training on minimal dataset."""
    print(f"\n{'='*60}")
    print(f"Testing Training on Stage {stage} Dataset")
    print(f"{'='*60}")

    # Load dataset
    dataset = MinimalDataset(dataset_dir)
    print(f"Loaded {len(dataset)} samples")

    if len(dataset) == 0:
        print("[ERROR] No samples found!")
        return False

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=True)

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = NetLoader.load("LightweightODIN", num_classes=config.NUM_CLASSES, vector_dim=15).to(device)
    print(f"Model: LightweightODIN")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Test forward pass
    print("\n--- Testing Forward Pass ---")
    model.eval()
    with torch.no_grad():
        for imgs, vecs, labels in dataloader:
            imgs, vecs, labels = imgs.to(device), vecs.to(device), labels.to(device)
            logits = model(imgs, vecs)
            print(f"Input shape: {imgs.shape}")
            print(f"Vector shape: {vecs.shape}")
            print(f"Output shape: {logits.shape}")
            print(f"Labels: {labels.cpu().numpy()}")
            print(f"Predictions: {torch.argmax(logits, dim=1).cpu().numpy()}")
            break

    # Test training step
    print("\n--- Testing Training Step ---")
    model.train()

    for epoch in range(2):  # 2 epochs
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (imgs, vecs, labels) in enumerate(dataloader):
            imgs, vecs, labels = imgs.to(device), vecs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs, vecs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            print(f"Epoch {epoch+1}, Batch {batch_idx+1}: Loss={loss.item():.4f}")

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}: Loss={epoch_loss/len(dataloader):.4f}, Accuracy={accuracy:.2f}%")

    print("\n[OK] Training test passed!")
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3], help="Stage to test")
    args = parser.parse_args()

    dataset_dir = config.BASE_DIR / "dataset" / config.OBJECT_MODE / f"stage{args.stage}"

    if not dataset_dir.exists():
        print(f"[ERROR] Dataset not found: {dataset_dir}")
        print(f"Run: uv run python scripts/prepare_stage_datasets.py --stage {args.stage} --samples 8")
        sys.exit(1)

    success = test_training(dataset_dir, args.stage)
    sys.exit(0 if success else 1)
