"""
Multi-frame training for Bayesian segmentation model.

Trains the model on variable numbers of input frames (1-5) to handle
sequential observation scenarios where the agent receives frames one at a time.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
import sys
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from PIL import Image
import random

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.vision.metrics import get_accuracy_difference
from src.vision.models import NetLoader
import config


class MultiFrameDataset(Dataset):
    """Dataset that loads variable numbers of frames (1-5) from stage datasets."""

    def __init__(self, data_dir, max_frames=5, min_frames=1):
        self.data_dir = Path(data_dir)
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.samples = []

        # Collect all samples
        for sample_dir in sorted(self.data_dir.glob("sample_*")):
            if not sample_dir.is_dir():
                continue

            # Check if sample has required files
            cameras_file = sample_dir / "cameras.json"
            color_map_file = sample_dir / "color_map.json"

            if not cameras_file.exists() or not color_map_file.exists():
                continue

            # Load metadata
            with open(cameras_file) as f:
                cameras = json.load(f)
            with open(color_map_file) as f:
                color_map = json.load(f)

            # Get target object class (first object in color_map)
            target_class = None
            for entry in color_map:
                if entry.get("category_name") == "target_object":
                    target_class = entry.get("category_id", 1) - 1  # Convert to 0-indexed
                    break

            if target_class is None:
                continue

            # Get available views
            view_ids = list(cameras.keys())
            if len(view_ids) < self.min_frames:
                continue

            self.samples.append({
                "sample_dir": sample_dir,
                "view_ids": view_ids,
                "label": target_class,
                "cameras": cameras
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Randomly select number of frames (1 to max_frames)
        num_frames = random.randint(self.min_frames, min(self.max_frames, len(sample["view_ids"])))

        # Randomly select which frames to use
        selected_views = random.sample(sample["view_ids"], num_frames)

        # Load frames
        frames = []
        vectors = []

        for view_id in selected_views:
            # Load RGB
            rgb_path = sample["sample_dir"] / "rgb" / f"{view_id}.png"
            rgb = np.array(Image.open(rgb_path).convert('RGB')).astype(np.float32) / 255.0

            # Load depth
            depth_path = sample["sample_dir"] / "depth" / f"{view_id}.npy"
            depth = np.load(depth_path)
            depth_normalized = np.clip(depth / 10.0, 0, 1)

            # Stack RGB + D -> [4, H, W]
            image_4ch = np.concatenate([rgb.transpose(2, 0, 1), depth_normalized[np.newaxis, :, :]], axis=0)
            frames.append(image_4ch)

            # Create vector from camera data
            cam_data = sample["cameras"][view_id]
            vector = np.zeros(15, dtype=np.float32)
            vector[:3] = cam_data["position"]
            vector[3:6] = cam_data["target"]
            vectors.append(vector)

        # Stack frames: [num_frames, 4, H, W]
        frames = np.stack(frames, axis=0)
        vectors = np.stack(vectors, axis=0)

        return (
            torch.FloatTensor(frames),
            torch.FloatTensor(vectors),
            torch.LongTensor([sample["label"]]).squeeze(),
            num_frames
        )


def collate_multiframe(batch):
    """Custom collate function to handle variable frame counts.

    Pads sequences to max length in batch.
    """
    frames_list, vectors_list, labels_list, num_frames_list = zip(*batch)

    # Find max frames in this batch
    max_frames_batch = max(num_frames_list)

    # Pad sequences
    padded_frames = []
    padded_vectors = []
    masks = []

    for frames, vectors, num_frames in zip(frames_list, vectors_list, num_frames_list):
        # Pad frames
        if num_frames < max_frames_batch:
            pad_size = max_frames_batch - num_frames
            frames_pad = torch.zeros(pad_size, *frames.shape[1:])
            frames = torch.cat([frames, frames_pad], dim=0)

            vectors_pad = torch.zeros(pad_size, *vectors.shape[1:])
            vectors = torch.cat([vectors, vectors_pad], dim=0)

        # Create mask (1 for real frames, 0 for padding)
        mask = torch.zeros(max_frames_batch)
        mask[:num_frames] = 1

        padded_frames.append(frames)
        padded_vectors.append(vectors)
        masks.append(mask)

    return (
        torch.stack(padded_frames),  # [B, max_frames, 4, H, W]
        torch.stack(padded_vectors),  # [B, max_frames, 15]
        torch.stack(labels_list),     # [B]
        torch.stack(masks)            # [B, max_frames]
    )


class SequentialWrapper(nn.Module):
    """Wrapper to process multiple frames sequentially through a single-frame model."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, frames, vectors, mask):
        """
        Args:
            frames: [B, T, 4, H, W] where T is max frames in batch
            vectors: [B, T, 15]
            mask: [B, T] binary mask (1 for real frames, 0 for padding)

        Returns:
            logits: [B, num_classes] aggregated predictions
        """
        B, T = frames.shape[:2]

        # Process each frame
        all_logits = []
        for t in range(T):
            frame_t = frames[:, t]  # [B, 4, H, W]
            vector_t = vectors[:, t]  # [B, 15]

            logits_t = self.base_model(frame_t, vector_t)  # [B, num_classes]
            all_logits.append(logits_t)

        all_logits = torch.stack(all_logits, dim=1)  # [B, T, num_classes]

        # Mask out padding frames
        mask_expanded = mask.unsqueeze(-1)  # [B, T, 1]
        all_logits = all_logits * mask_expanded

        # Average over valid frames
        num_valid = mask.sum(dim=1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]
        num_valid = torch.clamp(num_valid, min=1)  # Avoid division by zero

        aggregated_logits = all_logits.sum(dim=1) / num_valid.squeeze(1)  # [B, num_classes]

        return aggregated_logits


def plot_learning_curves(csv_path, png_path, window_size):
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return

    metrics = ["Loss", "Accuracy", "F1_Score", "Acc_Diff"]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    phases = df['Phase'].unique()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for phase in phases:
            phase_df = df[df['Phase'] == phase].reset_index(drop=True)
            if phase_df.empty:
                continue
            ax.plot(phase_df.index, phase_df[metric], alpha=0.3, label=f"{phase} {metric} (raw)")
            ma = phase_df[metric].rolling(window=window_size, min_periods=1).mean()
            ax.plot(phase_df.index, ma, linewidth=2, label=f"{phase} {metric} (MA {window_size})")
        ax.set_title(metric)
        ax.set_xlabel("Batch Step")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()


def train_multiframe(stage=None, max_frames=5, min_frames=1):
    """Train model on multi-frame sequences.

    Args:
        stage: Training stage (1, 2, or 3). If None, uses config.SCENE_STAGE
        max_frames: Maximum number of frames per sample
        min_frames: Minimum number of frames per sample
    """
    if stage is None:
        stage = config.SCENE_STAGE

    dataset_dir = config.BASE_DIR / "dataset" / config.OBJECT_MODE / f"stage{stage}"

    if not dataset_dir.exists():
        print(f"Dataset directory not found at {dataset_dir}.")
        print(f"Run: python scripts/prepare_stage_datasets.py --stage {stage}")
        return

    print(f"Training on Stage {stage} dataset with {min_frames}-{max_frames} frames")

    dataset = MultiFrameDataset(dataset_dir, max_frames=max_frames, min_frames=min_frames)
    if len(dataset) == 0:
        print("Dataset is empty.")
        return

    print(f"Found {len(dataset)} samples")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.CNN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_multiframe
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.CNN_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_multiframe
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load base model and wrap it
    base_model = NetLoader.load(config.CNN_ARCHITECTURE, num_classes=config.NUM_CLASSES, vector_dim=15)
    model = SequentialWrapper(base_model).to(device)

    if config.CNN_RESUME and config.CNN_MODEL_PATH.exists():
        print(f"Resuming training from {config.CNN_MODEL_PATH}...")
        model.base_model.load_state_dict(torch.load(config.CNN_MODEL_PATH, map_location=device))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.CNN_LR)

    num_epochs = config.CNN_EPOCHS
    best_val_loss = float('inf')

    run_dir = config.get_run_dir(f"multiframe_train_stage{stage}")
    csv_path = run_dir / "training_log.csv"
    png_path = run_dir / "learning_curve.png"

    # Initialize CSV
    if not config.CNN_RESUME or not csv_path.exists():
        pd.DataFrame(columns=["Epoch", "Batch", "Phase", "Loss", "Accuracy", "F1_Score", "Acc_Diff"]).to_csv(csv_path, index=False)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_metrics = []

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, (frames, vectors, labels, masks) in enumerate(train_pbar):
            frames = frames.to(device)
            vectors = vectors.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(frames, vectors, masks)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # Metrics
            _, predicted = torch.max(logits.data, 1)
            batch_acc = 100 * (predicted == labels).sum().item() / labels.size(0)

            preds_np = predicted.cpu().numpy()
            labels_np = labels.cpu().numpy()
            batch_f1 = f1_score(labels_np, preds_np, average='macro', zero_division=0)

            acc_diff = get_accuracy_difference(logits).mean().item()

            epoch_metrics.append({
                "Epoch": epoch + 1,
                "Batch": batch_idx,
                "Phase": "Train",
                "Loss": loss.item(),
                "Accuracy": batch_acc,
                "F1_Score": batch_f1,
                "Acc_Diff": acc_diff
            })

            running_loss += loss.item() * labels.size(0)
            train_pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.1f}%")

        epoch_loss = running_loss / len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for batch_idx, (frames, vectors, labels, masks) in enumerate(val_pbar):
                frames = frames.to(device)
                vectors = vectors.to(device)
                labels = labels.to(device)
                masks = masks.to(device)

                logits = model(frames, vectors, masks)
                batch_loss = criterion(logits, labels)
                val_loss += batch_loss.item() * labels.size(0)

                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                batch_correct = (predicted == labels).sum().item()
                correct += batch_correct

                # Metrics
                batch_acc = 100 * batch_correct / labels.size(0)
                preds_np = predicted.cpu().numpy()
                labels_np = labels.cpu().numpy()
                batch_f1 = f1_score(labels_np, preds_np, average='macro', zero_division=0)
                acc_diff = get_accuracy_difference(logits).mean().item()

                epoch_metrics.append({
                    "Epoch": epoch + 1,
                    "Batch": batch_idx,
                    "Phase": "Val",
                    "Loss": batch_loss.item(),
                    "Accuracy": batch_acc,
                    "F1_Score": batch_f1,
                    "Acc_Diff": acc_diff
                })

        avg_val_loss = val_loss / len(val_dataset)
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.2f}%")

        # Save metrics and plot
        pd.DataFrame(epoch_metrics).to_csv(csv_path, mode='a', header=False, index=False)
        plot_learning_curves(csv_path, png_path, config.PLOT_MOVING_AVERAGE_WINDOW)

        # Save best weights
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            weights_dir = config.CNN_WEIGHTS_DIR / f"stage{stage}"
            weights_dir.mkdir(parents=True, exist_ok=True)

            torch.save(model.base_model.state_dict(), weights_dir / f"multiframe_best_stage{stage}.pt")
            torch.save(model.base_model.state_dict(), run_dir / f"multiframe_best_stage{stage}.pt")

        # Save last weights
        weights_dir = config.CNN_WEIGHTS_DIR / f"stage{stage}"
        weights_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.base_model.state_dict(), weights_dir / f"multiframe_last_stage{stage}.pt")
        torch.save(model.base_model.state_dict(), run_dir / f"multiframe_last_stage{stage}.pt")

    print(f"Training complete.")
    print(f"Best model saved to {weights_dir / f'multiframe_best_stage{stage}.pt'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], help="Training stage")
    parser.add_argument("--max-frames", type=int, default=5, help="Maximum frames per sample")
    parser.add_argument("--min-frames", type=int, default=1, help="Minimum frames per sample")
    args = parser.parse_args()

    train_multiframe(stage=args.stage, max_frames=args.max_frames, min_frames=args.min_frames)
