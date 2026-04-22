import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import sys
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Обязательно добавляем корень проекта в sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.vision.metrics import get_accuracy_difference
from src.vision.models import NetLoader
import config

class NBVDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        for class_id in range(config.NUM_CLASSES):
            class_dir = self.data_dir / f"{class_id:02d}"
            if class_dir.exists():
                for rgb_path in class_dir.glob("*_rgb.png"):
                    sample_name = rgb_path.name.replace("_rgb.png", "")
                    depth_path = class_dir / f"{sample_name}_depth.npy"
                    meta_path = class_dir / f"{sample_name}_meta.json"
                    if depth_path.exists() and meta_path.exists():
                        self.samples.append({
                            "rgb": rgb_path,
                            "depth": depth_path,
                            "meta": meta_path,
                            "label": class_id
                        })
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # 1. Load Image (RGB-D)
        rgb = np.array(Image.open(s["rgb"]).convert('RGB')).astype(np.float32) / 255.0 # [H, W, 3]
        depth = np.load(s["depth"]) # [H, W]
        depth_normalized = np.clip(depth / 10.0, 0, 1)
        
        # Stack RGB + D -> [4, H, W]
        image_4ch = np.concatenate([rgb.transpose(2,0,1), depth_normalized[np.newaxis, :, :]], axis=0)
        
        # 3. Load Vector (From meta)
        import json
        with open(s["meta"], "r") as f:
            meta = json.load(f)
        
        # Placeholder for 15D vector (matches environment)
        # In real training we'd need exact joint states, here we use cam_eye as base
        vector = np.zeros(15, dtype=np.float32)
        vector[:3] = meta["cam_eye"]
        vector[3:6] = meta["target_pos"]
        
        return torch.FloatTensor(image_4ch), torch.FloatTensor(vector), torch.LongTensor([s["label"]]).squeeze()

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

def train_cnn():
    dataset_dir = config.DATASET_DIR
    if not dataset_dir.exists():
        print(f"Dataset directory not found at {dataset_dir}. Run dataset.py first.")
        return
        
    dataset = NBVDataset(dataset_dir)
    if len(dataset) == 0:
        print("Dataset is empty.")
        return
        
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.CNN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.CNN_BATCH_SIZE, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = NetLoader.load(config.CNN_ARCHITECTURE, num_classes=config.NUM_CLASSES, vector_dim=15).to(device)
    if config.CNN_RESUME and config.CNN_MODEL_PATH.exists():
        print(f"Resuming training from {config.CNN_MODEL_PATH}...")
        model.load_state_dict(torch.load(config.CNN_MODEL_PATH, map_location=device))
    
    criterion_cls = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=config.CNN_LR)
    
    num_epochs = config.CNN_EPOCHS
    best_val_loss = float('inf')
    
    run_dir = config.get_run_dir("cnn_train")
    csv_path = run_dir / "cnn_training_log.csv"
    png_path = run_dir / "cnn_learning_curve.png"
    
    # Initialize empty CSV if starting fresh
    if not config.CNN_RESUME or not csv_path.exists():
        pd.DataFrame(columns=["Epoch", "Batch", "Phase", "Loss", "Loss_Cls", "Accuracy", "F1_Score", "Acc_Diff"]).to_csv(csv_path, index=False)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_metrics = []
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, (imgs, vecs, labels) in enumerate(train_pbar):
            imgs, vecs, labels = imgs.to(device), vecs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(imgs, vecs)
            if isinstance(logits, tuple): logits = logits[0]
            
            loss_cls = criterion_cls(logits, labels)
            
            loss = loss_cls
            loss.backward()
            optimizer.step()
            
            # Metrics calculation
            _, predicted = torch.max(logits.data, 1)
            batch_acc = 100 * (predicted == labels).sum().item() / max(labels.size(0), 1)
            
            preds_np = predicted.cpu().numpy()
            labels_np = labels.cpu().numpy()
            batch_f1 = f1_score(labels_np, preds_np, average='macro', zero_division=0)
            
            acc_diff = get_accuracy_difference(logits).mean().item()
            
            epoch_metrics.append({
                "Epoch": epoch + 1,
                "Batch": batch_idx,
                "Phase": "Train",
                "Loss": loss.item(),
                "Loss_Cls": loss_cls.item(),
                "Accuracy": batch_acc,
                "F1_Score": batch_f1,
                "Acc_Diff": acc_diff
            })
            
            running_loss += loss.item() * imgs.size(0)
            train_pbar.set_postfix(loss=f"{loss.item():.4f}", cls=f"{loss_cls.item():.2f}")
            
        epoch_loss = running_loss / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for batch_idx, (imgs, vecs, labels) in enumerate(val_pbar):
                imgs, vecs, labels = imgs.to(device), vecs.to(device), labels.to(device)
                logits = model(imgs, vecs)
                if isinstance(logits, tuple): logits = logits[0]
                
                l_cls = criterion_cls(logits, labels)
                batch_loss = l_cls
                val_loss += batch_loss.item() * imgs.size(0)
                
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                batch_correct = (predicted == labels).sum().item()
                correct += batch_correct
                
                # Metrics calculation
                batch_acc = 100 * batch_correct / max(labels.size(0), 1)
                
                preds_np = predicted.cpu().numpy()
                labels_np = labels.cpu().numpy()
                batch_f1 = f1_score(labels_np, preds_np, average='macro', zero_division=0)
                
                acc_diff = get_accuracy_difference(logits).mean().item()
                
                epoch_metrics.append({
                    "Epoch": epoch + 1,
                    "Batch": batch_idx,
                    "Phase": "Val",
                    "Loss": batch_loss.item(),
                    "Loss_Cls": l_cls.item(),
                    "Accuracy": batch_acc,
                    "F1_Score": batch_f1,
                    "Acc_Diff": acc_diff
                })
                
        avg_val_loss = val_loss / len(val_dataset)
        val_acc = 100 * correct / max(total, 1)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.2f}%")
        
        # Save metrics and plot
        pd.DataFrame(epoch_metrics).to_csv(csv_path, mode='a', header=False, index=False)
        plot_learning_curves(csv_path, png_path, config.PLOT_MOVING_AVERAGE_WINDOW)
        
        # Save best weights
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            config.CNN_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            
            torch.save(model.state_dict(), config.CNN_WEIGHTS_DIR / config.CNN_BEST_MODEL_NAME)
            torch.save(model.state_dict(), run_dir / config.CNN_BEST_MODEL_NAME)
            
        # Save last weights
        torch.save(model.state_dict(), config.CNN_WEIGHTS_DIR / config.CNN_LAST_MODEL_NAME)
        torch.save(model.state_dict(), run_dir / config.CNN_LAST_MODEL_NAME)
            
    print(f"Training complete.")
    print(f"Last model saved to {run_dir / config.CNN_LAST_MODEL_NAME}")
    print(f"Best model saved to {config.CNN_WEIGHTS_DIR / config.CNN_BEST_MODEL_NAME}")

if __name__ == "__main__":
    train_cnn()
