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

# Обязательно добавляем корень проекта в sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

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
                    mask_path = class_dir / f"{sample_name}_mask.png"
                    meta_path = class_dir / f"{sample_name}_meta.json"
                    
                    if depth_path.exists() and mask_path.exists() and meta_path.exists():
                        self.samples.append({
                            "rgb": rgb_path,
                            "depth": depth_path,
                            "mask": mask_path,
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
        
        # 2. Load Mask [1, H, W]
        mask = np.array(Image.open(s["mask"]).convert('L')).astype(np.float32) / 255.0
        mask = mask[np.newaxis, :, :] 
        
        # 3. Load Vector (From meta)
        import json
        with open(s["meta"], "r") as f:
            meta = json.load(f)
        
        # Placeholder for 15D vector (matches environment)
        # In real training we'd need exact joint states, here we use cam_eye as base
        vector = np.zeros(15, dtype=np.float32)
        vector[:3] = meta["cam_eye"]
        vector[3:6] = meta["target_pos"]
        # 나머지 9개는 0으로 채움 (joints etc)
        
        return torch.FloatTensor(image_4ch), torch.FloatTensor(vector), torch.LongTensor([s["label"]]).squeeze(), torch.FloatTensor(mask)

def train_cnn():
    dataset_dir = config.BASE_DIR / "dataset"
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
    
    model = NetLoader.load("MultiModalNet", vector_dim=15).to(device)
    if config.CNN_RESUME and config.CNN_MODEL_PATH.exists():
        print(f"Resuming training from {config.CNN_MODEL_PATH}...")
        model.load_state_dict(torch.load(config.CNN_MODEL_PATH, map_location=device))
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_seg = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=config.CNN_LR)
    
    num_epochs = config.CNN_EPOCHS
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for imgs, vecs, labels, masks in train_pbar:
            imgs, vecs, labels, masks = imgs.to(device), vecs.to(device), labels.to(device), masks.to(device)
            
            optimizer.zero_grad()
            logits, pred_masks = model(imgs, vecs)
            
            loss_cls = criterion_cls(logits, labels)
            loss_seg = criterion_seg(pred_masks, masks)
            
            loss = loss_cls + loss_seg
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            train_pbar.set_postfix(loss=f"{loss.item():.4f}", cls=f"{loss_cls.item():.2f}", seg=f"{loss_seg.item():.2f}")
            
        epoch_loss = running_loss / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for imgs, vecs, labels, masks in val_pbar:
                imgs, vecs, labels, masks = imgs.to(device), vecs.to(device), labels.to(device), masks.to(device)
                logits, pred_masks = model(imgs, vecs)
                
                l_cls = criterion_cls(logits, labels)
                l_seg = criterion_seg(pred_masks, masks)
                val_loss += (l_cls + l_seg).item() * imgs.size(0)
                
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        avg_val_loss = val_loss / len(val_dataset)
        val_acc = 100 * correct / max(total, 1)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.2f}%")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            config.CNN_WEIGHTS_DIR.mkdir(exist_ok=True)
            torch.save(model.state_dict(), config.CNN_MODEL_PATH)
            
    print(f"Training complete. Best model saved to {config.CNN_MODEL_PATH}")

if __name__ == "__main__":
    train_cnn()
