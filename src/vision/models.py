import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class MultiModalNet(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES, vector_dim=15):
        super().__init__()
        
        # 1. Image Encoder (RGB-D 4 channels)
        self.enc_conv1 = nn.Sequential(Unit(4, 32), Unit(32, 32), nn.MaxPool2d(2)) # 112x112
        self.enc_conv2 = nn.Sequential(Unit(32, 64), Unit(64, 64), nn.MaxPool2d(2)) # 56x56
        self.enc_conv3 = nn.Sequential(Unit(64, 128), Unit(128, 128), nn.MaxPool2d(2)) # 28x28
        self.enc_conv4 = nn.Sequential(Unit(128, 256), Unit(256, 256), nn.MaxPool2d(2)) # 14x14
        
        # 2. Vector Encoder (MLP for poses/joints)
        self.vector_enc = nn.Sequential(
            nn.Linear(vector_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # 3. Fusion & Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        # 4. Segmentation Decoder
        self.dec_up1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # 28x28
        self.dec_up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 56x56
        self.dec_up3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)   # 112x112
        self.dec_up4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)   # 224x224
        self.seg_head = nn.Conv2d(16, 1, kernel_size=1) 

    def forward(self, image, vector):
        x = self.enc_conv1(image)
        x = self.enc_conv2(x)
        x = self.enc_conv3(x)
        feat_map = self.enc_conv4(x)
        
        # Classification
        img_feat = self.global_pool(feat_map).view(feat_map.size(0), -1)
        vec_feat = self.vector_enc(vector)
        fused = torch.cat([img_feat, vec_feat], dim=1)
        logits = self.fc(fused)
        
        # Segmentation
        d = F.relu(self.dec_up1(feat_map))
        d = F.relu(self.dec_up2(d))
        d = F.relu(self.dec_up3(d))
        d = F.relu(self.dec_up4(d))
        mask = self.seg_head(d)
        
        return logits, mask

class SimpleNet(nn.Module):
    # Updated to 4 channels
    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(16, num_classes)
        )
    def forward(self, x, vector=None):
        return self.net(x)

class NetLoader:
    @staticmethod
    def load(arch="BasicNet", num_classes=config.NUM_CLASSES, vector_dim=15):
        if arch == "BasicNet" or arch == "MultiModalNet":
            return MultiModalNet(num_classes, vector_dim)
        elif arch == "SimpleNet":
            return SimpleNet(num_classes)
        else:
            raise ValueError(f"Unknown architecture {arch}")
