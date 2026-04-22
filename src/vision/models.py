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

class LightweightODIN(nn.Module):
    """Lightweight ODIN-inspired architecture for CPU inference with Bayesian uncertainty estimation.

    Key simplifications from full ODIN:
    - Replaced MSDeformAttn with standard convolutions
    - Replaced Transformer decoder with lightweight attention
    - Single-view processing (no cross-view attention)
    - Smaller feature dimensions
    - MC Dropout for uncertainty estimation
    """
    def __init__(self, num_classes=config.NUM_CLASSES, vector_dim=15, hidden_dim=128, dropout_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # 1. Hierarchical Image Encoder (RGB-D 4 channels)
        self.enc_conv1 = nn.Sequential(Unit(4, 32), Unit(32, 32), nn.MaxPool2d(2))  # 112x112
        self.enc_conv2 = nn.Sequential(Unit(32, 64), Unit(64, 64), nn.MaxPool2d(2))  # 56x56
        self.enc_conv3 = nn.Sequential(Unit(64, 128), Unit(128, 128), nn.MaxPool2d(2))  # 28x28
        self.enc_conv4 = nn.Sequential(Unit(128, 256), Unit(256, 256), nn.MaxPool2d(2))  # 14x14

        # 2. Lightweight Pixel Decoder (replaces MSDeformAttn)
        self.pixel_decoder = nn.Sequential(
            nn.Conv2d(256, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)  # Spatial dropout for uncertainty
        )

        # 3. Query Embeddings (learnable object queries like DETR/ODIN)
        self.num_queries = num_classes
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)

        # 4. Lightweight Self-Attention (replaces full Transformer)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout_rate, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout_rate, batch_first=True)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout_rate)
        )

        # 5. Vector Encoder (robot state)
        self.vector_enc = nn.Sequential(
            nn.Linear(vector_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, hidden_dim),
            nn.Dropout(dropout_rate)
        )

        # 6. Classification Head with dropout
        self.class_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )

        # Compatibility: Add global_pool for evaluate.py hook registration
        # This is a dummy layer that won't be used in forward pass
        self.global_pool = nn.Identity()

    def forward(self, image, vector, mc_samples=1):
        """Forward pass with optional MC Dropout sampling.

        Args:
            image: [B, 4, H, W] RGB-D input
            vector: [B, vector_dim] robot state
            mc_samples: Number of MC dropout samples for uncertainty estimation
                       If mc_samples=1, standard forward pass
                       If mc_samples>1, returns mean and std of predictions

        Returns:
            If mc_samples=1: logits [B, num_queries]
            If mc_samples>1: (mean_logits, std_logits) both [B, num_queries]
        """
        if mc_samples == 1:
            return self._forward_single(image, vector)
        else:
            return self._forward_mc_dropout(image, vector, mc_samples)

    def _forward_single(self, image, vector):
        """Single forward pass."""
        B = image.size(0)

        # Encode image through hierarchical CNN
        x = self.enc_conv1(image)
        x = self.enc_conv2(x)
        x = self.enc_conv3(x)
        feat_map = self.enc_conv4(x)  # [B, 256, 14, 14]

        # Pixel decoder
        pixel_feat = self.pixel_decoder(feat_map)  # [B, hidden_dim, 14, 14]

        # Flatten spatial dimensions for attention
        H, W = pixel_feat.shape[2:]
        pixel_feat_flat = pixel_feat.flatten(2).permute(0, 2, 1)  # [B, H*W, hidden_dim]

        # Query embeddings
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, num_queries, hidden_dim]

        # Self-attention on queries
        queries2, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + self.dropout1(queries2))

        # Cross-attention: queries attend to image features
        queries2, _ = self.cross_attn(queries, pixel_feat_flat, pixel_feat_flat)
        queries = self.norm2(queries + self.dropout2(queries2))

        # FFN
        queries = queries + self.ffn(queries)

        # Encode vector (robot state) and fuse
        vec_feat = self.vector_enc(vector).unsqueeze(1)  # [B, 1, hidden_dim]
        queries = queries + vec_feat  # Broadcast addition

        # Classification: which query corresponds to the target object?
        logits = self.class_head(queries).squeeze(-1)  # [B, num_queries]

        return logits

    def _forward_mc_dropout(self, image, vector, mc_samples):
        """MC Dropout forward pass for uncertainty estimation."""
        self.train()  # Enable dropout during inference

        logits_samples = []
        for _ in range(mc_samples):
            logits = self._forward_single(image, vector)
            logits_samples.append(logits)

        logits_samples = torch.stack(logits_samples, dim=0)  # [mc_samples, B, num_queries]

        mean_logits = logits_samples.mean(dim=0)  # [B, num_queries]
        std_logits = logits_samples.std(dim=0)    # [B, num_queries]

        return mean_logits, std_logits

    def predict_with_uncertainty(self, image, vector, mc_samples=20):
        """Predict with uncertainty estimation using MC Dropout.

        Args:
            image: [B, 4, H, W] RGB-D input
            vector: [B, vector_dim] robot state
            mc_samples: Number of MC dropout samples (default 20)

        Returns:
            mean_probs: [B, num_queries] mean predicted probabilities
            std_probs: [B, num_queries] std of predicted probabilities (epistemic uncertainty)
            entropy: [B] predictive entropy (total uncertainty)
        """
        mean_logits, std_logits = self.forward(image, vector, mc_samples=mc_samples)

        # Convert to probabilities
        mean_probs = torch.softmax(mean_logits, dim=-1)

        # Epistemic uncertainty (model uncertainty)
        std_probs = std_logits  # Std of logits as uncertainty measure

        # Predictive entropy (total uncertainty)
        entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)

        return mean_probs, std_probs, entropy

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

        return logits

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

class TNet(nn.Module):
    """T-Net for learning transformation matrix (3x3 or KxK)."""
    def __init__(self, k=3):
        super().__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=False)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        identity = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(batch_size, 1)
        x = x + identity
        x = x.view(-1, self.k, self.k)

        return x

class PointNet(nn.Module):
    """PointNet architecture for point cloud classification.

    Input: Point cloud [B, 3, N] where N is number of points
    Output: Class logits [B, num_classes]

    Features:
    - Input transform (T-Net) for rotation invariance
    - Feature transform (T-Net) for feature space alignment
    - Symmetric max pooling for permutation invariance
    - Optional vector fusion for robot state
    """
    def __init__(self, num_classes=config.NUM_CLASSES, vector_dim=15, num_points=1024, use_vector=True):
        super().__init__()
        self.num_points = num_points
        self.use_vector = use_vector

        # Input transform
        self.input_transform = TNet(k=3)

        # MLP 1
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        # Feature transform
        self.feature_transform = TNet(k=64)

        # MLP 2
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        # Vector encoder (robot state)
        if self.use_vector:
            self.vector_enc = nn.Sequential(
                nn.Linear(vector_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.ReLU()
            )
            fc_input_dim = 1024 + 128
        else:
            fc_input_dim = 1024

        # Classification head
        self.fc1 = nn.Linear(fc_input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

        # Compatibility: Add global_pool for evaluate.py hook registration
        self.global_pool = nn.Identity()

    def forward(self, points, vector=None):
        """Forward pass.

        Args:
            points: [B, 3, N] point cloud (x, y, z coordinates)
            vector: [B, vector_dim] robot state (optional)

        Returns:
            logits: [B, num_classes]
        """
        batch_size = points.size(0)

        # Input transform
        trans = self.input_transform(points)
        points = torch.bmm(trans, points)

        # MLP 1
        x = F.relu(self.bn1(self.conv1(points)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Feature transform
        trans_feat = self.feature_transform(x)
        x = torch.bmm(trans_feat, x)

        # MLP 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # Symmetric max pooling
        x = torch.max(x, 2, keepdim=False)[0]

        # Fuse with vector if provided
        if self.use_vector and vector is not None:
            vec_feat = self.vector_enc(vector)
            x = torch.cat([x, vec_feat], dim=1)

        # Classification
        x = F.relu(self.bn6(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

class NetLoader:
    @staticmethod
    def load(arch="BasicNet", num_classes=config.NUM_CLASSES, vector_dim=15):
        if arch == "BasicNet" or arch == "MultiModalNet":
            return MultiModalNet(num_classes, vector_dim)
        elif arch == "SimpleNet":
            return SimpleNet(num_classes)
        elif arch == "LightweightODIN":
            return LightweightODIN(num_classes, vector_dim)
        elif arch == "PointNet":
            return PointNet(num_classes, vector_dim)
        else:
            raise ValueError(f"Unknown architecture: {arch}")

