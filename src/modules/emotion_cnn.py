"""
emotion_cnn.py
--------------
Lightweight CNN for facial emotion feature extraction.
Architecture: 4-block depthwise-separable CNN → global avg pool → 512-D embedding.
Designed for real-time use; ~0.8M parameters, ~5ms on CPU per frame.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise-separable convolution block.
    Much cheaper than standard conv while retaining representational power.
    Depthwise: applies a single filter per input channel.
    Pointwise: 1x1 conv to project to output channels.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=stride, padding=1,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block — lightweight channel attention.
    Globally pools spatial info, learns channel importance weights.
    This is the 'attention mechanism' that focuses on key facial regions.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w.expand_as(x)


class EmotionCNN(nn.Module):
    """
    Lightweight CNN for real-time emotion feature extraction.

    Input : (B, 1, 48, 48) — grayscale face crop
    Output: (B, 512)        — L2-normalised embedding used in CAG lookup

    Block structure:
      Block 1: 1→32   ch, stride 1  → 48x48 feature map
      Block 2: 32→64  ch, stride 2  → 24x24 feature map
      Block 3: 64→128 ch, stride 2  → 12x12 feature map
      Block 4: 128→256 ch, stride 2  → 6x6  feature map
      SE attention on 256-ch maps
      Global average pool → 256-D
      FC projection 256→512 → L2 norm
    """
    def __init__(self, embedding_dim: int = 512, num_emotions: int = 7):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_emotions = num_emotions

        # Entry conv (standard) — captures low-level edges/textures
        self.entry = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        # DS-Conv blocks
        self.block1 = DepthwiseSeparableConv(32, 32, stride=1)
        self.block2 = DepthwiseSeparableConv(32, 64, stride=2)
        self.block3 = DepthwiseSeparableConv(64, 128, stride=2)
        self.block4 = DepthwiseSeparableConv(128, 256, stride=2)

        # Channel attention — focuses on emotionally discriminative feature maps
        self.se = SEBlock(256, reduction=8)

        # Global average pool → compact descriptor
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Dropout for regularisation
        self.dropout = nn.Dropout(0.3)

        # Project to embedding space
        self.embed_proj = nn.Linear(256, embedding_dim)

        # Separate classification head (used during training, bypassed in CAG)
        self.classifier = nn.Linear(embedding_dim, num_emotions)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract normalised 512-D embedding.
        This is the vector compared against the KV cache during inference.
        """
        x = self.entry(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.se(x)              # attention re-weighting
        x = self.gap(x).flatten(1)  # (B, 256)
        x = self.dropout(x)
        emb = self.embed_proj(x)    # (B, 512)
        return F.normalize(emb, dim=1)  # L2-normalise for cosine similarity

    def forward(self, x: torch.Tensor):
        """Full forward pass (used during training)."""
        emb = self.extract_embedding(x)
        logits = self.classifier(emb)
        return logits, emb


# Quick sanity check
if __name__ == "__main__":
    model = EmotionCNN()
    dummy = torch.randn(4, 1, 48, 48)
    logits, emb = model(dummy)
    print(f"Logits: {logits.shape}, Embedding: {emb.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
