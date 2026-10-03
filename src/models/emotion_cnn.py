# src/models/emotion_cnn.py
# ─────────────────────────────────────────────────────────────
# Lightweight CNN for real-time facial emotion feature extraction.
#
# Architecture overview:
#   Input  → [1 × 48 × 48] grayscale face patch
#   Conv blocks (4×) → progressively richer spatial features
#   Attention gate   → highlights discriminative facial regions
#   FC head          → 256-d feature vector  (used by CAG engine)
#   Classifier       → 7-class softmax       (used during training)
#
# Designed for <10 ms CPU inference per face crop.
# ─────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Squeeze-and-Excitation (channel attention) ────────────────
class SEBlock(nn.Module):
    """
    Recalibrates channel-wise feature responses adaptively.
    Helps the model focus on emotion-relevant channels (e.g.,
    channels encoding mouth curvature or brow raise).
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


# ── Depthwise-separable conv block ───────────────────────────
class DSConvBlock(nn.Module):
    """
    Depthwise-separable convolution: ~8× fewer multiply-adds than
    a standard conv, crucial for staying under 10 ms per frame.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride,
                            padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)   # swish – smoother gradients

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


# ── Spatial attention ─────────────────────────────────────────
class SpatialAttention(nn.Module):
    """
    Produces a 2-D attention map over the spatial grid.
    Forces the model to attend to face subregions (eyes/mouth)
    rather than background pixels leaking through the crop.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


# ── Main model ────────────────────────────────────────────────
class EmotionCNN(nn.Module):
    """
    Lightweight CNN that maps a 48×48 grayscale face crop to:
      • A 256-d L2-normalised feature vector  (used by CAG KV cache)
      • A 7-class emotion logit vector        (used during training)

    Parameter count: ~180 K  (vs InceptionV3's 24 M)
    """

    def __init__(self, num_classes: int = 7, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim

        # ── Stage 1: low-level edge / texture ─────────────────
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),          # 48 → 24
        )  # output: [B, 32, 24, 24]

        # ── Stage 2: mid-level shape ───────────────────────────
        self.stage2 = nn.Sequential(
            DSConvBlock(32, 64),
            SEBlock(64),
            DSConvBlock(64, 64),
            nn.MaxPool2d(2),          # 24 → 12
        )  # output: [B, 64, 12, 12]

        # ── Stage 3: high-level expression ────────────────────
        self.stage3 = nn.Sequential(
            DSConvBlock(64, 128),
            SEBlock(128),
            DSConvBlock(128, 128),
            nn.MaxPool2d(2),          # 12 → 6
        )  # output: [B, 128, 6, 6]

        # ── Stage 4: semantic compression ─────────────────────
        self.stage4 = nn.Sequential(
            DSConvBlock(128, 256),
            SEBlock(256),
        )  # output: [B, 256, 6, 6]

        # ── Spatial attention ──────────────────────────────────
        self.spatial_attn = SpatialAttention()

        # ── Global pooling → feature vector ───────────────────
        self.gap = nn.AdaptiveAvgPool2d(1)          # → [B, 256, 1, 1]

        # ── Feature projection (used for CAG embedding) ────────
        self.feature_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.BatchNorm1d(feature_dim),
        )

        # ── Classifier head (used during supervised training) ──
        self.dropout   = nn.Dropout(0.4)
        self.classifier = nn.Linear(feature_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the L2-normalised 256-d feature vector.
        This is the value injected into – and compared against – the KV cache.
        """
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.spatial_attn(x)
        x = self.gap(x)
        x = self.feature_proj(x)
        return F.normalize(x, p=2, dim=1)   # L2 norm → unit sphere

    def forward(self, x: torch.Tensor):
        """
        Full forward pass – returns (logits, features).
        During CAG inference only `extract_features` is called.
        """
        feats  = self.extract_features(x)
        logits = self.classifier(self.dropout(feats))
        return logits, feats


# ── Utility: count parameters ─────────────────────────────────
def count_params(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total:,}  Trainable: {train:,}"


if __name__ == "__main__":
    m = EmotionCNN()
    print(m)
    print(count_params(m))
    x = torch.randn(4, 1, 48, 48)
    logits, feats = m(x)
    print("Logits:", logits.shape)   # [4, 7]
    print("Feats :", feats.shape)    # [4, 256]
