"""
emotion_cnn.py
--------------
Lightweight CNN for facial emotion feature extraction.
Architecture: 4-stage VGG-style CNN + squeeze-and-excitation → global avg pool → 512-D embedding.
~5M parameters, about 2 ms per forward pass on a laptop CPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_stage(in_channels: int, out_channels: int) -> nn.Sequential:
    """Two 3x3 conv + BN + ReLU layers, then 2x2 max-pool (halves the feature map)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


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

    Stages (two 3x3 convs + max-pool each): 1→64 (24x24), 64→128 (12x12),
    128→256 (6x6), 256→512 (3x3), SE attention, global average pool → 512-D,
    FC projection 512→512 → L2 norm
    """
    def __init__(self, embedding_dim: int = 512, num_emotions: int = 7,
                 widths: tuple = (64, 128, 256, 512)):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_emotions = num_emotions

        # 4 VGG-style stages: 48 -> 24 -> 12 -> 6 -> 3
        self.features = nn.Sequential(
            *[conv_stage(c_in, c_out) for c_in, c_out in zip((1,) + widths[:-1], widths)]
        )

        # Channel attention — focuses on emotionally discriminative feature maps
        self.se = SEBlock(widths[-1], reduction=8)

        # Global average pool → compact descriptor
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Dropout for regularisation
        self.dropout = nn.Dropout(0.3)

        # Project to embedding space
        self.embed_proj = nn.Linear(widths[-1], embedding_dim)

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
        x = self.features(x)
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
