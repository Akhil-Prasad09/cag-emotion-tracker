"""
trainer.py
----------
Training pipeline for EmotionCNN on FER-2013 or any compatible dataset.
After training, updates the KV cache prototype vectors with real embeddings.

Dataset format expected:
  data/
    train/
      angry/   *.jpg|*.png
      happy/   ...
      ...
    test/
      ...

Or FER-2013 CSV format (kaggle dataset).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import time
from pathlib import Path
from typing import Optional, List, Tuple
from PIL import Image

from src.modules.emotion_cnn import EmotionCNN
from src.modules.kv_cache import EmotionKVCache, EMOTION_LABELS


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class EmotionDataset(Dataset):
    """
    Loads grayscale face images from a directory tree.
    Expects: root/{emotion_label}/{image_file}
    """
    LABEL_MAP = {e: i for i, e in enumerate(EMOTION_LABELS)}

    def __init__(self, root: str, augment: bool = True):
        self.samples: List[Tuple[str, int]] = []
        root = Path(root)

        for label_dir in sorted(root.iterdir()):
            if not label_dir.is_dir():
                continue
            label_str = label_dir.name.lower()
            if label_str not in self.LABEL_MAP:
                continue
            label_idx = self.LABEL_MAP[label_str]
            for img_path in label_dir.glob("*.*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((str(img_path), label_idx))

        if augment:
            self.transform = transforms.Compose([
                transforms.Grayscale(1),
                transforms.Resize((48, 48)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Grayscale(1),
                transforms.Resize((48, 48)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])

        print(f"[Dataset] Loaded {len(self.samples)} samples from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


class FERCSVDataset(Dataset):
    """
    Loads FER-2013 CSV dataset (from Kaggle).
    CSV columns: emotion, pixels, Usage
    """
    LABEL_MAP = {i: e for i, e in enumerate(EMOTION_LABELS)}

    def __init__(self, csv_path: str, split: str = "Training", augment: bool = True):
        import pandas as pd
        df = pd.read_csv(csv_path)
        df = df[df["Usage"] == split]

        self.pixels = df["pixels"].tolist()
        self.labels = df["emotion"].tolist()
        self.augment = augment

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            ])
        print(f"[FERDataset] {split}: {len(self.pixels)} samples")

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx):
        px = np.array(self.pixels[idx].split(), dtype=np.float32).reshape(48, 48)
        px = (px - px.mean()) / (px.std() + 1e-8)
        tensor = torch.from_numpy(px).unsqueeze(0)  # (1, 48, 48)
        if self.augment and torch.rand(1).item() > 0.5:
            tensor = torch.flip(tensor, dims=[2])   # horizontal flip
        return tensor, self.labels[idx]


# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------
class EmotionTrainer:
    def __init__(
        self,
        model_save_path: str = "models/emotion_cnn.pt",
        cache_path: str = "cache/emotion_cache.pt",
        device: str = "auto",
        embedding_dim: int = 512,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_save_path = Path(model_save_path)
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_path
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = EmotionCNN(
            embedding_dim=embedding_dim,
            num_emotions=len(EMOTION_LABELS)
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        print(f"[Trainer] Device: {self.device}")

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        best_acc = 0.0
        history = []

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss, correct, total = 0.0, 0, 0

            for imgs, labels in train_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                logits, _ = self.model(imgs)
                loss = self.criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item() * len(imgs)
                preds = logits.argmax(1)
                correct += (preds == labels).sum().item()
                total += len(imgs)

            self.scheduler.step()
            train_acc = correct / total
            avg_loss = total_loss / total

            val_acc = 0.0
            if val_loader is not None:
                val_acc = self._evaluate(val_loader)
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(self.model.state_dict(), self.model_save_path)
                    print(f"  ✓ Saved best model (val_acc={val_acc:.4f})")

            history.append({"epoch": epoch, "loss": avg_loss,
                             "train_acc": train_acc, "val_acc": val_acc})

            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}/{self.epochs} | "
                      f"loss={avg_loss:.4f} train_acc={train_acc:.4f} "
                      f"val_acc={val_acc:.4f}")

        print(f"[Trainer] Done. Best val_acc={best_acc:.4f}")

        # Update KV cache with real learned embeddings
        self._update_cache(train_loader)
        return history

    def _evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                logits, _ = self.model(imgs)
                correct += (logits.argmax(1) == labels).sum().item()
                total += len(imgs)
        return correct / total

    def _update_cache(self, loader: DataLoader, n_batches: int = 20) -> None:
        """
        After training, compute per-class mean embeddings and update the KV cache.
        This makes the cache reflect REAL learned representations, not synthetic ones.
        """
        print("[Trainer] Updating KV cache with learned embeddings...")
        self.model.eval()
        kv = EmotionKVCache(embedding_dim=self.embedding_dim)
        kv.build_from_synthetic()  # initialise structure

        class_sums = torch.zeros(len(EMOTION_LABELS), self.embedding_dim)
        class_counts = torch.zeros(len(EMOTION_LABELS))

        with torch.no_grad():
            for i, (imgs, labels) in enumerate(loader):
                if i >= n_batches:
                    break
                imgs = imgs.to(self.device)
                embs = self.model.extract_embedding(imgs).cpu()
                for cls in range(len(EMOTION_LABELS)):
                    mask = (torch.tensor(labels) == cls)
                    if mask.any():
                        class_sums[cls] += embs[mask].sum(0)
                        class_counts[cls] += mask.sum()

        for cls in range(len(EMOTION_LABELS)):
            if class_counts[cls] > 0:
                mean_emb = class_sums[cls] / class_counts[cls]
                mean_emb = torch.nn.functional.normalize(mean_emb.unsqueeze(0), dim=1).squeeze(0)
                kv.key_matrix[cls] = mean_emb
                kv.entries[EMOTION_LABELS[cls]].key = mean_emb

        kv.save(self.cache_path)
        print(f"[Trainer] Cache updated and saved to {self.cache_path}")
