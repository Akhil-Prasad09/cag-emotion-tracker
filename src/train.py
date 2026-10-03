"""
train.py
--------
Training entry point.
Run:  python -m src.train --data data/  [--epochs 50] [--batch 64]

Supports two dataset modes:
  1. Folder-based (--data points to train/test dirs)
  2. FER-2013 CSV  (--fer_csv path/to/fer2013.csv)
"""

import argparse
import sys
from pathlib import Path
from torch.utils.data import DataLoader

from src.modules.trainer import EmotionTrainer, EmotionDataset, FERCSVDataset


def parse_args():
    p = argparse.ArgumentParser(description="Train EmotionCNN")
    p.add_argument("--data", type=str, default=None,
                   help="Root dir with train/ and test/ subdirs")
    p.add_argument("--fer_csv", type=str, default=None,
                   help="Path to fer2013.csv")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embedding_dim", type=int, default=512)
    p.add_argument("--model_out", type=str, default="models/emotion_cnn.pt")
    p.add_argument("--cache_out", type=str, default="cache/emotion_cache.pt")
    return p.parse_args()


def main():
    args = parse_args()

    if args.data is None and args.fer_csv is None:
        print("ERROR: Provide --data or --fer_csv")
        print("\nQuick demo: building synthetic cache only (no training data needed)")
        from src.modules.kv_cache import EmotionKVCache
        kv = EmotionKVCache(embedding_dim=args.embedding_dim)
        kv.build_from_synthetic()
        kv.save(args.cache_out)
        print("Synthetic cache saved. Use --data or --fer_csv for real training.")
        return

    trainer = EmotionTrainer(
        model_save_path=args.model_out,
        cache_path=args.cache_out,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch,
        embedding_dim=args.embedding_dim,
    )

    if args.fer_csv:
        train_ds = FERCSVDataset(args.fer_csv, split="Training", augment=True)
        val_ds   = FERCSVDataset(args.fer_csv, split="PublicTest", augment=False)
    else:
        train_ds = EmotionDataset(Path(args.data) / "train", augment=True)
        val_ds   = EmotionDataset(Path(args.data) / "test",  augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, num_workers=2, pin_memory=True)

    history = trainer.train(train_loader, val_loader)
    print("\nTraining complete. History (last 5 epochs):")
    for h in history[-5:]:
        print(f"  Epoch {h['epoch']}: loss={h['loss']:.4f} "
              f"train={h['train_acc']:.4f} val={h['val_acc']:.4f}")


if __name__ == "__main__":
    main()
