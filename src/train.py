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
import json
import torch
from torch.utils.data import DataLoader, Subset

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
        test_ds  = FERCSVDataset(args.fer_csv, split="PrivateTest", augment=False)
        clean_ds = FERCSVDataset(args.fer_csv, split="Training", augment=False)
    else:
        # Hold out 10% of train/ for checkpoint selection; test/ is only scored once, at the end.
        root = Path(args.data)
        aug_ds, clean_ds = EmotionDataset(root / "train", augment=True), EmotionDataset(root / "train", augment=False)
        perm = torch.randperm(len(aug_ds), generator=torch.Generator().manual_seed(0)).tolist()
        n_val = len(perm) // 10
        train_ds, val_ds = Subset(aug_ds, perm[n_val:]), Subset(clean_ds, perm[:n_val])
        clean_ds = Subset(clean_ds, perm[n_val:])
        test_ds  = EmotionDataset(root / "test", augment=False)

    def loader(ds, shuffle=False):
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle,
                          num_workers=4, persistent_workers=True)

    history = trainer.train(loader(train_ds, shuffle=True), loader(val_ds))
    print("\nTraining complete. History (last 5 epochs):")
    for h in history[-5:]:
        print(f"  Epoch {h['epoch']}: loss={h['loss']:.4f} "
              f"train={h['train_acc']:.4f} val={h['val_acc']:.4f}")

    # Prototypes = per-class mean embedding of the best checkpoint over the clean train split
    trainer.update_cache(loader(clean_ds))
    report = evaluate(trainer, loader(test_ds), args.cache_out)
    report.update(epochs=args.epochs, best_val_acc=max(h["val_acc"] for h in history),
                  train_samples=len(train_ds), test_samples=len(test_ds))
    Path(args.model_out).with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def evaluate(trainer, test_loader, cache_path):
    """Test accuracy of the classifier head and of the CAG prototype lookup used at inference."""
    from src.modules.kv_cache import EmotionKVCache
    kv = EmotionKVCache(embedding_dim=trainer.embedding_dim)
    kv.load(cache_path)
    keys = kv.key_matrix.to(trainer.device)
    model = trainer.model.eval()
    head = cag = total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(trainer.device), labels.to(trainer.device)
            logits, emb = model(imgs)
            logits_f, emb_f = model(torch.flip(imgs, dims=[3]))   # test-time flip averaging
            logits, emb = logits + logits_f, emb + emb_f
            head += (logits.argmax(1) == labels).sum().item()
            cag  += ((emb @ keys.T).argmax(1) == labels).sum().item()
            total += len(labels)
    return {"test_acc_classifier_head": round(head / total, 4),
            "test_acc_cag_lookup": round(cag / total, 4)}


if __name__ == "__main__":
    main()
