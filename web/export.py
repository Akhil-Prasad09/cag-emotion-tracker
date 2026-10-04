"""
Export the trained CNN + prototype cache for the browser demo in web/.

    pip install onnx onnxruntime onnxscript
    python web/export.py [--check-test fer2013/test]

Writes web/model.onnx (face crop -> 512-D embedding) and web/prototypes.json,
then checks ONNX Runtime matches PyTorch. With --check-test it also scores the
exported model on the FER-2013 test set, using the same flip-averaged cache lookup as the app.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.modules.emotion_cnn import EmotionCNN  # noqa: E402
from src.modules.kv_cache import EmotionKVCache, EMOTION_LABELS  # noqa: E402

WEB = ROOT / "web"


class Embedder(torch.nn.Module):
    def __init__(self, cnn):
        super().__init__()
        self.cnn = cnn

    def forward(self, x):
        return self.cnn.extract_embedding(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check-test", type=str, default=None, help="FER-2013 test dir to score the export on")
    args = p.parse_args()

    cnn = EmotionCNN()
    cnn.load_state_dict(torch.load(ROOT / "models/emotion_cnn.pt", map_location="cpu", weights_only=True))
    model = Embedder(cnn).eval()

    dummy = torch.zeros(2, 1, 48, 48)
    torch.onnx.export(model, (dummy,), str(WEB / "model.onnx"), input_names=["face"],
                      output_names=["embedding"], dynamic_axes={"face": {0: "batch"}},
                      dynamo=False, opset_version=17)

    kv = EmotionKVCache()
    kv.load(str(ROOT / "cache/emotion_cache.pt"))
    keys = kv.key_matrix.cpu().numpy()
    (WEB / "prototypes.json").write_text(json.dumps({
        "labels": kv.labels,
        "keys": np.round(keys, 6).tolist(),
    }))

    sess = ort.InferenceSession(str(WEB / "model.onnx"))
    x = torch.randn(8, 1, 48, 48)
    with torch.no_grad():
        ref = model(x).numpy()
    out = sess.run(None, {"face": x.numpy()})[0]
    diff = float(np.abs(out - ref).max())
    assert diff < 1e-4, f"ONNX/PyTorch mismatch: {diff}"
    print(f"model.onnx: {(WEB / 'model.onnx').stat().st_size / 1e6:.1f} MB, max |ONNX - PyTorch| = {diff:.2e}")

    if args.check_test:
        from src.modules.trainer import EmotionDataset
        ds = EmotionDataset(args.check_test, augment=False)
        correct = 0
        for i in range(0, len(ds), 256):
            batch = [ds[j] for j in range(i, min(i + 256, len(ds)))]
            imgs = np.stack([b[0].numpy() for b in batch])
            labels = np.array([b[1] for b in batch])
            emb = sess.run(None, {"face": imgs})[0] + sess.run(None, {"face": imgs[..., ::-1].copy()})[0]
            correct += int(((emb @ keys.T).argmax(1) == labels).sum())
        print(f"exported model, FER-2013 test accuracy: {correct / len(ds):.4f} (n={len(ds)})")


if __name__ == "__main__":
    main()
