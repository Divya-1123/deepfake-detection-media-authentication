"""Evaluate the trained checkpoint on held-out splits and write real metrics to disk.

    python -m ml.evaluate --data ../data --precropped --splits test unseen_faceapp unseen_celebdf
Outputs results/metrics.json and results/confusion_matrix.png (first split). A split containing only one class
(e.g. an unseen generator with no matching reals) reports recall (fake-only) or specificity (real-only).
"""
import argparse
import json
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from config import MODEL_PATH, ROOT
from .dataset import FaceFolderDataset
from .model import load_model
from .metrics import compute_metrics
from .train import predict_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--precropped", action="store_true")
    ap.add_argument("--model", type=Path, default=MODEL_PATH)
    ap.add_argument("--threshold", type=float, help="override the checkpoint's tuned threshold")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    a = ap.parse_args()

    model, mode, meta = load_model(a.model)
    if model is None:
        raise SystemExit(f"No checkpoint at {a.model}. Train first: python -m ml.train --data {a.data}")
    thr = a.threshold if a.threshold is not None else float(meta.get("threshold", 0.5))
    results = {"threshold": thr, "checkpoint_meta": meta, "splits": {}}
    first = None
    for split in a.splits:
        if not (a.data / split).exists():
            print(f"skip {split}: folder not found")
            continue
        ds = FaceFolderDataset(a.data, split, precropped=a.precropped)
        ys, ps = predict_probs(model, DataLoader(ds, batch_size=32))
        m = compute_metrics(ys, ps, thr)
        m["class_counts"] = dict(zip(["real", "fake"], ds.counts()))
        results["splits"][split] = m
        first = first or m
        print(split, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "metrics.json").write_text(json.dumps(results, indent=2))
    if first:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(first["confusion_matrix"], cmap="Blues")
            ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["real", "fake"], yticklabels=["real", "fake"], xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix ({a.splits[0]})")
            for (i, j), v in np.ndenumerate(first["confusion_matrix"]):
                ax.text(j, i, v, ha="center", va="center")
            fig.tight_layout()
            fig.savefig(a.out / "confusion_matrix.png", dpi=150)
        except ImportError:
            print("matplotlib not installed - skipped confusion_matrix.png")
    print(f"Saved to {a.out}")


if __name__ == "__main__":
    main()
