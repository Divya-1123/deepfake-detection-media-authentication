"""Fine-tune a detector on a folder dataset (see ml/prepare_faces.py, ml/prepare_safl.py).

    python -m ml.train --data ../data --precropped --epochs 8 --dataset-name "SAFL images"

Layout: data/{train,val,test}/{real,fake}/*.png  (split by VIDEO/IDENTITY, never by frame).
Default model: ImageNet-pretrained EfficientNet-B0 (needs internet once for the weights via timm, or pass
--pretrained-weights <timm .pth>). --freeze-blocks N freezes the stem + first N blocks: much faster on CPU.
The best epoch by validation ROC-AUC is saved with a decision threshold tuned on the validation split.
"""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from config import MODEL_PATH
from .dataset import FaceFolderDataset
from .model import build_model, ARCHS
from .metrics import compute_metrics, best_threshold


def predict_probs(model, loader):
    model.eval()
    device = next(model.parameters()).device
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps += torch.softmax(model(x.to(device)), 1)[:, 1].tolist()
            ys += y.tolist()
    return ys, ps


def freeze_prefix(model, n_blocks: int):
    """Freeze conv_stem, bn1 and blocks[:n_blocks] (EfficientNet). Returns the frozen modules."""
    if n_blocks <= 0 or not hasattr(model, "blocks"):
        return []
    frozen = [model.conv_stem, model.bn1, *list(model.blocks)[:n_blocks]]
    for m in frozen:
        for p in m.parameters():
            p.requires_grad = False
    return frozen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--arch", choices=ARCHS, default="efficientnet_b0")
    ap.add_argument("--pretrained-weights", type=Path, help="local timm efficientnet_b0 .pth (otherwise downloaded via timm)")
    ap.add_argument("--no-pretrained", action="store_true", help="random initialisation (not recommended)")
    ap.add_argument("--freeze-blocks", type=int, default=0)
    ap.add_argument("--precropped", action="store_true")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--out", type=Path, default=MODEL_PATH)
    ap.add_argument("--dataset-name", default="unspecified")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    train_ds = FaceFolderDataset(a.data, "train", augment=True, precropped=a.precropped)
    val_ds = FaceFolderDataset(a.data, "val", precropped=a.precropped)
    counts = train_ds.counts()
    print(f"train {len(train_ds)} {dict(zip(['real', 'fake'], counts))} | val {len(val_ds)} {dict(zip(['real', 'fake'], val_ds.counts()))}", flush=True)
    weights = torch.tensor([sum(counts) / (2 * max(c, 1)) for c in counts], dtype=torch.float32)
    train_dl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=a.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(a.arch, pretrained=not a.no_pretrained, weights_file=a.pretrained_weights).to(device)
    frozen = freeze_prefix(model, a.freeze_blocks)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"device {device} | arch {a.arch} | trainable params {sum(p.numel() for p in trainable):,} / {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(train_dl), pct_start=0.2)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.05)

    best_auc, best_state, best_epoch = -1.0, None, 0
    for epoch in range(1, a.epochs + 1):
        model.train()
        for m in frozen:
            m.eval()                      # keep BatchNorm statistics of frozen layers fixed
        t0, total = time.time(), 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(y)
        ys, ps = predict_probs(model, val_dl)
        m = compute_metrics(ys, ps)
        print(f"epoch {epoch:02d} loss {total/len(train_ds):.4f} val_auc {m['roc_auc']:.4f} val_acc@0.5 {m['accuracy']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if m["roc_auc"] >= best_auc:
            best_auc, best_epoch = m["roc_auc"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    ys, ps = predict_probs(model, val_dl)
    threshold = best_threshold(ys, ps)
    val_metrics = compute_metrics(ys, ps, threshold)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"arch": a.arch, "classes": ["real", "fake"], "epochs": a.epochs, "best_epoch": best_epoch, "freeze_blocks": a.freeze_blocks,
            "threshold": threshold, "val_metrics": val_metrics, "train_counts": counts, "dataset_name": a.dataset_name, "seed": a.seed,
            "pretrained": "imagenet" if not a.no_pretrained else "none"}
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "meta": meta}, a.out)
    print(f"saved best epoch {best_epoch} (val AUC {best_auc:.4f}, threshold {threshold:.3f}) -> {a.out}\nNext: python -m ml.evaluate --data {a.data} --precropped", flush=True)


if __name__ == "__main__":
    main()
