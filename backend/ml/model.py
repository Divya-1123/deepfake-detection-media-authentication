"""PyTorch models and checkpoint loading.

Architectures (stored in the checkpoint's meta["arch"]):
  * "efficientnet_b0" - ImageNet-pretrained EfficientNet-B0 fine-tuned for real-vs-fake (default for new training).
  * "tinycnn"         - the original small CNN trained from scratch (kept for tests / very small experiments).

A checkpoint is either a raw state_dict (-> tinycnn) or {"state_dict": ..., "meta": {...}} as written by ml/train.py.
"""
from pathlib import Path
import torch
from torch import nn
from config import MODEL_PATH

DEFAULT_CHECKPOINT = MODEL_PATH
ARCHS = ("efficientnet_b0", "tinycnn")


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(arch: str = "efficientnet_b0", pretrained: bool = False, weights_file: Path | None = None) -> nn.Module:
    """Create a model. For efficientnet_b0 the ImageNet weights come from timm's hub (pretrained=True, needs internet)
    or from a local timm .pth file (weights_file) - handy on machines that cannot reach the hub."""
    if arch == "tinycnn":
        return TinyCNN()
    if arch != "efficientnet_b0":
        raise ValueError(f"Unknown architecture {arch!r}. Choose from {ARCHS}.")
    import timm
    if weights_file:
        model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1000)
        model.load_state_dict(torch.load(weights_file, map_location="cpu", weights_only=True))
        model.reset_classifier(2)
        model.drop_rate = 0.3
        return model
    return timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=2, drop_rate=0.3)


def cam_layer(model: nn.Module) -> nn.Module:
    """Layer whose activations Grad-CAM uses (stride-16 stage for EfficientNet, last conv for TinyCNN)."""
    if isinstance(model, TinyCNN):
        return [m for m in model.features if isinstance(m, nn.Conv2d)][-1]
    return model.blocks[4]


def load_model(checkpoint: Path = DEFAULT_CHECKPOINT):
    """Return (model | None, mode, meta). mode is "TRAINED" or "DEMO"."""
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        return None, "DEMO", {}
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state, meta = (payload["state_dict"], payload.get("meta", {})) if "state_dict" in payload else (payload, {})
    model = build_model(meta.get("arch", "tinycnn"))
    model.load_state_dict(state)
    model.eval()
    return model, "TRAINED", meta
