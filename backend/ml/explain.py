"""Grad-CAM explainability: which regions of the face crop pushed the model toward "fake".

Only meaningful for a TRAINED model. On an untrained network the map is noise, so inference.py
never produces one in DEMO mode. A heat-map shows what the model reacted to; it is not proof of tampering.
"""
from __future__ import annotations
import base64
import cv2
import numpy as np
import torch
from torch import nn
from .model import cam_layer


def gradcam(model: nn.Module, tensor: torch.Tensor, target_class: int = 1, layer: nn.Module | None = None) -> np.ndarray:
    """Heat-map in [0, 1] at input resolution."""
    layer = layer or cam_layer(model)
    store = {}

    def fwd(_m, _i, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))   # tensor hook: safe with in-place activations

    handle = layer.register_forward_hook(fwd)
    try:
        model.zero_grad()
        x = tensor.clone().requires_grad_(True)
        model(x)[0, target_class].backward()
        weights = store["grad"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * store["act"]).sum(dim=1, keepdim=True))
        cam = nn.functional.interpolate(cam, size=tensor.shape[2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().numpy()
    finally:
        handle.remove()
    cam = cam - cam.min()
    return cam / cam.max() if cam.max() > 0 else cam


def overlay_png_b64(crop_bgr: np.ndarray, cam: np.ndarray, size: int = 224) -> str:
    """Blend the heat-map over the crop and return a base64 PNG (no data-URI prefix)."""
    base = cv2.resize(crop_bgr, (size, size), interpolation=cv2.INTER_AREA)
    heat = cv2.applyColorMap((cv2.resize(cam, (size, size)) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    ok, buf = cv2.imencode(".png", cv2.addWeighted(base, 0.55, heat, 0.45, 0))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")
