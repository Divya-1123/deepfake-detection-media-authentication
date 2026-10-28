"""Inference orchestration.

Two modes, always reported to the caller as `model_mode`:
  * TRAINED - a checkpoint exists at MODEL_PATH; the CNN's softmax output is used.
  * DEMO    - no checkpoint; a transparent heuristic over forensic signals is used. It is NOT a
              validated deepfake detector and its output must not be presented as one.
"""
import numpy as np
import torch
from config import DECISION_THRESHOLD
from .model import load_model
from .preprocessing import prepare_input
from .explain import gradcam, overlay_png_b64

_MODEL, MODEL_MODE, MODEL_META = load_model()


def reload_model():
    """Reload the checkpoint from disk (used by tests and after training)."""
    global _MODEL, MODEL_MODE, MODEL_META
    _MODEL, MODEL_MODE, MODEL_META = load_model()


def _forensic_score(analysis: dict) -> float:
    # Conservative demo heuristic: indicators, not proof of manipulation.
    score = 0.05
    blur = analysis.get("blur", {}).get("variance", 0)
    noise = analysis.get("noise", {}).get("stddev", 0)
    compression = analysis.get("compression", {}).get("blockiness", 0)
    if blur < 35: score += 0.18
    elif blur < 80: score += 0.08
    if noise < 7: score += 0.12
    if compression > 12: score += 0.12
    if analysis.get("edges", {}).get("density", 0) < 0.025: score += 0.08
    return float(np.clip(score, 0.02, 0.90))


def get_threshold() -> float:
    """Decision threshold: tuned on the validation split at training time (stored in the checkpoint), else 0.5."""
    return float(MODEL_META.get("threshold", DECISION_THRESHOLD)) if _MODEL is not None else DECISION_THRESHOLD


def _label(fake_probability: float) -> tuple[str, float]:
    if fake_probability >= get_threshold():
        return "POTENTIALLY MANIPULATED", fake_probability
    return "REAL", 1.0 - fake_probability


def predict_bgr(bgr: np.ndarray, forensic_analysis: dict, explain: bool = False) -> dict:
    """Run the full pipeline (face crop -> model/heuristic -> label) on one BGR image or video frame."""
    tensor, crop, face_info = prepare_input(bgr)
    heatmap = None
    if _MODEL is not None:
        with torch.no_grad():
            fake_probability = float(torch.softmax(_MODEL(tensor), dim=1)[0, 1])
        if explain:
            heatmap = overlay_png_b64(crop, gradcam(_MODEL, tensor))
        mode = MODEL_MODE
    else:
        fake_probability = _forensic_score(forensic_analysis)
        mode = "DEMO"
    prediction, confidence = _label(fake_probability)
    out = {"prediction": prediction, "confidence": confidence, "fake_probability": fake_probability, "threshold": get_threshold(), "model_mode": mode, **face_info}
    if heatmap:
        out["heatmap_png_b64"] = heatmap
    return out
