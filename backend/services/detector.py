import time
from pathlib import Path
import cv2
import numpy as np
from config import MAX_VIDEO_FRAMES
from .forensics import image_forensics, video_forensics, gray_signals
from ml.inference import predict_bgr
from ml import inference
from ml.preprocessing import decode_bytes


def analyze_image(path: Path) -> dict:
    started = time.perf_counter()
    forensic = image_forensics(path)
    bgr = decode_bytes(path.read_bytes())
    prediction = predict_bgr(bgr, forensic, explain=True)
    return {**prediction, "processing_time_ms": round((time.perf_counter() - started) * 1000, 2), "forensic_analysis": forensic}


def _label(p: float):
    return ("POTENTIALLY MANIPULATED", p) if p >= inference.get_threshold() else ("REAL", 1 - p)


def analyze_video(path: Path, max_frames: int = MAX_VIDEO_FRAMES) -> dict:
    started = time.perf_counter()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("The uploaded video could not be decoded by OpenCV.")
    try:
        metadata = video_forensics(cap, path.stat().st_size, path.suffix)
        total, fps = metadata["frame_count"], metadata["fps"]
        if total <= 0:
            raise ValueError("The video contains no readable frames.")
        indices = np.unique(np.linspace(0, total - 1, min(max_frames, total), dtype=int))
        results, suspicious, top = [], [], (-1.0, None)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            signals = gray_signals(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            pred = predict_bgr(frame, signals)
            item = {"frame": int(idx), "timestamp_seconds": round(float(idx / fps), 3) if fps else 0, **pred}
            results.append(item)
            if pred["fake_probability"] >= inference.get_threshold():
                suspicious.append(item)
            if pred["fake_probability"] > top[0]:
                top = (pred["fake_probability"], frame)
        if not results:
            raise ValueError("No frames could be decoded from this video.")
        fake_probability = float(np.mean([r["fake_probability"] for r in results]))
        prediction, confidence = _label(fake_probability)
        out = {
            "prediction": prediction, "confidence": round(confidence, 4), "fake_probability": round(fake_probability, 4),
            "frames_analyzed": len(results), "threshold": inference.get_threshold(), "max_frame_probability": round(max(r["fake_probability"] for r in results), 4), "frames_with_face": sum(r["face_detected"] for r in results),
            "suspicious_frames": suspicious, "frame_results": results, "metadata": metadata,
            "model_mode": results[0]["model_mode"],
        }
        if inference._MODEL is not None and top[1] is not None:  # Grad-CAM for the most suspicious frame
            hm = predict_bgr(top[1], {}, explain=True).get("heatmap_png_b64")
            if hm:
                out["heatmap_png_b64"] = hm
        out["processing_time_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return out
    finally:
        cap.release()
