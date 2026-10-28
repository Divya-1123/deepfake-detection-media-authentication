from pathlib import Path
from utils.images import load_image
import cv2
import numpy as np


def _round(v):
    return round(float(v), 3)


def gray_signals(gray: np.ndarray) -> dict:
    """Forensic indicators for one grayscale image/frame. Shared by image and video analysis."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    edges = cv2.Canny(gray, 100, 200)
    h, w = gray.shape
    g16 = gray.astype(np.int16)
    # JPEG-like blockiness indicator: mean absolute differences at 8-pixel boundaries.
    vertical = np.abs(g16[:, 8:w:8] - g16[:, 7:w - 1:8]).mean() if w > 8 else 0
    horizontal = np.abs(g16[8:h:8, :] - g16[7:h - 1:8, :]).mean() if h > 8 else 0
    return {
        "blur": {"laplacian_variance": _round(lap.var()), "variance": _round(lap.var()), "interpretation": "lower values indicate stronger blur"},
        "noise": {"stddev": _round(lap.std())},
        "brightness_contrast": {"brightness": _round(gray.mean()), "contrast": _round(gray.std())},
        "edges": {"density": _round(np.count_nonzero(edges) / edges.size)},
        "compression": {"blockiness": _round((vertical + horizontal) / 2), "interpretation": "boundary differences at 8px intervals; heuristic only"},
    }


def image_forensics(path: Path) -> dict:
    size_bytes = path.stat().st_size
    image, exif = load_image(path)          # EXIF-rotated, size-capped, closes the file handle (safe on Windows)
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    return {
        "resolution": {"width": image.width, "height": image.height, "megapixels": _round(image.width * image.height / 1_000_000)},
        "file": {"size_bytes": size_bytes, "format": image.info.get("format") or path.suffix.lstrip(".").upper()},
        "exif": exif,
        **gray_signals(gray),
    }


def video_forensics(cap: cv2.VideoCapture, file_size: int, suffix: str) -> dict:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frames / fps if fps > 0 else 0
    codec_int = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    codec = "".join(chr((codec_int >> 8 * i) & 0xFF) for i in range(4)).strip("\x00") or "unknown"
    return {"resolution": {"width": width, "height": height}, "file": {"size_bytes": file_size, "format": suffix.lstrip(".").upper()}, "fps": _round(fps), "duration_seconds": _round(duration), "frame_count": frames, "codec": codec}
