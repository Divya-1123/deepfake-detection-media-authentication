"""Face detection and cropping (synopsis Step 2: face detection + facial-region extraction).

Primary detector: OpenCV DNN (SSD/ResNet-10, files in ml/assets) - finds small, slightly tilted and
off-axis faces better than Haar. Fallback: the Haar cascade bundled with opencv-python.
Neither finds heavily rotated (e.g. 90 degree) or strongly occluded faces. When no face is found the
pipeline falls back to the whole frame and reports face_detected = false.
"""
from __future__ import annotations
import threading
from pathlib import Path
import cv2
import numpy as np

ASSETS = Path(__file__).parent / "assets"
_PROTO, _WEIGHTS = ASSETS / "deploy.prototxt", ASSETS / "res10_300x300_ssd_iter_140000.caffemodel"
DNN_CONFIDENCE = 0.6
MAX_DETECT_SIDE = 960
_LOCK = threading.Lock()   # OpenCV detectors keep internal state and are not safe to share across threads

_DNN = cv2.dnn.readNetFromCaffe(str(_PROTO), str(_WEIGHTS)) if _PROTO.exists() and _WEIGHTS.exists() else None
_HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if _DNN is None and _HAAR.empty():  # pragma: no cover
    raise RuntimeError("No face detector available.")
DETECTOR_NAME = "dnn" if _DNN is not None else "haar"


def _clip(x, y, w, h, W, H):
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def _detect_dnn(bgr):
    H, W = bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    with _LOCK:
        _DNN.setInput(blob)
        out = _DNN.forward()[0, 0]
    boxes = []
    for d in out:
        if d[2] >= DNN_CONFIDENCE:
            x0, y0, x1, y1 = d[3] * W, d[4] * H, d[5] * W, d[6] * H
            b = _clip(int(x0), int(y0), int(x1 - x0), int(y1 - y0), W, H)
            if b[2] > 20 and b[3] > 20:
                boxes.append(b)
    return boxes


def _detect_haar(bgr):
    H, W = bgr.shape[:2]
    scale = min(1.0, MAX_DETECT_SIDE / max(H, W))
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bgr
    gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    min_side = max(30, int(min(gray.shape[:2]) * 0.08))
    with _LOCK:
        found = _HAAR.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side))
    return [tuple(int(v / scale) for v in b) for b in found]


def detect_faces(bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return face boxes (x, y, w, h) in original-image coordinates, largest first."""
    boxes = _detect_dnn(bgr) if _DNN is not None else []
    if not boxes:
        boxes = _detect_haar(bgr)
    return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)


def crop_face(bgr: np.ndarray, box: tuple[int, int, int, int], margin: float = 0.25) -> np.ndarray:
    """Square crop centred on the box with a relative margin; replicates edge pixels if it leaves the image."""
    x, y, w, h = box
    side = int(max(w, h) * (1 + 2 * margin))
    side = max(side, 8)
    cx, cy = x + w // 2, y + h // 2
    x0, y0 = cx - side // 2, cy - side // 2
    H, W = bgr.shape[:2]
    pad = max(0, -x0, -y0, x0 + side - W, y0 + side - H)
    if pad:
        bgr = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x0, y0 = x0 + pad, y0 + pad
    return bgr[y0:y0 + side, x0:x0 + side]
