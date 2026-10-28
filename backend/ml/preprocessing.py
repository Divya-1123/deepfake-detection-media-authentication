"""Preprocessing shared by inference, training and evaluation.

ONE pipeline for all three guarantees the model sees identical inputs at train and test time:
  decode (EXIF-rotated) -> detect largest face -> square crop with margin (or whole frame if none)
  -> harmonise resolution -> normalise (ImageNet mean/std) -> NCHW tensor.

Why "harmonise"? Real-vs-fake training sets often differ in image size and compression (e.g. tiny
JPEG celebrity crops vs 1024px GAN renders). A network can then score well just by detecting
"small and blurry" - a shortcut that fails on real uploads. Every crop is therefore shrunk to the
same small size and re-encoded as JPEG at a fixed quality before it reaches the model, so size and
compression stop carrying label information.
"""
import cv2
import numpy as np
import torch
from utils.images import load_image
from .face import detect_faces, crop_face, DETECTOR_NAME

MODEL_SIDE = 128        # network input size
HARMONISE_SIDE = 96     # every crop is reduced to this size first ...
JPEG_QUALITY = 80       # ... and re-encoded at this quality
CAP_SIDE = 256          # huge crops are first capped to this (also what ml.prepare_faces stores)
IMAGE_SIZE = MODEL_SIDE  # backwards-compatible name
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def decode_bytes(data: bytes) -> np.ndarray:
    """Decode image bytes to a BGR uint8 array (raises ValueError if corrupt or too large)."""
    image, _ = load_image(data)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _resize(img, side):
    interp = cv2.INTER_AREA if max(img.shape[:2]) > side else cv2.INTER_CUBIC
    return cv2.resize(img, (side, side), interpolation=interp)


def harmonise(crop: np.ndarray, side: int = HARMONISE_SIDE, quality: int = JPEG_QUALITY, blur: float = 0.0) -> np.ndarray:
    """Resolution/compression normalisation. Deterministic unless blur/side/quality are randomised (training only)."""
    if max(crop.shape[:2]) > CAP_SIDE:
        crop = _resize(crop, CAP_SIDE)
    if blur > 0:
        crop = cv2.GaussianBlur(crop, (0, 0), blur)
    small = _resize(crop, side)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    small = cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else small
    return cv2.resize(small, (MODEL_SIDE, MODEL_SIDE), interpolation=cv2.INTER_CUBIC)


def bgr_to_tensor(bgr: np.ndarray) -> torch.Tensor:
    """Normalise an (already harmonised) BGR image into a 1x3xHxW float tensor."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (MODEL_SIDE, MODEL_SIDE):
        rgb = cv2.resize(rgb, (MODEL_SIDE, MODEL_SIDE), interpolation=cv2.INTER_AREA)
    arr = (rgb.astype(np.float32) / 255.0 - _MEAN) / _STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()


def prepare_input(bgr: np.ndarray):
    """Return (tensor, crop_bgr, face_info) for one BGR image/frame. crop_bgr is the unharmonised crop (for display)."""
    faces = detect_faces(bgr)
    if faces:
        crop = crop_face(bgr, faces[0])
        info = {"face_detected": True, "faces_found": len(faces), "face_box": list(faces[0]), "face_detector": DETECTOR_NAME}
    else:
        crop = bgr
        info = {"face_detected": False, "faces_found": 0, "face_box": None, "face_detector": DETECTOR_NAME}
    return bgr_to_tensor(harmonise(crop)), crop, info


# Backwards-compatible helpers used by the original prototype API.
def preprocess_image(data: bytes) -> torch.Tensor:
    return prepare_input(decode_bytes(data))[0]


def frame_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    return prepare_input(frame_bgr)[0]
