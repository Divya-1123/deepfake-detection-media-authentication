"""Unit tests for the ML plumbing: face detection, preprocessing, checkpoint loading, Grad-CAM, TRAINED-mode path.

NOTE: tests that use TinyCNN() with random weights verify the PLUMBING (shapes, modes, response fields).
They say nothing about detection accuracy - that needs a real dataset (see README).
"""
import base64
import cv2
import numpy as np
import pytest
import torch

from ml import inference
from ml.explain import gradcam, overlay_png_b64
from ml.face import detect_faces, crop_face
from ml.model import TinyCNN, load_model
from ml.preprocessing import prepare_input, decode_bytes, harmonise, IMAGE_SIZE
from services.forensics import gray_signals
from tests.conftest import FIXTURES


def test_face_detected_in_real_photo_and_not_in_noise():
    img = cv2.imread(str(FIXTURES / "astronaut_face.jpg"))
    boxes = detect_faces(img)
    assert len(boxes) >= 1
    assert detect_faces(np.random.default_rng(1).integers(0, 255, (300, 300, 3), dtype=np.uint8)) == []


def test_crop_face_is_square_even_at_the_image_edge():
    img = np.zeros((100, 100, 3), np.uint8)
    crop = crop_face(img, (80, 80, 40, 40))
    assert crop.shape[0] == crop.shape[1] > 0


def test_dnn_detector_is_active_and_handles_a_tilted_face():
    from ml import face
    assert face.DETECTOR_NAME == "dnn"
    img = cv2.imread(str(FIXTURES / "astronaut_face.jpg"))
    tilted = cv2.warpAffine(img, cv2.getRotationMatrix2D((256, 256), 25, 1), (512, 512))
    assert len(detect_faces(tilted)) >= 1


def test_harmonise_is_deterministic_and_fixed_size():
    img = cv2.imread(str(FIXTURES / "astronaut_face.jpg"))
    a, b = harmonise(img), harmonise(img)
    assert a.shape == (IMAGE_SIZE, IMAGE_SIZE, 3) and a.dtype == np.uint8 and (a == b).all()


def test_harmonise_narrows_the_sharpness_gap_between_big_and_small_copies():
    """The point of harmonisation: a 512px and a 140px copy of the same face must look alike to the model."""
    big = cv2.imread(str(FIXTURES / "astronaut_face.jpg"))
    small = cv2.resize(big, (140, 140), interpolation=cv2.INTER_AREA)
    lap = lambda im: cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    raw = lambda im: cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    before = abs(np.log(lap(raw(big)) / lap(raw(small))))
    after = abs(np.log(lap(harmonise(big)) / lap(harmonise(small))))
    assert after < before


def test_prepare_input_shape_and_face_info():
    tensor, crop, info = prepare_input(cv2.imread(str(FIXTURES / "astronaut_face.jpg")))
    assert tensor.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE) and tensor.dtype == torch.float32
    assert info["face_detected"] is True and info["face_detector"] == "dnn"
    assert crop.shape[0] < 512 and crop.shape[0] == crop.shape[1]  # it really cropped, square


def test_decode_bytes_rejects_garbage():
    with pytest.raises(ValueError):
        decode_bytes(b"not an image")


def test_gray_signals_blur_orders_correctly():
    rng = np.random.default_rng(2)
    sharp = rng.integers(0, 255, (128, 128), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (15, 15), 5)
    assert gray_signals(sharp)["blur"]["variance"] > gray_signals(blurred)["blur"]["variance"]


def test_demo_heuristic_is_bounded():
    for sig in ({}, {"blur": {"variance": 1}, "noise": {"stddev": 1}, "compression": {"blockiness": 99}, "edges": {"density": 0}}):
        assert 0.02 <= inference._forensic_score(sig) <= 0.90


def test_load_model_returns_demo_when_checkpoint_missing(tmp_path):
    model, mode, meta = load_model(tmp_path / "nope.pt")
    assert model is None and mode == "DEMO"


@pytest.mark.parametrize("wrap", [False, True])
def test_load_model_accepts_raw_and_wrapped_checkpoints(tmp_path, wrap):
    sd = TinyCNN().state_dict()
    path = tmp_path / "m.pt"
    torch.save({"state_dict": sd, "meta": {"dataset_name": "x"}} if wrap else sd, path)
    model, mode, meta = load_model(path)
    assert mode == "TRAINED" and model is not None and (meta.get("dataset_name") == "x") == wrap


def test_gradcam_shape_range_and_png():
    model = TinyCNN().eval()
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    cam = gradcam(model, x)
    assert cam.shape == (IMAGE_SIZE, IMAGE_SIZE) and 0 <= cam.min() and cam.max() <= 1
    png = base64.b64decode(overlay_png_b64(np.zeros((50, 60, 3), np.uint8), cam))
    assert png.startswith(b"\x89PNG")


def test_trained_mode_path_returns_heatmap_and_reports_mode(client, face_jpg, monkeypatch):
    """Plumbing check with random weights: proves TRAINED-mode wiring (mode flag, Grad-CAM, health)."""
    monkeypatch.setattr(inference, "_MODEL", TinyCNN().eval())
    monkeypatch.setattr(inference, "MODEL_MODE", "TRAINED")
    assert client.get("/api/health").json()["model_mode"] == "TRAINED"
    r = client.post("/api/detect/image", files={"file": ("f.jpg", face_jpg, "image/jpeg")})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["model_mode"] == "TRAINED" and base64.b64decode(b["heatmap_png_b64"]).startswith(b"\x89PNG")
    assert abs(b["fake_probability"] + (1 - b["fake_probability"]) - 1) < 1e-9


def test_trained_mode_video_includes_heatmap(client, face_video, monkeypatch):
    monkeypatch.setattr(inference, "_MODEL", TinyCNN().eval())
    monkeypatch.setattr(inference, "MODEL_MODE", "TRAINED")
    r = client.post("/api/detect/video", files={"file": ("c.mp4", face_video, "video/mp4")})
    assert r.status_code == 200 and r.json()["model_mode"] == "TRAINED" and "heatmap_png_b64" in r.json()


def test_video_with_no_decodable_frames_is_an_error_not_a_fake_REAL(tmp_path, monkeypatch):
    """Regression: the original prototype returned prediction=REAL (0 frames analysed) for such videos."""
    import cv2 as _cv2
    import services.detector as detector

    class UnreadableCapture:
        def __init__(self, *_): pass
        def isOpened(self): return True
        def get(self, prop): return {_cv2.CAP_PROP_FRAME_COUNT: 10, _cv2.CAP_PROP_FPS: 10}.get(prop, 0)
        def set(self, *_): return True
        def read(self): return False, None
        def release(self): pass

    monkeypatch.setattr(detector.cv2, "VideoCapture", UnreadableCapture)
    clip = tmp_path / "x.mp4"
    clip.write_bytes(b"0" * 10)
    with pytest.raises(ValueError, match="No frames"):
        detector.analyze_video(clip)


def test_efficientnet_checkpoint_roundtrip_gradcam_and_threshold(tmp_path, monkeypatch):
    """EfficientNet plumbing with random weights (no download): arch is read from the checkpoint, Grad-CAM works, threshold is honoured."""
    from ml.model import build_model
    net = build_model("efficientnet_b0")
    path = tmp_path / "e.pt"
    torch.save({"state_dict": net.state_dict(), "meta": {"arch": "efficientnet_b0", "threshold": 0.3, "dataset_name": "x"}}, path)
    model, mode, meta = load_model(path)
    assert mode == "TRAINED" and type(model).__name__ == "EfficientNet" and meta["threshold"] == 0.3
    cam = gradcam(model, torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert cam.shape == (IMAGE_SIZE, IMAGE_SIZE) and 0 <= cam.min() and cam.max() <= 1
    monkeypatch.setattr(inference, "_MODEL", model)
    monkeypatch.setattr(inference, "MODEL_META", meta)
    assert inference.get_threshold() == 0.3
    assert inference._label(0.35)[0] == "POTENTIALLY MANIPULATED" and inference._label(0.25)[0] == "REAL"


def test_unknown_architecture_is_rejected():
    from ml.model import build_model
    with pytest.raises(ValueError):
        build_model("resnet9000")


def test_health_shows_evaluation_only_for_the_matching_checkpoint(client, monkeypatch, tmp_path):
    import json
    import app as app_module
    results = tmp_path / "results"
    results.mkdir()
    meta = {"arch": "tinycnn", "dataset_name": "d", "seed": 1}
    (results / "metrics.json").write_text(json.dumps({"checkpoint_meta": meta, "splits": {"test": {"accuracy": 0.5}}}))
    monkeypatch.setattr(app_module, "ROOT", tmp_path)
    monkeypatch.setattr(inference, "_MODEL", TinyCNN().eval())
    monkeypatch.setattr(inference, "MODEL_MODE", "TRAINED")
    monkeypatch.setattr(inference, "MODEL_META", meta)
    assert client.get("/api/health").json()["evaluation"]["splits"]["test"]["accuracy"] == 0.5
    monkeypatch.setattr(inference, "MODEL_META", {**meta, "seed": 2})          # different checkpoint -> stale results are hidden
    assert client.get("/api/health").json()["evaluation"] is None
