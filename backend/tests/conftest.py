import os
import tempfile
from pathlib import Path

# Must be set BEFORE the app modules are imported: isolate uploads and force DEMO mode (no checkpoint).
_TMP = Path(tempfile.mkdtemp(prefix="deepfake_tests_"))
os.environ["UPLOAD_DIR"] = str(_TMP / "uploads")
os.environ["MODEL_PATH"] = str(_TMP / "no_such_model.pt")

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import app
import config

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


@pytest.fixture()
def face_jpg() -> bytes:
    return (FIXTURES / "astronaut_face.jpg").read_bytes()


@pytest.fixture()
def noise_png() -> bytes:
    rng = np.random.default_rng(0)
    ok, buf = cv2.imencode(".png", rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture()
def face_video(tmp_path) -> bytes:
    """A short real MP4 made from the face photo with slight motion, so frames differ."""
    img = cv2.imread(str(FIXTURES / "astronaut_face.jpg"))
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (512, 512))
    assert writer.isOpened(), "OpenCV could not open an mp4v writer in this environment"
    for i in range(30):
        M = np.float32([[1, 0, i % 5], [0, 1, 0]])
        writer.write(cv2.warpAffine(img, M, (512, 512)))
    writer.release()
    return path.read_bytes()


def uploads_left() -> list:
    return list(config.UPLOAD_DIR.glob("*"))
