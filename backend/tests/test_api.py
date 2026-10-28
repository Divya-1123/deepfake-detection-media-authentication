"""End-to-end API tests against the real FastAPI app (DEMO mode - no checkpoint is loaded)."""
from tests.conftest import uploads_left
import routes.detection as detection_routes


def post(client, path, name, data, ctype):
    return client.post(path, files={"file": (name, data, ctype)})


def test_health_reports_demo_mode(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["demo_mode"] is True and body["model_mode"] == "DEMO"


def test_image_with_face(client, face_jpg):
    r = post(client, "/api/detect/image", "face.jpg", face_jpg, "image/jpeg")
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["media_type"] == "image" and b["model_mode"] == "DEMO"
    assert b["prediction"] in {"REAL", "POTENTIALLY MANIPULATED"}
    assert 0 <= b["fake_probability"] <= 1 and 0.5 <= b["confidence"] <= 1
    assert b["face_detected"] is True and b["faces_found"] >= 1 and len(b["face_box"]) == 4
    assert b["face_detector"] == "dnn" and b["threshold"] == 0.5
    assert "heatmap_png_b64" not in b            # never produced in demo mode
    assert b["forensic_analysis"]["resolution"]["width"] == 512
    assert b["processing_time_ms"] > 0


def test_image_without_face_falls_back_to_whole_frame(client, noise_png):
    r = post(client, "/api/detect/image", "noise.png", noise_png, "image/png")
    assert r.status_code == 200
    assert r.json()["face_detected"] is False


def test_metadata_endpoint(client, face_jpg):
    r = post(client, "/api/analyze/metadata", "face.jpg", face_jpg, "image/jpeg")
    assert r.status_code == 200
    f = r.json()["forensic_analysis"]
    assert {"resolution", "blur", "noise", "edges", "compression"} <= f.keys()


def test_rejects_wrong_extension(client):
    assert post(client, "/api/detect/image", "notes.txt", b"hello", "text/plain").status_code == 400


def test_rejects_wrong_mime(client, face_jpg):
    assert post(client, "/api/detect/image", "face.jpg", face_jpg, "application/pdf").status_code == 400


def test_rejects_empty_file(client):
    assert post(client, "/api/detect/image", "empty.png", b"", "image/png").status_code == 400


def test_rejects_text_renamed_as_image(client):
    r = post(client, "/api/detect/image", "fake.png", b"this is not an image at all", "image/png")
    assert r.status_code == 400 and "does not match" in r.json()["detail"]


def test_corrupt_image_with_valid_header_returns_422(client):
    r = post(client, "/api/detect/image", "bad.png", b"\x89PNG\r\n\x1a\n" + b"garbage" * 50, "image/png")
    assert r.status_code == 422


def test_oversize_upload_returns_413(client, face_jpg, monkeypatch):
    monkeypatch.setattr(detection_routes, "MAX_IMAGE_BYTES", 1000)
    assert post(client, "/api/detect/image", "face.jpg", face_jpg, "image/jpeg").status_code == 413


def test_missing_file_field_returns_422(client):
    assert client.post("/api/detect/image").status_code == 422


def test_video_analysis(client, face_video):
    r = post(client, "/api/detect/video", "clip.mp4", face_video, "video/mp4")
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["media_type"] == "video" and b["frames_analyzed"] > 0
    assert b["frames_with_face"] > 0
    assert b["metadata"]["frame_count"] == 30 and b["metadata"]["resolution"]["width"] == 512
    assert len(b["frame_results"]) == b["frames_analyzed"]
    assert 0 <= b["fake_probability"] <= 1


def test_corrupt_video_returns_422(client):
    r = post(client, "/api/detect/video", "bad.mp4", b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 200, "video/mp4")
    assert r.status_code == 422


def test_text_renamed_as_video_rejected(client):
    assert post(client, "/api/detect/video", "x.mp4", b"not a video", "video/mp4").status_code == 400


def test_uploaded_files_are_always_deleted(client, face_jpg, noise_png):
    post(client, "/api/detect/image", "face.jpg", face_jpg, "image/jpeg")
    post(client, "/api/detect/image", "bad.png", b"\x89PNG\r\n\x1a\n" + b"x" * 40, "image/png")
    post(client, "/api/detect/image", "notes.txt", b"hello", "text/plain")
    assert uploads_left() == []


def test_cors_allows_only_configured_origin(client):
    ok = client.options("/api/detect/image", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    bad = client.options("/api/detect/image", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "access-control-allow-origin" not in bad.headers


# ---- regressions found during the final acceptance audit ----
def _phone_photo(face_jpg: bytes) -> bytes:
    """Photo stored sideways with EXIF Orientation=6, exactly how many phones save images."""
    import io
    from PIL import Image
    img = Image.open(io.BytesIO(face_jpg)).rotate(90, expand=True)
    exif = Image.Exif()
    exif[0x0112] = 6
    exif[0x010F] = "TestCam"          # Make
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    return buf.getvalue()


def test_exif_rotated_phone_photo_still_gets_face_detected(client, face_jpg):
    r = post(client, "/api/detect/image", "phone.jpg", _phone_photo(face_jpg), "image/jpeg")
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["face_detected"] is True
    assert b["forensic_analysis"]["exif"].get("Make") == "TestCam"      # EXIF still reported
    assert b["forensic_analysis"]["file"]["format"] == "JPEG"


def test_huge_pixel_count_is_rejected_before_decoding(client, monkeypatch):
    import io
    from PIL import Image
    import utils.images
    monkeypatch.setattr(utils.images, "MAX_IMAGE_PIXELS", 10_000)       # 100x100
    buf = io.BytesIO()
    Image.new("L", (400, 400), 128).save(buf, "PNG")
    r = post(client, "/api/detect/image", "big.png", buf.getvalue(), "image/png")
    assert r.status_code == 422 and "too large" in r.json()["detail"]
    assert uploads_left() == []
