import secrets
from pathlib import Path
from fastapi import UploadFile, HTTPException

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_VIDEO_BYTES = 100 * 1024 * 1024
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
VIDEO_TYPES = {"video/mp4", "video/x-msvideo", "video/quicktime", "video/x-matroska", "application/octet-stream"}


def _looks_like_image(head: bytes) -> bool:
    """Check magic bytes so a renamed text file is rejected early."""
    return (
        head.startswith(b"\xff\xd8\xff")                      # JPEG
        or head.startswith(b"\x89PNG\r\n\x1a\n")             # PNG
        or (head[:4] == b"RIFF" and head[8:12] == b"WEBP")    # WEBP
    )


def _looks_like_video(head: bytes) -> bool:
    return (
        head[4:8] == b"ftyp"                                  # MP4 / MOV
        or head.startswith(b"\x1a\x45\xdf\xa3")              # MKV / WebM (EBML)
        or (head[:4] == b"RIFF" and head[8:12] == b"AVI ")    # AVI
    )


async def save_upload(upload: UploadFile, dest: Path, allowed_exts: set[str], allowed_types: set[str], max_bytes: int) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in allowed_exts:
        raise HTTPException(400, f"Unsupported file extension. Allowed: {', '.join(sorted(allowed_exts))}")
    if upload.content_type not in allowed_types:
        raise HTTPException(400, "Unsupported MIME type.")
    safe_name = f"{secrets.token_hex(16)}{suffix}"
    target = dest / safe_name
    total = 0
    head = b""
    try:
        with target.open("wb") as f:
            while chunk := await upload.read(1024 * 1024):
                if total == 0:
                    head = chunk[:16]
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(413, f"File exceeds the {max_bytes // (1024*1024)} MB limit.")
                f.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    if total == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(400, "Uploaded file is empty.")
    check = _looks_like_image if allowed_exts == IMAGE_EXTS else _looks_like_video
    if not check(head):
        target.unlink(missing_ok=True)
        raise HTTPException(400, "File content does not match its extension (not a valid image/video file).")
    return target
