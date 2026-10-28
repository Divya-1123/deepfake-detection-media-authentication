from fastapi import APIRouter, UploadFile, File, HTTPException
from starlette.concurrency import run_in_threadpool
from config import UPLOAD_DIR
from services.detector import analyze_image, analyze_video
from utils.files import save_upload, IMAGE_EXTS, VIDEO_EXTS, IMAGE_TYPES, VIDEO_TYPES, MAX_IMAGE_BYTES, MAX_VIDEO_BYTES

router = APIRouter()


@router.post("/image")
async def detect_image(file: UploadFile = File(...)):
    path = await save_upload(file, UPLOAD_DIR, IMAGE_EXTS, IMAGE_TYPES, MAX_IMAGE_BYTES)
    try:
        return {"media_type": "image", "filename": file.filename, **await run_in_threadpool(analyze_image, path)}
    except Exception as exc:
        raise HTTPException(422, f"Image analysis failed. The file may be corrupted or unsupported: {exc}") from exc
    finally:
        path.unlink(missing_ok=True)


@router.post("/video")
async def detect_video(file: UploadFile = File(...)):
    path = await save_upload(file, UPLOAD_DIR, VIDEO_EXTS, VIDEO_TYPES, MAX_VIDEO_BYTES)
    try:
        return {"media_type": "video", "filename": file.filename, **await run_in_threadpool(analyze_video, path)}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Video analysis failed: {exc}") from exc
    finally:
        path.unlink(missing_ok=True)
