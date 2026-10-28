from fastapi import APIRouter, UploadFile, File, HTTPException
from starlette.concurrency import run_in_threadpool
from config import UPLOAD_DIR
from services.forensics import image_forensics
from utils.files import save_upload, IMAGE_EXTS, IMAGE_TYPES, MAX_IMAGE_BYTES

router = APIRouter()


@router.post("/metadata")
async def analyze_metadata(file: UploadFile = File(...)):
    path = await save_upload(file, UPLOAD_DIR, IMAGE_EXTS, IMAGE_TYPES, MAX_IMAGE_BYTES)
    try:
        return {"filename": file.filename, "forensic_analysis": await run_in_threadpool(image_forensics, path)}
    except Exception as exc:
        raise HTTPException(422, f"Metadata analysis failed: {exc}") from exc
    finally:
        path.unlink(missing_ok=True)
