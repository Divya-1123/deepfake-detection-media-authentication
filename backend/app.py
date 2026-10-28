import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import ALLOWED_ORIGINS, ROOT
from routes.detection import router as detection_router
from routes.metadata import router as metadata_router
from ml import inference

app = FastAPI(title="Deepfake Detection and Media Authentication Platform", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _evaluation():
    """Measured results from `python -m ml.evaluate`, shown only if they belong to the CURRENT checkpoint."""
    path = ROOT / "results" / "metrics.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if data.get("checkpoint_meta") == inference.MODEL_META else None


@app.get("/api/health")
def health():
    """Reports the REAL model state so the UI can never claim a trained model when none is loaded."""
    mode = inference.MODEL_MODE
    return {"status": "ok", "service": "deepfake-detector", "model_mode": mode, "demo_mode": mode == "DEMO",
            "model_meta": inference.MODEL_META, "evaluation": _evaluation() if mode == "TRAINED" else None}


app.include_router(detection_router, prefix="/api/detect")
app.include_router(metadata_router, prefix="/api/analyze")
