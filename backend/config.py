"""Central configuration. Values can be overridden with environment variables (see .env.example)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE lines). Real environment variables take precedence."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(Path(__file__).resolve().parent / ".env")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", ROOT / "uploads"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "deepfake_detector.pt"))
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173").split(",")
    if o.strip()
]
MAX_VIDEO_FRAMES = int(os.getenv("MAX_VIDEO_FRAMES", "24"))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "40000000"))  # ~40 MP; a tiny PNG can otherwise expand to GBs of RAM
DECISION_THRESHOLD = 0.5

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
