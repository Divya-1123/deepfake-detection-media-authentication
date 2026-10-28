# Deepfake Detection and Media Authentication Platform

B.Tech CSE final-year project. Upload an image or video; the platform finds the face, normalises it, classifies it with a fine-tuned **EfficientNet-B0**, and shows a verdict, a probability scale, forensic readings and (images) a Grad-CAM heat-map.

**Stack:** React + Vite · Python + FastAPI · OpenCV (DNN face detector, preprocessing, frame extraction) · PyTorch + timm.

## Model status and honest accuracy claims (read before your demo)
`models/deepfake_detector.pt` was trained **in this project** on the public **SAFL** image dataset (CelebA real faces vs StyleGAN-generated and FaceApp-edited faces). All numbers below were measured by `python -m ml.evaluate` on images the model never trained on (`results/metrics.json`, `results/training_log.txt`).

| Test group | Images | Accuracy | Fakes caught | Reals kept | ROC-AUC |
|---|---|---|---|---|---|
| Held-out test, all fake types | 615 | 91.7% | 88.0% | 95.2% | 0.959 |
| GAN-generated faces | 514 | 96.3% | 98.0% | 95.2% | 0.991 |
| FaceApp edits (paired with their originals) | 415 | 88.7% | 68.3% | 95.2% | 0.895 |
| **Celeb-DF face swaps, never seen in training** | 445 | n/a | **1.6%** | n/a | n/a |

* **It does not detect face-swap video** (Celeb-DF). The training data contains no face-swap examples with matching real frames, and the model flags only 7 of 445. Do not present it as a general deepfake detector. It is a detector for GAN-generated faces and app-edited faces.
* Test images come from the same dataset as the training images, so the first three rows are optimistic for real-world photos. The unseen-source row is the better guide to generalisation, and it is poor. This is the gap your synopsis describes.
* Real faces in SAFL are **CelebA** (non-commercial research use only) and the model was trained on them. Keep this project for academic use and cite SAFL (CC BY 4.0) and CelebA.
* An earlier model (v1) scored 98% on its test set but flagged only 1.6% of FaceApp edits: it had learned "CelebA photo vs GAN portrait", not manipulation. See `results/v1_metrics_shortcut_failure.json`. v2 fixes this by training on FaceApp edits, which have the same size and source as the real photos. Explaining this is a strong viva point.
* Without a checkpoint the server falls back to **DEMO MODE** (simple image statistics, not a validated detector) and the UI says so.

## UI and sign-in prototype
The frontend now opens with a responsive sign-in / create-account screen, then shows a workspace header with the signed-in demo profile and a log-out action. This is **prototype-only client-side session state**, not production authentication: it does not verify passwords, persist user accounts in a database, or protect API endpoints. Before public deployment, add backend-issued sessions/JWT, password hashing (Argon2id/bcrypt), a database, rate limiting, email verification/reset, CSRF protections where applicable, and authorization on every protected API route. Never store raw passwords in browser storage.

## Suggested next upgrades
1. **Production authentication + roles:** FastAPI auth endpoints, PostgreSQL/SQLite for development, hashed passwords, access control, and account recovery.
2. **Scan history:** save only user-authorized scan metadata and results; make retention/deletion explicit. Avoid keeping original uploads by default.
3. **Model evaluation page:** show test accuracy, precision, recall, F1, ROC-AUC, confusion matrix, dataset details, and the date/checkpoint hash for the currently loaded model.
4. **Generalise beyond GAN portraits:** add properly licensed, diverse face-swap and reenactment data (e.g. FaceForensics++ / Celeb-DF where permitted), split by identity/video, and report cross-dataset performance. Never tune on the final test set.
5. **Better video evidence:** temporal models, face tracking, per-frame timeline, downloadable PDF report, and comparison of frame-level scores.
6. **Deployment and safety:** Docker Compose, HTTPS, environment-based secrets, upload limits, antivirus/content checks, request throttling, logging without sensitive media, and a clear privacy/retention policy.

## Architecture
```text
React (Vite) --multipart--> FastAPI
   utils/files.py       extension + MIME + magic-byte + size checks, random temp name, deleted after each request
   utils/images.py      safe decode: EXIF rotation applied, >40 MP rejected (decompression-bomb guard)
   ml/face.py           OpenCV DNN face detector (Haar fallback), square crop with margin
   ml/preprocessing.py  crop -> harmonise (shrink to 96 px + JPEG q80, so size/compression carry no label info) -> 128 px tensor
   ml/model.py          EfficientNet-B0 (ImageNet-pretrained, fine-tuned) | TinyCNN; arch stored in the checkpoint
   ml/inference.py      trained model + threshold tuned on validation data | demo heuristic
   ml/explain.py        Grad-CAM heat-map (trained mode only)
   services/detector    image flow / video flow (<=24 evenly sampled frames, mean fake-probability)
```
Endpoints: `GET /api/health` (model state + measured evaluation) · `POST /api/detect/image` · `POST /api/detect/video` · `POST /api/analyze/metadata`. Swagger UI: http://127.0.0.1:8000/docs

## Setup and run (Windows PowerShell; Linux/macOS: `source venv/bin/activate`)
Prerequisites: Python 3.10+ (developed on 3.12), Node.js 18+, about 3 GB free disk for PyTorch. No API keys.
```powershell
# 1) Backend
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload          # http://127.0.0.1:8000  (first start can take 20-30 s while PyTorch loads)

# 2) Frontend (second terminal)
cd frontend
npm install
npm run dev                       # http://127.0.0.1:5173
```
Check `http://127.0.0.1:8000/api/health` shows `"model_mode": "TRAINED"`. Optional config: `copy .env.example backend\.env`; frontend API URL in `frontend/.env.example`.

## Tests (run from the folders shown)
```powershell
cd backend ; pip install scikit-learn matplotlib pytest httpx ; python -m pytest -q     # 44 tests
cd ..\frontend ; npm test ; npm run build                                               # 12 UI tests + build
```

## Retrain or reproduce (needs internet; GPU optional, e.g. Google Colab)
```powershell
cd backend
python -m ml.prepare_safl --download --out ..\data        # sparse-clones only images/ (about 1 GB), detects faces, pair-aware split
python -m ml.train --data ..\data --precropped --freeze-blocks 5 --epochs 6 --lr 6e-4 --dataset-name "SAFL v2"
python -m ml.evaluate --data ..\data --precropped --splits test test_stylegan test_faceapp unseen_celebdf
```
Training downloads the ImageNet EfficientNet-B0 weights once through timm (or pass `--pretrained-weights file.pth`). Omit `--freeze-blocks` on a GPU to fine-tune all layers. A rerun gives a near-identical (not byte-identical) split and slightly different numbers; the shipped numbers belong to the shipped checkpoint, and the server only displays evaluation results that match the loaded checkpoint.
For your own data use `data/{train,val,test}/{real,fake}/` split **by video or person, never by frame**, and `python -m ml.extract_frames` for videos. For better generalisation add real face-swap data (FaceForensics++, Celeb-DF) with matching real frames.

## Troubleshooting
| Symptom | Fix |
|---|---|
| UI says "Server offline" | Start `uvicorn app:app --reload` in `backend/`; port 8000 |
| Browser CORS error | Add your frontend origin to `ALLOWED_ORIGINS` in `backend/.env` |
| `ModuleNotFoundError: ml` | Run uvicorn/pytest from inside `backend/` |
| "DEMO MODE" badge | `models/deepfake_detector.pt` missing or `MODEL_PATH` wrong |
| "No face was found" | Detector finds roughly frontal faces up to moderate tilt; the whole image is used and flagged as less reliable |
| Image rejected as too large | Over 40 megapixels (`MAX_IMAGE_PIXELS`) |
| Video fails | Re-encode to H.264 MP4 |

## Other limitations
* Video = per-frame classification averaged over <=24 sampled frames, no temporal model. Transformers (mentioned in the synopsis) are not implemented.
* Only the largest face per image is analysed. Faces rotated by 90 degrees or heavily occluded are missed.
* Inputs are shrunk to 96 px before classification by design (to remove size/compression shortcuts), so very fine high-resolution artefacts are discarded.
* No database-backed accounts yet; sign-in is a front-end prototype only. Uploads are deleted after each request.
* Not tested on Windows or with a GPU. A GPU path exists in `ml/train.py` but was never run.

## Accuracy is not a promise
Do not claim 100% accuracy. The included evaluation documents strong results on some held-out SAFL subsets but very poor recall on unseen Celeb-DF face swaps. This is a dataset/generalisation limitation, not something a UI change can fix. Present the model as a research aid and show measured metrics and limitations to evaluators.
