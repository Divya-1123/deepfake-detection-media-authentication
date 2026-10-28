# Project guide: verified results, demo, viva, report

## 1. What was actually run (Linux sandbox, Python 3.12, CPU only, 1 core)
| Check | Result |
|---|---|
| Backend `pytest` (44 tests: API, validation, EXIF, pixel cap, DNN face detection, harmonisation, EfficientNet + Grad-CAM, checkpoint/threshold/evaluation handling, prepare/train/evaluate scripts) | 44 passed |
| Frontend `vitest` (12 tests, API mocked) and `vite build` | 12 passed, build passed |
| Real Chromium driving the real Vite app and real backend (19 checks: badge, upload, face outline, heat-map toggle, no-face, bad file, video strip, model card, console errors, CORS, keyboard, 390 px mobile) | 19 of 19 passed |
| Server started from the default model path: `/api/health` = TRAINED, evaluation attached | passed |
| End-to-end on raw full-size originals from the held-out test split, through HTTP (45 images per group) | reals kept 42/45, GAN fakes caught 45/45, FaceApp edits caught 33/45 (consistent with the table in the README) |
| Training run (7 and 6 epoch CPU runs, logs in `results/`), evaluation on held-out and unseen splits | done; numbers in README |
| Mutation checks (disabling a safeguard makes its test fail) | done for magic-byte check, face detector, zero-frame guard, EXIF rotation, pixel cap |
| **Not run:** Windows, GPU training, Safari/Firefox, a full re-run of `prepare_safl` (about 25 min; only a dry run of the split logic), real face-swap video detection (the model fails it) | |

Bugs found by this testing and fixed: EXIF-rotated phone photos missed faces; a 137 KB image could use about 900 MB RAM; the UI claimed the cropped face was analysed in demo mode; fonts fell back to serif offline; a CSS class collision washed out the header (found only by looking at a real screenshot); a v1 model that scored 98% by learning a dataset shortcut.

## 2. Demo script (10 minutes)
1. Start backend and frontend. Badge: **Trained model active**. Open "About the model and how well it works" and read the table out loud, **including the Celeb-DF row**. Saying the limitation first earns trust.
2. Open `/docs` (Swagger) to show the API.
3. Open https://thispersondoesnotexist.com (the source of the GAN faces in training), save a face, upload it: expect **Looks manipulated**, the face outlined, a high probability, and a Grad-CAM view via the Heat-map button.
4. Upload a real photo of a person (your own, a CelebA-style portrait): expect **Looks authentic**. Photos very different from the training data (unusual lighting, full-body shots) can be misjudged; say so if it happens.
5. Upload `backend/tests/fixtures/astronaut_face.jpg` (public domain NASA photo) and a landscape (no face): the app says no face was found and the result is less reliable.
6. Upload a short MP4: frame-by-frame strip, "face found in N of M frames". Be clear the video path averages per-frame scores and cannot detect face swaps.
7. Upload `notes.txt` (rejected instantly), then stop the backend and reload (Server offline banner).
8. Show `results/confusion_matrix.png`, `results/metrics.json` and `results/v1_metrics_shortcut_failure.json`.

## 3. Viva: the story to tell
* **Pipeline:** decode (EXIF-rotated) -> DNN face detector -> square crop -> *harmonise* -> EfficientNet-B0 -> softmax -> threshold -> Grad-CAM.
* **Transfer learning:** ImageNet-pretrained EfficientNet-B0, early blocks frozen (faster, less overfitting), later blocks fine-tuned on face data.
* **Shortcut learning (your best answer):** in SAFL the real photos are small CelebA JPEGs and most fakes are 1024 px GAN images. A model can score 98% by detecting image size and style. Countermeasures: harmonisation (every crop shrunk and re-JPEG'd identically), augmentation of size/quality/blur, and FaceApp edits as fakes because they share size and source with the reals. v1 vs v2 shows the effect with real numbers.
* **Evaluation honesty:** split FaceApp edits with their originals (no pair straddles train/test), held out Celeb-DF entirely, report precision/recall/F1/AUC/confusion matrix per source, tuned the threshold on validation data only.
* **Likely questions:** Why DNN not Haar (found 60/60 vs 58/60 CelebA faces, handles tilt)? Why 96 px (removes size/compression shortcuts, costs fine detail)? Why does Celeb-DF fail (no face-swap training data, domain shift)? How would you fix it (FaceForensics++/Celeb-DF with matching real videos, split by video, temporal model, larger backbone, cross-dataset validation)? Why not 99%? (because that number measures the shortcut).

## 4. Figures for the report
Architecture diagram (README) · data-flow for image and video · `docs/screenshots/*.png` (real captures: GAN fake with heat-map, real result, model card, mobile) · `results/confusion_matrix.png` · per-source table · v1 vs v2 table · `results/training_log.txt` · test-run output. Cite SAFL (CC BY 4.0), CelebA, timm/EfficientNet, OpenCV.
