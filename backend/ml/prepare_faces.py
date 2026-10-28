"""Extract face crops from a folder of images (the face-detection step of the pipeline, done once for training).

    python -m ml.prepare_faces --src ../raw/real --out ../data/train/real

Each image -> largest detected face -> square crop with margin -> capped at 256 px -> saved as PNG.
Images with no detectable face are skipped (and counted) unless --keep-no-face is given.
Then train with --precropped so the detector is not run again.
"""
import argparse
from pathlib import Path
import cv2
from .dataset import EXTS
from .face import detect_faces, crop_face
from .preprocessing import CAP_SIDE


def crop_to_file(src: Path, dst: Path, keep_no_face: bool = False) -> bool:
    bgr = cv2.imread(str(src))
    if bgr is None:
        return False
    faces = detect_faces(bgr)
    if faces:
        crop = crop_face(bgr, faces[0])
    elif keep_no_face:
        crop = bgr
    else:
        return False
    if max(crop.shape[:2]) > CAP_SIDE:
        s = CAP_SIDE / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return bool(cv2.imwrite(str(dst), crop))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keep-no-face", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ok = skipped = 0
    for p in sorted(x for x in a.src.rglob("*") if x.suffix.lower() in EXTS):
        if crop_to_file(p, a.out / f"{p.stem}.png", a.keep_no_face):
            ok += 1
        else:
            skipped += 1
    print(f"saved {ok} face crops to {a.out}; skipped {skipped} (no face / unreadable)")


if __name__ == "__main__":
    main()
