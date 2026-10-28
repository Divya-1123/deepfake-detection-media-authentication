"""Turn a folder of videos into frames for the training layout (e.g. FaceForensics++ videos).

    python -m ml.extract_frames --videos ../raw/real --out ../data/train/real --every 15 --max-per-video 20
Frames are named <video-stem>_<frame>.jpg. Put WHOLE videos into train, val or test - never split one video across splits.
"""
import argparse
from pathlib import Path
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--every", type=int, default=15)
    ap.add_argument("--max-per-video", type=int, default=20)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    saved = 0
    for v in sorted(p for p in a.videos.iterdir() if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}):
        cap, i, n = cv2.VideoCapture(str(v)), 0, 0
        while n < a.max_per_video:
            ok, frame = cap.read()
            if not ok:
                break
            if i % a.every == 0:
                cv2.imwrite(str(a.out / f"{v.stem}_{i:05d}.jpg"), frame)
                n += 1
            i += 1
        cap.release()
        saved += n
    print(f"saved {saved} frames to {a.out}")


if __name__ == "__main__":
    main()
