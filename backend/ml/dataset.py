"""Folder dataset: <root>/<split>/{real,fake}/*.jpg|png|webp  (real -> 0, fake -> 1)."""
from pathlib import Path
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .preprocessing import prepare_input, bgr_to_tensor, harmonise

EXTS = {".jpg", ".jpeg", ".png", ".webp"}
CLASSES = ["real", "fake"]


class FaceFolderDataset(Dataset):
    """precropped=True means the images are already face crops (made by ml.prepare_faces), so no detection runs.

    Training augmentation (augment=True) randomises exactly the properties that could act as shortcuts:
    flip, brightness/contrast, pre-blur, harmonisation size (80-112 px) and JPEG quality (55-95).
    Evaluation uses the same deterministic harmonise() as inference.
    """

    def __init__(self, root: Path, split: str, augment: bool = False, precropped: bool = False):
        self.augment, self.precropped = augment, precropped
        self.items = [(p, label) for label, cls in enumerate(CLASSES) for p in sorted((Path(root) / split / cls).glob("*")) if p.suffix.lower() in EXTS]
        if not self.items:
            raise FileNotFoundError(f"No images found under {Path(root) / split}/{{real,fake}}")
        self._cache: dict[Path, np.ndarray] = {}

    def counts(self):
        return [sum(1 for _, y in self.items if y == c) for c in range(len(CLASSES))]

    def __len__(self):
        return len(self.items)

    def _crop(self, path: Path) -> np.ndarray:
        if self.precropped:
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Unreadable image: {path}")
            return img
        if path not in self._cache:           # Haar/DNN detection is the slow step, so cache the crop
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise ValueError(f"Unreadable image: {path}")
            self._cache[path] = prepare_input(bgr)[1]
        return self._cache[path]

    def __getitem__(self, i):
        path, label = self.items[i]
        crop = self._crop(path)
        if self.augment:
            if random.random() < 0.5:
                crop = cv2.flip(crop, 1)
            crop = np.clip(crop.astype(np.float32) * random.uniform(0.85, 1.15) + random.uniform(-12, 12), 0, 255).astype(np.uint8)
            img = harmonise(crop, side=random.randint(80, 112), quality=random.randint(55, 95), blur=random.uniform(0, 1.2) if random.random() < 0.5 else 0.0)
        else:
            img = harmonise(crop)
        return bgr_to_tensor(img)[0], torch.tensor(label)
