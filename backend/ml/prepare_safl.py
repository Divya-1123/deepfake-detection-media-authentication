"""Build a leakage-aware training layout from the public SAFL dataset (https://github.com/oigres5/SAFL, CC BY 4.0).

    python -m ml.prepare_safl --download --out ../data          # clones ONLY images/ via sparse git (~1 GB)
    python -m ml.prepare_safl --src path/to/SAFL/images --out ../data

SAFL image groups (from the filename prefix) and how they are used:
  real  RC*    CelebA stills (178x218)                       -> real  (random 70/15/15 split)
  fake  TPNE*  StyleGAN faces (thispersondoesnotexist.com)   -> fake  (random 70/15/15 split)
  fake  D*, FA* mixed synthetic faces                        -> fake  (random 70/15/15 split)
  fake  FARC*  FaceApp edits of CelebA faces                 -> fake, placed in the SAME split as the CelebA original
               (FARC#### is an edit of RC####, so pairs never straddle train/test)
  fake  DFL*   Celeb-DF face-swap video frames               -> HELD OUT as  unseen_celebdf  (never trained on)
Also written: test_stylegan/ and test_faceapp/ (copies of test reals + that source's fakes) for a per-source breakdown.
Some images exist as both .jpg and .png with the same stem; only the first is kept (so duplicates cannot leak across splits).
  real  RCV*, RDFL* only ~10 video-frame reals               -> ignored (too few to balance DFL)
Real images are CelebA: non-commercial research use only. Do not redistribute the images.
"""
import argparse
import json
import random
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from .dataset import EXTS
from .prepare_faces import crop_to_file

REPO = "https://github.com/oigres5/SAFL"
POOL = {("real", "RC"): "real", ("fake", "TPNE"): "fake", ("fake", "D"): "fake", ("fake", "FA"): "fake"}
HELD_OUT = {("fake", "DFL"): "unseen_celebdf"}
PAIRED = ("fake", "FARC")


def download(dest: Path) -> Path:
    subprocess.run(["git", "clone", "--filter=blob:none", "--sparse", "--depth", "1", REPO, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", "images"], check=True)
    return dest / "images"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, help="existing SAFL/images folder")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="only print the planned split sizes")
    a = ap.parse_args()
    src = a.src or (download(Path(tempfile.mkdtemp(prefix="safl_")) / "SAFL") if a.download else None)
    if not src:
        raise SystemExit("Give --src <SAFL/images> or --download")

    pools, held, paired, seen = defaultdict(list), defaultdict(list), [], set()
    for cls in ("real", "fake"):
        for f in sorted((src / cls).iterdir()):
            if f.suffix.lower() not in EXTS or (cls, f.stem) in seen:
                continue                                   # skip .jpg/.png twins
            seen.add((cls, f.stem))
            prefix = re.match(r"[A-Za-z]+", f.name).group(0)
            if (cls, prefix) in POOL:
                pools[POOL[(cls, prefix)]].append(f)
            elif (cls, prefix) in HELD_OUT:
                held[HELD_OUT[(cls, prefix)]].append(f)
            elif (cls, prefix) == PAIRED:
                paired.append(f)
    rng, jobs, split_of_real = random.Random(a.seed), [], {}
    for cls, files in pools.items():
        rng.shuffle(files)
        n = len(files)
        cut = {"train": files[: int(.70 * n)], "val": files[int(.70 * n): int(.85 * n)], "test": files[int(.85 * n):]}
        for split, fs in cut.items():
            for f in fs:
                jobs.append((split, cls, f))
                if cls == "real":
                    split_of_real[f.stem] = split
    for f in paired:                                       # edit goes where its original went
        original = "RC" + re.search(r"\d+", f.stem).group(0)
        jobs.append((split_of_real.get(original) or rng.choices(["train", "val", "test"], [.7, .15, .15])[0], "fake", f))
    jobs += [(name, "fake", f) for name, fs in held.items() for f in fs]
    plan = Counter(f"{sp}/{c}" for sp, c, _ in jobs)
    if a.dry_run:
        print(json.dumps(dict(sorted(plan.items())), indent=2))
        return

    kept, dropped, outputs = Counter(), Counter(), {}
    for split, cls, f in jobs:
        d = a.out / split / cls
        d.mkdir(parents=True, exist_ok=True)
        if crop_to_file(f, d / f"{f.stem}.png"):
            kept[f"{split}/{cls}"] += 1
            outputs[(split, cls, f.stem)] = d / f"{f.stem}.png"
        else:
            dropped[f"{split}/{cls}"] += 1
    for name, prefixes in (("test_stylegan", ("TPNE", "D", "FA")), ("test_faceapp", ("FARC",))):   # per-source test subsets
        for (split, cls, stem), path in outputs.items():
            if split == "test" and (cls == "real" or re.match(r"[A-Za-z]+", stem).group(0) in prefixes):
                (a.out / name / cls).mkdir(parents=True, exist_ok=True)
                shutil.copy(path, a.out / name / cls / path.name)
    manifest = {"source": REPO, "license": "SAFL CC BY 4.0; real images from CelebA (non-commercial research)", "seed": a.seed,
                "kept": dict(sorted(kept.items())), "dropped_no_face": dict(sorted(dropped.items()))}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
