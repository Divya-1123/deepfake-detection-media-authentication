"""Smoke tests: prepare -> train -> evaluate run end to end on a tiny SYNTHETIC dataset.
They verify the TOOLING. The resulting numbers are meaningless and never shipped."""
import json
import sys
import cv2
import numpy as np
import pytest

pytest.importorskip("sklearn")
from ml import train, evaluate, prepare_faces
from ml.metrics import compute_metrics, best_threshold
from ml.model import load_model
from tests.conftest import FIXTURES


def test_compute_metrics_known_values():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])
    assert m["accuracy"] == 0.5 and m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5 and m["specificity"] == 0.5
    assert m["confusion_matrix"] == [[1, 1], [1, 1]] and m["roc_auc"] == 0.75


def test_single_class_split_metrics_do_not_crash():
    fake_only = compute_metrics([1, 1, 1, 1], [0.9, 0.8, 0.2, 0.7])
    assert fake_only["recall"] == 0.75 and fake_only["roc_auc"] is None and fake_only["specificity"] is None


def test_best_threshold_separates_clean_scores():
    t = best_threshold([0, 0, 0, 1, 1, 1], [0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert 0.3 < t <= 0.7


def test_prepare_faces_crops_a_face_and_skips_images_without_one(tmp_path):
    ok = prepare_faces.crop_to_file(FIXTURES / "astronaut_face.jpg", tmp_path / "a.png")
    crop = cv2.imread(str(tmp_path / "a.png"))
    assert ok and crop.shape[0] == crop.shape[1] and crop.shape[0] < 512
    noise = tmp_path / "n.png"
    cv2.imwrite(str(noise), np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8))
    assert prepare_faces.crop_to_file(noise, tmp_path / "n_out.png") is False
    assert prepare_faces.crop_to_file(noise, tmp_path / "n_out.png", keep_no_face=True) is True


def _make_dataset(root, unseen=True):
    rng = np.random.default_rng(0)
    plan = [("train", 12), ("val", 6), ("test", 6)]
    for split, n in plan:
        for cls in ("real", "fake"):
            d = root / split / cls
            d.mkdir(parents=True)
            for i in range(n):
                img = rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)
                if cls == "fake":
                    img = cv2.GaussianBlur(img, (9, 9), 3)       # trivially separable synthetic "fakes"
                cv2.imwrite(str(d / f"{i}.png"), img)
    if unseen:
        d = root / "unseen_x" / "fake"
        d.mkdir(parents=True)
        for i in range(4):
            cv2.imwrite(str(d / f"{i}.png"), cv2.GaussianBlur(rng.integers(0, 255, (96, 96, 3), dtype=np.uint8), (9, 9), 3))


@pytest.mark.parametrize("extra", [["--arch", "tinycnn"], ["--arch", "efficientnet_b0", "--no-pretrained", "--freeze-blocks", "5"]])
def test_train_and_evaluate_scripts_run(tmp_path, monkeypatch, extra):
    _make_dataset(tmp_path / "data")
    ckpt, out = tmp_path / "m.pt", tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["train", "--data", str(tmp_path / "data"), "--precropped", "--epochs", "1", "--batch-size", "8",
                                      "--out", str(ckpt), "--dataset-name", "synthetic-smoke-test", *extra])
    train.main()
    model, mode, meta = load_model(ckpt)
    assert mode == "TRAINED" and meta["dataset_name"] == "synthetic-smoke-test" and meta["train_counts"] == [12, 12]
    assert meta["arch"] == extra[1] and 0.0 <= meta["threshold"] <= 1.0 and "val_metrics" in meta
    monkeypatch.setattr(sys, "argv", ["evaluate", "--data", str(tmp_path / "data"), "--precropped", "--splits", "test", "unseen_x", "--model", str(ckpt), "--out", str(out)])
    evaluate.main()
    m = json.loads((out / "metrics.json").read_text())
    t = m["splits"]["test"]
    assert t["n_samples"] == 12 and sum(map(sum, t["confusion_matrix"])) == 12 and {"accuracy", "precision", "recall", "f1"} <= t.keys()
    assert m["splits"]["unseen_x"]["n_samples"] == 4 and m["checkpoint_meta"] == meta


def test_evaluate_without_checkpoint_exits_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate", "--data", str(tmp_path), "--model", str(tmp_path / "none.pt")])
    with pytest.raises(SystemExit):
        evaluate.main()
