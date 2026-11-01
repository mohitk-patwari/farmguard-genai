"""Fine-tune yolov8n on data/yolo and report mAP50 per class, split by day/night and label source.

    python scripts/train_detector.py train --epochs 60
    python scripts/train_detector.py eval  [--world weights/yolov8s-worldv2.pt]

eval writes results/detector_eval.json. Each subset is evaluated with and without CLAHE on the same
test images. Machine-labelled subsets are reported under their own names: a model scored against
labels from another model is agreement, not accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from farmguard.fast.detector import apply_clahe  # noqa: E402

DATA = ROOT / "data" / "yolo"
WEIGHTS = ROOT / "weights" / "farmguard_yolov8n.pt"
CLASSES = ["boar", "monkey", "nilgai", "cattle"]
# Prompts in the dataset's class order, so YOLO-World can be scored on the same labels.
WORLD_PROMPTS = ["wild boar", "monkey", "nilgai", "cow"]


def train(args) -> None:
    from ultralytics import YOLO
    model = YOLO(str(ROOT / "weights" / "yolov8n.pt"))
    res = model.train(data=str(DATA / "data.yaml"), epochs=args.epochs, imgsz=640, batch=args.batch,
                      patience=15, device=args.device, workers=4, seed=0, deterministic=True,
                      project=str(ROOT / "runs"), name="farmguard", exist_ok=True, plots=True)
    shutil.copy2(Path(res.save_dir) / "weights" / "best.pt", WEIGHTS)
    print(WEIGHTS)


def subsets() -> dict[str, list[str]]:
    """Test-split images by label source and day/night. Hard negatives go with the day human set."""
    out: dict[str, list[str]] = {}
    with (DATA / "manifest.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] != "test":
                continue
            when = "night" if r["night"] == "True" else "day"
            src = "human" if r["label_source"] == "none" else r["label_source"]
            out.setdefault(f"{when}/{src}", []).append(r["file"])
    return out


def _yaml_for(name: str, files: list[str], root: Path) -> Path:
    work = DATA / "subsets"
    work.mkdir(exist_ok=True)
    slug = name.replace("/", "_").replace("+", "_")
    listing = work / f"{slug}.txt"
    listing.write_text("\n".join(str(root / f) for f in files) + "\n", encoding="utf-8")
    y = work / f"{slug}.yaml"
    y.write_text(f"path: {root.as_posix()}\ntrain: {listing.as_posix()}\nval: {listing.as_posix()}\n"
                 f"names: {json.dumps(dict(enumerate(CLASSES)))}\n", encoding="utf-8")
    return y


def _clahe_copy(name: str, files: list[str]) -> Path:
    """CLAHE versions of a subset, as PNG so no second JPEG pass differs from the baseline arm."""
    root = DATA / "clahe" / name.replace("/", "_").replace("+", "_")
    for f in files:
        dst = (root / f).with_suffix(".png")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            cv2.imwrite(str(dst), apply_clahe(cv2.imread(str(DATA / f))))
        lab = (root / f.replace("images/", "labels/", 1)).with_suffix(".txt")
        lab.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2((DATA / f.replace("images/", "labels/", 1)).with_suffix(".txt"), lab)
    return root


def _score(model, yaml_path: Path, device: str, names: list[str]) -> dict:
    m = model.val(data=str(yaml_path), imgsz=640, batch=16, device=device, verbose=False, plots=False,
                  project=str(ROOT / "runs"), name="val", exist_ok=True, workers=2)
    per = {names[int(c)]: round(float(ap), 3) for c, ap in zip(m.box.ap_class_index, m.box.ap50)}
    inst = {names[i]: int(n) for i, n in enumerate(m.nt_per_class) if n} if hasattr(m, "nt_per_class") else {}
    return {"map50": round(float(np.mean(list(per.values()))), 3) if per else None,
            "ap50": per, "instances": inst}


def _false_positive_rate(weights: str, files: list[str], conf: float, device: str, clahe: bool) -> dict:
    """Hard negatives (people, dogs, goats, cars...): how often would a box appear at the alarm threshold?"""
    from ultralytics import YOLO
    model = YOLO(weights)
    hits = 0
    for f in files:
        img = cv2.imread(str(DATA / f))
        if clahe:
            img = apply_clahe(img)
        hits += len(model.predict(img, conf=conf, imgsz=640, device=device, verbose=False)[0].boxes) > 0
    return {"images": len(files), "with_detection": hits, "conf": conf}


def evaluate(args) -> None:
    from ultralytics import YOLO
    sets = subsets()
    report: dict = {"weights": str(WEIGHTS.name), "subsets": {}}
    for name, files in sorted(sets.items()):
        entry: dict = {"images": len(files), "machine_labels": "yolo_world" in name}
        entry["plain"] = _score(YOLO(str(WEIGHTS)), _yaml_for(name, files, DATA), args.device, CLASSES)
        cl_root = _clahe_copy(name, files)
        cl_files = [str(Path(f).with_suffix(".png").as_posix()) for f in files]
        entry["clahe"] = _score(YOLO(str(WEIGHTS)), _yaml_for(name + "_clahe", cl_files, cl_root),
                                args.device, CLASSES)
        if args.world and "yolo_world" not in name:  # scoring YOLO-World on its own labels is circular
            world = YOLO(args.world)
            world.set_classes(WORLD_PROMPTS)
            entry["yolo_world_baseline"] = _score(world, _yaml_for(name, files, DATA), args.device, CLASSES)
        report["subsets"][name] = entry
        print(name, json.dumps(entry))
    with (DATA / "manifest.csv").open(encoding="utf-8") as fh:
        negatives = [r["file"] for r in csv.DictReader(fh) if r["split"] == "test" and r["cls"] == "negative"]
    if negatives:
        report["hard_negatives"] = {
            "plain": _false_positive_rate(str(WEIGHTS), negatives, 0.25, args.device, False),
            "clahe": _false_positive_rate(str(WEIGHTS), negatives, 0.25, args.device, True)}
    frame = np.random.default_rng(0).integers(0, 255, (540, 960, 3), dtype=np.uint8)
    t = time.perf_counter()
    for _ in range(100):
        apply_clahe(frame)
    report["clahe_ms_per_960x540_frame"] = round((time.perf_counter() - t) * 10, 2)
    out = ROOT / "results" / "detector_eval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--device", default="0")
    t.set_defaults(fn=train)
    e = sub.add_parser("eval")
    e.add_argument("--device", default="0")
    e.add_argument("--world", default=None)
    e.set_defaults(fn=evaluate)
    a = ap.parse_args()
    a.fn(a)
