"""Save YOLO-World with the settings.yaml vocabulary already embedded, so runtime needs no CLIP.

set_classes() installs the CLIP package and downloads 338 MB of weights; that can't happen on a farm
with the network unplugged (invariants 1 and 4). Run once while online:
    python scripts/bake_world_vocab.py [--weights weights/yolov8s-worldv2.pt]
"""
from __future__ import annotations

import argparse

from ultralytics import YOLO

from farmguard.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="weights/yolov8s-worldv2.pt")
    args = ap.parse_args()
    cfg = load_config()
    model = YOLO(args.weights)
    model.set_classes(cfg.get("detector.classes"))
    out = cfg.path("detector.model")
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print(out)


if __name__ == "__main__":
    main()
