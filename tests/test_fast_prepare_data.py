"""The dataset rules the mAP numbers rely on: no clip in two splits, no copies across splits."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "prepare_data", Path(__file__).resolve().parent.parent / "scripts" / "prepare_data.py")
pd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pd)


def test_split_is_stable_and_roughly_as_asked():
    splits = [pd.split_for("s", f"clip{i}") for i in range(5000)]
    assert splits == [pd.split_for("s", f"clip{i}") for i in range(5000)]
    assert 0.17 < splits.count("test") / 5000 < 0.23
    assert 0.08 < splits.count("val") / 5000 < 0.12


def test_roboflow_copies_and_video_frames_share_a_group():
    a = pd.roboflow_group("boar_12_jpg.rf.aaa111")
    b = pd.roboflow_group("boar_12_jpg.rf.bbb222")
    assert a == b
    assert pd.roboflow_group("farmcam-frame-0031_jpg.rf.x") == pd.roboflow_group("farmcam-frame-0932_jpg.rf.y")


def test_night_flag():
    rng = np.random.default_rng(0)
    ir = np.repeat(rng.integers(40, 220, (64, 64, 1), dtype=np.uint8), 3, axis=2)  # grey = infrared
    day = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    dark = rng.integers(0, 30, (64, 64, 3), dtype=np.uint8)
    assert pd.is_night(ir) and pd.is_night(dark) and not pd.is_night(day)


def test_box_conversion_and_clipping():
    assert pd.xyxy_to_xywhn(0, 0, 50, 100, 100, 100) == [.25, .5, .5, 1.0]
    assert pd.yolo_label_lines([[.5, .5, 1.2, -.1]], 2) == ["2 0.500000 0.500000 1.000000 0.000000"]


def test_build_keeps_a_group_together_and_skips_unlabelled(tmp_path, monkeypatch):
    import csv
    import cv2
    monkeypatch.setattr(pd, "RAW", tmp_path / "raw")
    monkeypatch.setattr(pd, "OUT", tmp_path / "yolo")
    src = tmp_path / "raw" / "s"
    src.mkdir(parents=True)
    rows = []
    for i in range(20):
        cv2.imwrite(str(src / f"{i}.jpg"), np.full((32, 32, 3), 120, np.uint8))
        rows.append({"image": f"{i}.jpg", "cls": "boar", "group": f"g{i % 4}",
                     "boxes": None if i == 0 else [[.5, .5, .2, .2]], "label_source": "human",
                     "licence": "x", "author": "", "url": ""})
    pd._write_manifest(src, rows)
    pd.build()
    with (tmp_path / "yolo" / "manifest.csv").open(encoding="utf-8") as fh:
        out = list(csv.DictReader(fh))
    assert len(out) == 19  # the box-less image is not turned into a background image
    by_group = {}
    for r in out:
        by_group.setdefault(r["group"], set()).add(r["split"])
    assert all(len(s) == 1 for s in by_group.values())
