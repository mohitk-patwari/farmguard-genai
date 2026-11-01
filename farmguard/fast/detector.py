"""Detectors. All return list[Detection] with track ids when tracking is available.

UltralyticsDetector: YOLOv8 / YOLO11 / YOLO-World with ByteTrack.
  - athiru `detector/detect.py` 17: YOLO-World `set_classes([...])` gives Indian species with no
    training (open vocabulary).
  - gotogrub `src/core/detector.py` 73-84: `.track(persist=True, tracker="bytetrack.yaml")`.
MockDetector: replays boxes from a JSON annotation file, so the whole system (and the tests)
run without a GPU or model weights.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

from farmguard.events import Detection

log = logging.getLogger(__name__)


class Detector(Protocol):
    def detect(self, frame: np.ndarray, frame_index: int) -> list[Detection]: ...


class UltralyticsDetector:
    def __init__(self, model: str, classes: list[str] | None = None, open_vocabulary: bool = True,
                 confidence: float = 0.25, imgsz: int = 640, device: str = "",
                 tracker: str = "bytetrack.yaml",
                 species_mapper: Callable[[str], str | None] | None = None,
                 ignore_classes: list[str] | None = None):
        from ultralytics import YOLO  # lazy: heavy import

        self.model = YOLO(model)
        if open_vocabulary and classes and hasattr(self.model, "set_classes"):
            self.model.set_classes(classes)
            log.info("YOLO-World classes set: %s", classes)
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device or None
        self.tracker = tracker
        self.species_mapper = species_mapper or (lambda s: None)
        self.ignore = {c.lower() for c in (ignore_classes or [])}

    def detect(self, frame: np.ndarray, frame_index: int) -> list[Detection]:
        results = self.model.track(frame, persist=True, tracker=self.tracker, conf=self.confidence,
                                   imgsz=self.imgsz, device=self.device, verbose=False)
        out: list[Detection] = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            names = r.names
            ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
            for xyxy, conf, cls, tid in zip(boxes.xyxy.tolist(), boxes.conf.tolist(),
                                            boxes.cls.int().tolist(), ids):
                label = str(names.get(cls, cls) if isinstance(names, dict) else names[cls]).lower()
                if label in self.ignore:
                    continue
                out.append(Detection(xyxy=tuple(xyxy), confidence=float(conf), label=label,
                                     species=self.species_mapper(label), track_id=tid))
        return out


class MockDetector:
    """Replays annotations: {"frames": {"<index>": [{"xyxy": [...], "label": "wild boar",
    "conf": 0.9, "track_id": 1}, ...]}}. Frames not listed have no detections.
    Useful for tests, CI, the simulator demo and E3/E5 ground-truth replays."""

    def __init__(self, annotations: str | Path | dict,
                 species_mapper: Callable[[str], str | None] | None = None):
        data = annotations if isinstance(annotations, dict) else json.loads(
            Path(annotations).read_text(encoding="utf-8"))
        self.frames = {int(k): v for k, v in data.get("frames", {}).items()}
        self.species_mapper = species_mapper or (lambda s: None)

    def detect(self, frame: np.ndarray, frame_index: int) -> list[Detection]:
        return [Detection(xyxy=tuple(d["xyxy"]), confidence=float(d.get("conf", 0.9)),
                          label=d["label"], species=self.species_mapper(d["label"]),
                          track_id=d.get("track_id"))
                for d in self.frames.get(frame_index, [])]


def build_detector(cfg) -> Detector:
    backend = cfg.get("detector.backend", "ultralytics")
    if backend == "mock":
        return MockDetector(cfg.get("detector.mock_annotations"), cfg.species_for_label)
    return UltralyticsDetector(
        model=cfg.get("detector.model"), classes=cfg.get("detector.classes"),
        open_vocabulary=cfg.get("detector.open_vocabulary", True),
        confidence=cfg.get("detector.confidence", 0.25), imgsz=cfg.get("detector.imgsz", 640),
        device=cfg.get("detector.device", ""), tracker=cfg.get("detector.tracker", "bytetrack.yaml"),
        species_mapper=cfg.species_for_label, ignore_classes=cfg.get("detector.ignore_classes"))
