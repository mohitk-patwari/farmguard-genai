"""Detectors. All return list[Detection] with track ids when tracking is available.

UltralyticsDetector: YOLOv8 / YOLO11 / YOLO-World with ByteTrack.
  - athiru `detector/detect.py` 17: YOLO-World `set_classes([...])` gives Indian species with no
    training (open vocabulary).
  - gotogrub `src/core/detector.py` 73-84: `.track(persist=True, tracker="bytetrack.yaml")`.
MockDetector: replays boxes from a JSON annotation file, so the whole system (and the tests)
run without a GPU or model weights.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from farmguard.events import Detection

log = logging.getLogger(__name__)


class Detector(Protocol):
    def detect(self, frame: np.ndarray, frame_index: int) -> list[Detection]: ...


def results_to_detections(results, confidence: float = 0.0,
                          species_mapper: Callable[[str], str | None] | None = None,
                          ignore: set[str] | None = None) -> list[Detection]:
    """Ultralytics Results -> Detections. Pure, so it is tested without weights.

    boxes.id is None when ByteTrack returned no tracks (ultralytics 8.4 `trackers/track.py` 133-136
    keeps the raw detections in that case). Those stay track_id=None and vote under the per-species
    key in Trigger. The tracker gets low-score boxes for its second association stage, so a box
    below `confidence` is kept only if it continued an existing track.
    """
    mapper = species_mapper or (lambda s: None)
    ignore = ignore or set()
    out: list[Detection] = []
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        names = r.names
        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
        for xyxy, conf, cls, tid in zip(boxes.xyxy.tolist(), boxes.conf.tolist(),
                                        boxes.cls.int().tolist(), ids):
            if tid is None and conf < confidence:
                continue
            label = str(names.get(cls, cls) if isinstance(names, dict) else names[cls]).lower()
            if label in ignore:
                continue
            out.append(Detection(xyxy=tuple(xyxy), confidence=float(conf), label=label,
                                 species=mapper(label), track_id=tid))
    return out


def apply_clahe(frame: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
    """Contrast-limited equalisation of the lightness channel, for night frames. Returns a new
    image: the frame buffer (and so the VLM) keeps the original pixels."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _load_ultralytics_offline():
    """Import ultralytics with its network features off (invariant 1). Measured on 8.4.197: without
    this, plain .track() calls connect to www.google-analytics.com, and a missing `lap` triggers
    `pip install` from inside the frame loop. The env vars must be set before the first import."""
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    from ultralytics import YOLO
    from ultralytics.utils.events import events
    events.enabled = False  # in case something imported ultralytics before the env vars were set
    return YOLO


class UltralyticsDetector:
    def __init__(self, model: str | Path, classes: list[str] | None = None,
                 open_vocabulary: bool = False, confidence: float = 0.25, imgsz: int = 640,
                 device: str = "", tracker: str = "bytetrack.yaml",
                 species_mapper: Callable[[str], str | None] | None = None,
                 ignore_classes: list[str] | None = None, track_conf: float | None = None,
                 clahe: bool = False, clahe_clip_limit: float = 2.0, clahe_grid: int = 8):
        if not Path(model).exists():  # YOLO() would otherwise try to download it
            raise FileNotFoundError(f"Detector weights not found: {model}. See docs/HANDOFF.md.")
        if importlib.util.find_spec("lap") is None:  # fail at startup, not on the first animal
            raise RuntimeError("ByteTrack needs `lap`: pip install 'lap>=0.5.12'")
        YOLO = _load_ultralytics_offline()
        self.model = YOLO(str(model))
        if open_vocabulary and classes and hasattr(self.model, "set_classes"):
            # Needs the CLIP package and weights. Baked weights (scripts/bake_world_vocab.py) don't.
            log.warning("open_vocabulary=true needs CLIP on disk; bake the vocabulary for deployment")
            self.model.set_classes(classes)
            log.info("YOLO-World classes set: %s", classes)
        self.confidence = confidence
        self.track_conf = confidence if track_conf is None else track_conf
        self.imgsz = imgsz
        self.device = device or None
        self.tracker = tracker
        self.species_mapper = species_mapper or (lambda s: None)
        self.ignore = {c.lower() for c in (ignore_classes or [])}
        self.clahe = clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid = clahe_grid
        # Load onto the device now: the first inference takes seconds and must not land on an animal.
        self._track(np.zeros((imgsz, imgsz, 3), np.uint8))

    def _track(self, frame: np.ndarray):
        return self.model.track(frame, persist=True, tracker=self.tracker, conf=self.track_conf,
                                imgsz=self.imgsz, device=self.device, verbose=False)

    def detect(self, frame: np.ndarray, frame_index: int) -> list[Detection]:
        if self.clahe:
            frame = apply_clahe(frame, self.clahe_clip_limit, self.clahe_grid)
        return results_to_detections(self._track(frame), self.confidence, self.species_mapper,
                                     self.ignore)


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
        model=cfg.path("detector.model"), classes=cfg.get("detector.classes"),
        open_vocabulary=cfg.get("detector.open_vocabulary", False),
        confidence=cfg.get("detector.confidence", 0.25), imgsz=cfg.get("detector.imgsz", 640),
        device=cfg.get("detector.device", ""),
        # A file in the repo wins; otherwise the name of a tracker bundled with ultralytics.
        tracker=str(p) if (p := cfg.path("detector.tracker")).exists() else cfg.get("detector.tracker"),
        species_mapper=cfg.species_for_label, ignore_classes=cfg.get("detector.ignore_classes"),
        track_conf=cfg.get("detector.track_conf"), clahe=cfg.get("detector.clahe.enabled", False),
        clahe_clip_limit=cfg.get("detector.clahe.clip_limit", 2.0),
        clahe_grid=cfg.get("detector.clahe.grid", 8))
