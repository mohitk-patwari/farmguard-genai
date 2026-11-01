"""Field polygons (gotogrub `src/core/zone_manager.py` pattern, simplified to membership tests).

The bottom-center of the box (the animal's feet) is tested, not the box center, so an animal
standing at the edge of the field is not counted when only its head overlaps.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from farmguard.events import Detection


@dataclass
class Zone:
    name: str
    points: list[tuple[int, int]]
    target_species: list[str] = field(default_factory=list)  # empty = all species

    def __post_init__(self) -> None:
        self._contour = np.array(self.points, dtype=np.int32).reshape(-1, 1, 2)

    def contains(self, x: float, y: float) -> bool:
        return cv2.pointPolygonTest(self._contour, (float(x), float(y)), False) >= 0

    def applies_to(self, species: str | None) -> bool:
        return not self.target_species or (species in self.target_species)


class ZoneSet:
    def __init__(self, zones: list[Zone], frame_size: tuple[int, int] | None = None,
                 anchor: str = "bottom_center"):
        self.zones = zones
        self.frame_size = frame_size
        self.anchor = anchor

    @classmethod
    def load(cls, path: str | Path, anchor: str = "bottom_center") -> "ZoneSet":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        zones = [Zone(z["name"], [tuple(p) for p in z["points"]], z.get("target_species", []))
                 for z in data["zones"]]
        size = tuple(data["frame_size"]) if data.get("frame_size") else None
        return cls(zones, size, anchor)

    @classmethod
    def full_frame(cls, width: int, height: int) -> "ZoneSet":
        return cls([Zone("field", [(0, 0), (width, 0), (width, height), (0, height)])],
                   (width, height))

    def rescaled(self, width: int, height: int) -> "ZoneSet":
        """Zones are drawn on one resolution; rescale if frames arrive at another."""
        if not self.frame_size or tuple(self.frame_size) == (width, height):
            return self
        sx, sy = width / self.frame_size[0], height / self.frame_size[1]
        zones = [Zone(z.name, [(int(x * sx), int(y * sy)) for x, y in z.points], z.target_species)
                 for z in self.zones]
        return ZoneSet(zones, (width, height), self.anchor)

    def zone_of(self, det: Detection) -> str | None:
        x, y = det.anchor(self.anchor)
        for z in self.zones:
            if z.applies_to(det.species) and z.contains(x, y):
                return z.name
        return None

    def draw(self, frame: np.ndarray) -> np.ndarray:
        for z in self.zones:
            cv2.polylines(frame, [z._contour], True, (0, 200, 255), 2)
            cv2.putText(frame, z.name, z.points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        return frame
