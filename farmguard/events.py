"""Shared data types passed between the fast path and the GenAI agent."""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np


@dataclass
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    label: str                      # raw detector label, e.g. "wild boar"
    species: str | None = None      # mapped species key, e.g. "wild_boar"
    track_id: int | None = None

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def anchor(self, mode: str = "bottom_center") -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        if mode == "center":
            return ((x1 + x2) / 2, (y1 + y2) / 2)
        return ((x1 + x2) / 2, y2)  # bottom-center = where the animal stands


@dataclass
class FramePacket:
    """One processed frame. `t_read` etc. are time.perf_counter() values for latency (E1)."""
    index: int
    video_time: float               # seconds since start of the video / stream
    frame: np.ndarray
    detections: list[Detection] = field(default_factory=list)
    in_zone: list[Detection] = field(default_factory=list)  # target animals inside the field
    t_read: float = 0.0
    t_detect_done: float = 0.0

    @property
    def in_zone_track_ids(self) -> set[int]:
        return {d.track_id for d in self.in_zone if d.track_id is not None}


@dataclass
class IntrusionEvent:
    event_id: str
    species: str
    label: str
    track_id: int | None
    video_time: float
    frame_index: int
    bbox: tuple[float, float, float, float]
    confidence: float
    zone: str
    first_sound: str
    t_read: float                   # perf_counter when the triggering frame was read
    t_alarm_requested: float        # perf_counter when the alarm was queued
    wall_time: float = field(default_factory=time.time)

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:12]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RetreatStatus(str, Enum):
    GONE = "gone"
    LEAVING = "leaving"
    STILL_PRESENT = "still_present"
    UNCERTAIN = "uncertain"


@dataclass
class VLMReport:
    species: str                    # species key, "none" or "unclassifiable"
    count: int
    behaviour: str                  # eating | moving_through | resting | leaving | absent | unclear
    confidence: str                 # high | medium | low
    observation: str
    note: str
    raw: str = ""
    latency_s: float = 0.0
    error: str | None = None


@dataclass
class PolicyDecision:
    action: str                     # a sound id from the allowed list, or "escalate_farmer"
    reason: str
    confidence: float
    source: str                     # llm | fixed | random | rule_chain | bandit | fallback
    warnings: list[str] = field(default_factory=list)
    latency_s: float = 0.0


@dataclass
class Attempt:
    sound: str
    t_video: float
    status: RetreatStatus
    time_to_leave_s: float | None
    decision: PolicyDecision | None = None
    vlm: VLMReport | None = None
