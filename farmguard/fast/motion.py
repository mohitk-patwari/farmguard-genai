"""Cheap motion gate (critter-cam / pietro-foini pattern): skip the detector on still frames.

Tracked animals are still re-checked every `keepalive` frames so a resting animal inside the
field is not forgotten by the tracker.
"""
from __future__ import annotations

import cv2
import numpy as np


class MotionGate:
    def __init__(self, enabled: bool = True, min_changed_fraction: float = 0.002,
                 warmup_frames: int = 15, keepalive: int = 10):
        self.enabled = enabled
        self.min_changed_fraction = min_changed_fraction
        self.warmup_frames = warmup_frames
        self.keepalive = keepalive
        self._bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25,
                                                      detectShadows=False)
        self._n = 0
        self._since_run = 0

    def should_run(self, frame: np.ndarray, tracks_active: bool) -> bool:
        self._n += 1
        if not self.enabled:
            return True
        small = cv2.resize(frame, (320, int(320 * frame.shape[0] / frame.shape[1])))
        mask = self._bg.apply(small)
        if self._n <= self.warmup_frames:
            return True
        changed = float(np.count_nonzero(mask)) / mask.size
        self._since_run += 1
        run = changed >= self.min_changed_fraction or (tracks_active and
                                                        self._since_run >= self.keepalive)
        if run:
            self._since_run = 0
        return run
