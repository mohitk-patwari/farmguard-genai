"""Decides when an animal in the field becomes an intrusion event.

- Vote: an animal track must be inside the field in >= min_hits of the last `window` processed
  frames (subith2005 `app.py` 126-131 uses 4-of-7 majority; DMWeiy uses a 10-hit debounce).
- Per-track cooldown (TETRA040 `_is_on_cooldown`, gotogrub `zone:species` cooldown key):
  a track fires the first alarm once; later retries are the agent's job.
- Untracked detections fall back to a per-species key.
Pure logic, no I/O: easy to unit-test.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from farmguard.events import Detection


@dataclass
class TriggerHit:
    detection: Detection
    zone: str
    key: str
    first_stamp: float = 0.0  # stamp of the first in-zone vote that led to this hit


class Trigger:
    def __init__(self, window: int = 7, min_hits: int = 4, track_cooldown_s: float = 60.0,
                 min_box_area: float = 0.0):
        if min_hits > window:
            raise ValueError("min_hits must be <= window")
        self.window = window
        self.min_hits = min_hits
        self.cooldown = track_cooldown_s
        self.min_box_area = min_box_area
        self._votes: dict[str, deque[bool]] = defaultdict(lambda: deque(maxlen=window))
        self._last_fired: dict[str, float] = {}
        self._first_stamp: dict[str, float] = {}

    @staticmethod
    def key_for(det: Detection) -> str:
        return f"track:{det.track_id}" if det.track_id is not None else f"species:{det.species}"

    def update(self, detections_in_zone: list[tuple[Detection, str]],
               all_keys_seen: set[str], video_time: float,
               stamp: float | None = None) -> list[TriggerHit]:
        """detections_in_zone: target animals inside a zone this frame.
        all_keys_seen: keys of every target animal detected this frame (in or out of zone).
        stamp: clock value for this frame (t_read), echoed back as TriggerHit.first_stamp so E1 can
        report the voting delay as well as the delay after the vote."""
        stamp = video_time if stamp is None else stamp
        hits: list[TriggerHit] = []
        in_zone_keys = set()
        for det, zone in detections_in_zone:
            if det.area < self.min_box_area:
                continue
            key = self.key_for(det)
            in_zone_keys.add(key)
            self._votes[key].append(True)
            self._first_stamp.setdefault(key, stamp)
            votes = self._votes[key]
            if sum(votes) >= self.min_hits:
                last = self._last_fired.get(key)
                if last is None or video_time - last >= self.cooldown:
                    self._last_fired[key] = video_time
                    hits.append(TriggerHit(det, zone, key, self._first_stamp[key]))
        for key in list(self._votes):  # a frame without the animal in the zone is a "no" vote
            if key not in in_zone_keys:
                self._votes[key].append(False)
                if not any(self._votes[key]):
                    self._first_stamp.pop(key, None)
                    if key not in all_keys_seen:
                        del self._votes[key]
        return hits
