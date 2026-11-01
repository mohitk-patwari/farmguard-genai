"""Per-stage timing (adamp87/pigeon `pipeline.py` 36-130 pattern) and alarm-latency log for E1.

Nobody in the 100 on-topic repos we reviewed logs frame-to-alarm latency, so this file is the
evidence for the paper's first claim: the GenAI layer does not delay the alarm.
"""
from __future__ import annotations

import csv
import statistics
import threading
from collections import defaultdict
from pathlib import Path


class StageTimer:
    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def add(self, stage: str, seconds: float) -> None:
        with self._lock:
            self.samples[stage].append(seconds * 1000.0)

    def summary(self) -> dict[str, dict[str, float]]:
        out = {}
        for stage, vals in self.samples.items():
            if not vals:
                continue
            s = sorted(vals)
            out[stage] = {"n": len(s), "mean_ms": statistics.fmean(s),
                          "median_ms": s[len(s) // 2],
                          "p95_ms": s[min(len(s) - 1, int(0.95 * len(s)))], "max_ms": s[-1]}
        return out


class LatencyLog:
    """One row per alarm: frame read -> detect done -> alarm requested -> audio started."""
    FIELDS = ["event_id", "sound", "reason", "agent_enabled", "t_read", "t_detect_done",
              "t_alarm_requested", "t_alarm_started", "read_to_request_ms", "read_to_start_ms"]

    def __init__(self, path: str | Path | None, agent_enabled: bool):
        self.path = Path(path) if path else None
        self.agent_enabled = agent_enabled
        self.rows: list[dict] = []
        self._lock = threading.Lock()

    def add(self, event_id: str | None, sound: str, reason: str, t_read: float,
            t_detect_done: float, t_requested: float, t_started: float) -> None:
        row = {"event_id": event_id, "sound": sound, "reason": reason,
               "agent_enabled": self.agent_enabled, "t_read": t_read,
               "t_detect_done": t_detect_done, "t_alarm_requested": t_requested,
               "t_alarm_started": t_started,
               "read_to_request_ms": (t_requested - t_read) * 1000.0,
               "read_to_start_ms": (t_started - t_read) * 1000.0 if t_started else None}
        with self._lock:
            self.rows.append(row)

    def save(self) -> None:
        if not self.path or not self.rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self.FIELDS)
            if new:
                w.writeheader()
            w.writerows(self.rows)
