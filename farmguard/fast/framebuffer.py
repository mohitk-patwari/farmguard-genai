"""Ring buffer of recent processed frames shared by the fast path (writer) and the agent (reader).

The agent needs frames *around* the trigger (clip builder) and frames *after* each deterrent
(retreat check). It reads by video time so file playback and live streams behave the same.
"""
from __future__ import annotations

import bisect
import threading
import time
from collections import deque

from farmguard.events import FramePacket


class FrameBuffer:
    def __init__(self, max_seconds: float = 30.0):
        self.max_seconds = max_seconds
        self._frames: deque[FramePacket] = deque()
        self._cond = threading.Condition()
        self.finished = False  # set when the video source ends

    def push(self, pkt: FramePacket) -> None:
        with self._cond:
            self._frames.append(pkt)
            while self._frames and pkt.video_time - self._frames[0].video_time > self.max_seconds:
                self._frames.popleft()
            self._cond.notify_all()

    def finish(self) -> None:
        with self._cond:
            self.finished = True
            self._cond.notify_all()

    @property
    def latest_time(self) -> float:
        with self._cond:
            return self._frames[-1].video_time if self._frames else 0.0

    def window(self, t_start: float, t_end: float) -> list[FramePacket]:
        with self._cond:
            frames = list(self._frames)
        times = [f.video_time for f in frames]
        lo = bisect.bisect_left(times, t_start)
        hi = bisect.bisect_right(times, t_end)
        return frames[lo:hi]

    def around(self, t: float, before: int, after: int) -> list[FramePacket]:
        with self._cond:
            frames = list(self._frames)
        if not frames:
            return []
        times = [f.video_time for f in frames]
        i = min(range(len(times)), key=lambda k: abs(times[k] - t))
        return frames[max(0, i - before): i + after + 1]

    def wait_until(self, t: float, timeout_s: float = 120.0) -> bool:
        """Block until the buffer holds video at or after time t (or the source finished)."""
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while not self.finished and (not self._frames or self._frames[-1].video_time < t):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(timeout=min(remaining, 0.5))
            return bool(self._frames) and self._frames[-1].video_time >= t
