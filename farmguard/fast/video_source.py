"""Video input: file, webcam index or RTSP URL.

Live mode (athiru `video/stream.py` pattern): a capture thread writes into Queue(maxsize=1) so the
detector always gets the newest frame and stale frames are dropped. File/eval mode processes
every sampled frame in order so results are reproducible.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Iterator

import cv2
import numpy as np

log = logging.getLogger(__name__)


class VideoSource:
    def __init__(self, source: str | int, live: bool = False, target_fps: float = 0,
                 resize_width: int = 0):
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self.source = source
        self.live = live
        self.target_fps = target_fps
        self.resize_width = resize_width
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source!r}")
        if live:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # stale frames queued in the driver hide latency
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.native_fps = fps if fps and fps > 1 else 25.0
        self._stop = threading.Event()
        self._eof = threading.Event()
        self._is_file = isinstance(source, str) and "://" not in source
        self.dropped = 0  # live mode: frames replaced before the detector saw them (E1 load report)
        self._q: queue.Queue = queue.Queue(maxsize=1)

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if self.resize_width and frame.shape[1] > self.resize_width:
            scale = self.resize_width / frame.shape[1]
            frame = cv2.resize(frame, (self.resize_width, int(frame.shape[0] * scale)))
        return frame

    # ---- file / evaluation mode ------------------------------------------------------
    def _iter_file(self) -> Iterator[tuple[int, float, np.ndarray, float]]:
        step = 1
        if self.target_fps and self.target_fps < self.native_fps:
            step = max(1, round(self.native_fps / self.target_fps))
        idx = 0
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            t_read = time.perf_counter()
            if not ok:
                break
            if idx % step == 0:
                yield idx, idx / self.native_fps, self._resize(frame), t_read
            idx += 1

    # ---- live mode -------------------------------------------------------------------
    def _capture_loop(self) -> None:
        idx, t0 = 0, time.perf_counter()
        while not self._stop.is_set():
            if self._is_file:  # a file in live mode behaves like a camera: frames arrive on time
                delay = idx / self.native_fps - (time.perf_counter() - t0)
                if delay > 0:
                    time.sleep(delay)
            ok, frame = self.cap.read()
            t_read = time.perf_counter()
            if not ok:
                if self._is_file:
                    self._eof.set()
                    return
                time.sleep(0.05)
                continue
            item = (idx, t_read - t0, self._resize(frame), t_read)
            try:
                self._q.put_nowait(item)
            except queue.Full:  # drop the stale frame, keep the newest
                try:
                    self._q.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass
                self._q.put_nowait(item)
            idx += 1

    def _iter_live(self) -> Iterator[tuple[int, float, np.ndarray, float]]:
        threading.Thread(target=self._capture_loop, daemon=True, name="capture").start()
        min_gap = 1.0 / self.target_fps if self.target_fps else 0.0
        last = 0.0
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                if self._eof.is_set():
                    return
                continue
            if min_gap and item[3] - last < min_gap:
                continue
            last = item[3]
            yield item

    def frames(self) -> Iterator[tuple[int, float, np.ndarray, float]]:
        """Yields (frame_index, video_time_s, frame_bgr, t_read_perf_counter)."""
        return self._iter_live() if self.live else self._iter_file()

    def release(self) -> None:
        self._stop.set()
        self.cap.release()
