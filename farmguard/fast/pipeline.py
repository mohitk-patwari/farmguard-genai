"""Layer 1: the fast path. Runs on every sampled frame and NEVER calls a network or an LLM.

frame -> motion gate -> detector + tracker -> field zone -> 4-of-7 vote -> alarm (own thread)
                                                                      -> IntrusionEvent -> agent queue
"""
from __future__ import annotations

import json
import logging
import queue
import time
from collections.abc import Callable

import cv2
import numpy as np

from farmguard.config import Config
from farmguard.events import Detection, FramePacket, IntrusionEvent
from farmguard.fast.alarm import AlarmPlayer, AlarmRecord
from farmguard.fast.detector import Detector
from farmguard.fast.framebuffer import FrameBuffer
from farmguard.fast.motion import MotionGate
from farmguard.fast.timing import LatencyLog, StageTimer
from farmguard.fast.trigger import Trigger
from farmguard.fast.zones import ZoneSet

log = logging.getLogger(__name__)


class FastPath:
    def __init__(self, cfg: Config, detector: Detector, zones: ZoneSet | None, alarm: AlarmPlayer,
                 buffer: FrameBuffer, event_queue: queue.Queue[IntrusionEvent] | None,
                 latency_log: LatencyLog | None = None,
                 on_frame: Callable[[FramePacket, list[IntrusionEvent]], None] | None = None):
        self.cfg = cfg
        self.detector = detector
        self.zones = zones
        self.alarm = alarm
        self.buffer = buffer
        self.event_queue = event_queue
        self.on_frame = on_frame
        self.timer = StageTimer()
        self.latency_log = latency_log
        self.trigger = Trigger(window=cfg.get("trigger.window", 7),
                               min_hits=cfg.get("trigger.min_hits", 4),
                               track_cooldown_s=cfg.get("trigger.track_cooldown_s", 60),
                               species_cooldown_s=cfg.get("trigger.species_cooldown_s", 60))
        self.motion = MotionGate(enabled=cfg.get("motion_gate.enabled", True),
                                 min_changed_fraction=cfg.get("motion_gate.min_changed_fraction", 0.002),
                                 warmup_frames=cfg.get("motion_gate.warmup_frames", 15))
        self.events: list[IntrusionEvent] = []
        # species -> first sound recommended by the agent after the previous event (written by
        # the agent thread, read here; a dict lookup, so the alarm path still never waits on an LLM)
        self.first_sound_override: dict[str, str] = {}
        self._pending_latency: dict[str, tuple[float, float, float]] = {}
        self._last_dets: list[Detection] = []
        self._events_jsonl = cfg.path("logging.events_jsonl") if cfg.get("logging.events_jsonl") else None
        prev = alarm.on_played
        alarm.on_played = lambda rec: (self._on_alarm_played(rec), prev and prev(rec))

    # ---- alarm latency bookkeeping -----------------------------------------------------
    def _on_alarm_played(self, rec: AlarmRecord) -> None:
        if rec.event_id and rec.event_id in self._pending_latency and self.latency_log:
            t_read, t_det, t_first = self._pending_latency.pop(rec.event_id)
            self.latency_log.add(rec.event_id, rec.sound, rec.reason, t_read, t_det,
                                 rec.t_requested, rec.t_started, t_first, rec.backend, rec.ok)

    # ---- helpers -------------------------------------------------------------------------
    def _first_sound(self, species: str) -> str:
        info = self.cfg.species[species]
        override = self.first_sound_override.get(species)
        if override in info.allowed_sounds:
            return override
        if self.cfg.get("alarm.first_sound_policy") == "random_allowed":
            return str(np.random.choice(info.allowed_sounds))
        return info.default_sound

    def _is_target(self, det: Detection) -> bool:
        return det.species is not None and det.label not in self.cfg.non_targets

    def process(self, idx: int, video_time: float, frame: np.ndarray, t_read: float) -> FramePacket:
        if self.zones is None:
            h, w = frame.shape[:2]
            self.zones = ZoneSet.full_frame(w, h)
        else:
            self.zones = self.zones.rescaled(frame.shape[1], frame.shape[0])

        t0 = time.perf_counter()
        ran_detector = self.motion.should_run(frame, tracks_active=bool(self._last_dets))
        if ran_detector:
            dets = self.detector.detect(frame, idx)
            self._last_dets = dets
        else:
            dets = self._last_dets  # nothing moved: carry the last detections forward
        t_det = time.perf_counter()
        self.timer.add("detect", t_det - t0)

        targets = [d for d in dets if self._is_target(d)]
        min_area = self.cfg.get("trigger.min_box_area_frac", 0.0) * frame.shape[0] * frame.shape[1]
        self.trigger.min_box_area = min_area
        in_zone = [(d, z) for d in targets if (z := self.zones.zone_of(d))]
        # Only real detector runs vote: carried-forward boxes would let one false positive on a still
        # scene collect all 4 votes by itself.
        hits = (self.trigger.update(in_zone, {Trigger.key_for(d) for d in targets}, video_time, t_read)
                if ran_detector else [])

        new_events: list[IntrusionEvent] = []
        for hit in hits:
            det = hit.detection
            sound = self._first_sound(det.species)
            event_id = IntrusionEvent.new_id()
            self._pending_latency[event_id] = (t_read, t_det, hit.first_stamp)
            rec = self.alarm.play(sound, reason="first_alarm", event_id=event_id)  # non-blocking
            ev = IntrusionEvent(event_id=event_id, species=det.species, label=det.label,
                                track_id=det.track_id, video_time=video_time, frame_index=idx,
                                bbox=det.xyxy, confidence=det.confidence, zone=hit.zone,
                                first_sound=sound, t_read=t_read,
                                t_alarm_requested=rec.t_requested)
            new_events.append(ev)
            self.events.append(ev)
            log.info("INTRUSION %s %s track=%s zone=%s t=%.1fs -> %s", event_id, det.species,
                     det.track_id, hit.zone, video_time, sound)
            if self.event_queue is not None:
                self.event_queue.put_nowait(ev)  # unbounded queue: never blocks the frame loop
            if self._events_jsonl:
                self._events_jsonl.parent.mkdir(parents=True, exist_ok=True)
                with self._events_jsonl.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"type": "intrusion", **ev.to_dict()}) + "\n")
        self.timer.add("trigger", time.perf_counter() - t_det)

        pkt = FramePacket(index=idx, video_time=video_time, frame=frame, detections=dets,
                          in_zone=[d for d, _ in in_zone], t_read=t_read, t_detect_done=t_det)
        self.buffer.push(pkt)
        if self.on_frame:
            self.on_frame(pkt, new_events)
        return pkt

    def run(self, frames, max_frames: int | None = None) -> None:
        try:
            for n, (idx, vt, frame, t_read) in enumerate(frames):
                self.process(idx, vt, frame, t_read)
                if max_frames and n + 1 >= max_frames:
                    break
        finally:
            self.buffer.finish()
            if self.latency_log:
                # Wait for the alarm thread: a fixed sleep would drop exactly the slowest alarms.
                deadline = time.monotonic() + 10.0
                while self._pending_latency and time.monotonic() < deadline:
                    time.sleep(0.01)
                if self._pending_latency:
                    log.error("%d alarms never started; not in the latency log",
                              len(self._pending_latency))
                self.latency_log.save()


def annotate(pkt: FramePacket, zones: ZoneSet | None) -> np.ndarray:
    img = pkt.frame.copy()
    if zones:
        zones.draw(img)
    for d in pkt.detections:
        x1, y1, x2, y2 = map(int, d.xyxy)
        hot = any(d is z for z in pkt.in_zone)
        color = (0, 0, 255) if hot else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{d.label} #{d.track_id} {d.confidence:.2f}", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img
