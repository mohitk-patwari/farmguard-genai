"""Retreat check: did the animal leave after the deterrent?

This is the step NO reviewed system implements for real (AEGISFARM `services/observation.py`
returns a hard-coded "retreated"). We fuse two signals and keep both for experiment E3:
  - tracker evidence: frames after the sound in which the species is still inside the zone
  - VLM evidence: a few frames from the window, last-heavy, judged gone/leaving/still_present
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from farmguard.agent.clip_builder import sample_frames, to_jpeg
from farmguard.agent.llm.base import LLMClient
from farmguard.agent.prompts import SYSTEM_VISION, retreat_prompt, retreat_schema
from farmguard.events import FramePacket, RetreatStatus
from farmguard.fast.framebuffer import FrameBuffer

log = logging.getLogger(__name__)


@dataclass
class RetreatResult:
    status: RetreatStatus
    time_to_leave_s: float | None
    tracker_status: RetreatStatus
    vlm_status: RetreatStatus | None
    vlm_confidence: str | None
    frames_seen: int
    observation: str = ""


def tracker_evidence(pkts: list[FramePacket], species: str, t0: float,
                     gone_frames: int) -> tuple[RetreatStatus, float | None]:
    """Pure function, unit-tested. A frame counts as 'present' if any detection of the species
    is inside the zone (track ids can change after occlusion, so species-level is safer)."""
    if not pkts:
        return RetreatStatus.UNCERTAIN, None
    present = [any(d.species == species and d.track_id in p.in_zone_track_ids for d in p.detections)
               or any(d.species == species and d.track_id is None for d in p.detections)
               for p in pkts]
    # trailing absent streak
    streak = 0
    for flag in reversed(present):
        if flag:
            break
        streak += 1
    if streak >= min(gone_frames, len(pkts)):
        first_absent = pkts[len(pkts) - streak]
        return RetreatStatus.GONE, max(0.0, first_absent.video_time - t0)
    if present[-1] and sum(present[-gone_frames:]) < len(present[-gone_frames:]) / 2:
        return RetreatStatus.LEAVING, None
    return RetreatStatus.STILL_PRESENT, None


def fuse(tracker: RetreatStatus, vlm: RetreatStatus | None, vlm_conf: str | None) -> RetreatStatus:
    if vlm is None or vlm == RetreatStatus.UNCERTAIN:
        return tracker
    strong = vlm_conf in ("high", "medium")
    if tracker == vlm:
        return tracker
    if tracker == RetreatStatus.GONE and vlm == RetreatStatus.STILL_PRESENT:
        # tracker lost it (occlusion, darkness) but the VLM still sees it
        return RetreatStatus.STILL_PRESENT if strong else RetreatStatus.UNCERTAIN
    if tracker == RetreatStatus.STILL_PRESENT and vlm in (RetreatStatus.GONE, RetreatStatus.LEAVING):
        return RetreatStatus.LEAVING if strong else RetreatStatus.STILL_PRESENT
    if {tracker, vlm} == {RetreatStatus.GONE, RetreatStatus.LEAVING}:
        return RetreatStatus.LEAVING if tracker == RetreatStatus.LEAVING else RetreatStatus.GONE
    return tracker


class RetreatChecker:
    def __init__(self, buffer: FrameBuffer, client: LLMClient | None, observe_seconds: float = 12,
                 gone_frames: int = 15, use_vlm: bool = True, n_vlm_frames: int = 4,
                 max_side: int = 640):
        self.buffer = buffer
        self.client = client
        self.observe_seconds = observe_seconds
        self.gone_frames = gone_frames
        self.use_vlm = use_vlm and client is not None
        self.n_vlm_frames = n_vlm_frames
        self.max_side = max_side

    def check(self, species: str, species_label: str, t_deterrent: float) -> RetreatResult:
        t_end = t_deterrent + self.observe_seconds
        self.buffer.wait_until(t_end, timeout_s=self.observe_seconds * 10 + 30)
        pkts = self.buffer.window(t_deterrent, t_end)
        tr_status, ttl = tracker_evidence(pkts, species, t_deterrent, self.gone_frames)
        vlm_status, vlm_conf, obs = None, None, ""
        if self.use_vlm and pkts:
            chosen = sample_frames(pkts, self.n_vlm_frames, last_heavy=True)
            images = [to_jpeg(p.frame, self.max_side) for p in chosen]
            res = self.client.generate_json(retreat_prompt(len(images), species_label),
                                            retreat_schema(), images, SYSTEM_VISION)
            if res.data:
                try:
                    vlm_status = RetreatStatus(res.data.get("status", "uncertain"))
                except ValueError:
                    vlm_status = RetreatStatus.UNCERTAIN
                vlm_conf = res.data.get("confidence")
                obs = str(res.data.get("observation", ""))[:300]
        status = fuse(tr_status, vlm_status, vlm_conf)
        if status != RetreatStatus.GONE:
            ttl = None if status != RetreatStatus.LEAVING else ttl
        return RetreatResult(status, ttl, tr_status, vlm_status, vlm_conf, len(pkts), obs)
