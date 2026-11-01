"""Wires Layer 1 and Layer 2 together. Used by the CLI, the Streamlit app and the tests."""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from farmguard.agent.llm.providers import build_client
from farmguard.agent.memory import Memory
from farmguard.agent.notifier import Notifier
from farmguard.agent.orchestrator import AgentWorker, EventResult
from farmguard.agent.policy import build_policy
from farmguard.agent.retreat import RetreatChecker
from farmguard.agent.vlm_check import VLMChecker
from farmguard.config import Config
from farmguard.events import FramePacket, IntrusionEvent
from farmguard.fast.alarm import AlarmPlayer
from farmguard.fast.detector import Detector, build_detector
from farmguard.fast.framebuffer import FrameBuffer
from farmguard.fast.pipeline import FastPath
from farmguard.fast.timing import LatencyLog
from farmguard.fast.zones import ZoneSet

log = logging.getLogger(__name__)


@dataclass
class System:
    cfg: Config
    fast: FastPath
    agent: AgentWorker | None
    alarm: AlarmPlayer
    buffer: FrameBuffer
    memory: Memory
    latency: LatencyLog


def build_system(cfg: Config, detector: Detector | None = None, llm_client=None,
                 on_frame: Callable[[FramePacket, list[IntrusionEvent]], None] | None = None,
                 on_result: Callable[[EventResult], None] | None = None) -> System:
    buffer = FrameBuffer(cfg.get("video.buffer_seconds", 30))
    alarm = AlarmPlayer(cfg.path("alarm.sounds_dir"), cfg.get("alarm.backend", "auto"))
    zones_file = cfg.path("zones.file")
    zones = ZoneSet.load(zones_file, cfg.get("zones.anchor", "bottom_center")) if zones_file.exists() else None
    if zones is None:
        log.warning("No zones file at %s: the whole frame counts as the field", zones_file)
    memory = Memory(cfg.path("memory.db_path"))
    agent_enabled = bool(cfg.get("agent.enabled", True))
    latency = LatencyLog(cfg.path("logging.timing_csv"), agent_enabled)
    agent = None
    if agent_enabled:
        client = llm_client or build_client(cfg, "vlm")
        policy_client = llm_client or (client if cfg.get("agent.llm_model") == cfg.get("agent.vlm_model")
                                       else build_client(cfg, "llm"))
        agent = AgentWorker(
            cfg, buffer, alarm, memory,
            vlm=VLMChecker(cfg, client),
            retreat=RetreatChecker(buffer, client, cfg.get("agent.retreat.observe_seconds", 12),
                                   cfg.get("agent.retreat.gone_frames", 15),
                                   cfg.get("agent.retreat.use_vlm", True)),
            policy=build_policy(cfg.get("agent.policy.type", "llm"), policy_client),
            notifier=Notifier(cfg, policy_client), on_result=on_result).start()
    fast = FastPath(cfg, detector or build_detector(cfg), zones, alarm, buffer,
                    agent.queue if agent else None, latency, on_frame)
    if agent and cfg.get("agent.adaptive_first_sound", True):
        agent.first_sound_cache = fast.first_sound_override
    return System(cfg, fast, agent, alarm, buffer, memory, latency)


def run_video(system: System, frames, realtime: bool = False, max_frames: int | None = None,
              drain_timeout: float = 600) -> None:
    """Run the fast path over `frames`; if realtime, pace frames to video time (demo mode, so
    the agent's retreat windows line up with wall-clock time)."""
    if realtime:
        def paced(it):
            t0 = time.perf_counter()
            for item in it:
                delay = item[1] - (time.perf_counter() - t0)
                if delay > 0:
                    time.sleep(delay)
                # re-stamp t_read after pacing so E1 latency excludes the artificial wait
                yield (item[0], item[1], item[2], time.perf_counter())
        frames = paced(frames)
    system.fast.run(frames, max_frames)
    if system.agent:
        system.agent.stop(wait=True, timeout=drain_timeout)
    system.alarm.stop()
