"""Layer 2: the GenAI agent. One worker thread consumes IntrusionEvents from the fast path.

For each event:
  1. (parallel) VLM check on crop + full frame + neighbours  -> species, count, behaviour
  2. retreat check on the seconds after the first alarm      -> gone / leaving / still_present
  3. while not gone and attempts < max: policy picks next sound -> validator -> alarm -> retreat check
  4. LLM writes the farmer alert (template fallback) -> console / Telegram (+ voice)
  5. everything is logged to SQLite memory (feeds the next decision) and events.jsonl
The fast path never waits for any of this.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from farmguard.agent.clip_builder import build_event_images
from farmguard.agent.memory import Memory, hour_bucket
from farmguard.agent.notifier import Notifier
from farmguard.agent.policy import Policy, PolicyContext
from farmguard.agent.retreat import RetreatChecker, RetreatResult
from farmguard.agent.validators import ESCALATE
from farmguard.agent.vlm_check import VLMChecker
from farmguard.config import Config
from farmguard.events import IntrusionEvent, PolicyDecision, RetreatStatus, VLMReport
from farmguard.fast.alarm import AlarmPlayer
from farmguard.fast.framebuffer import FrameBuffer

log = logging.getLogger(__name__)


@dataclass
class AttemptLog:
    attempt_no: int
    sound: str
    source: str
    reason: str
    status: str
    time_to_leave_s: float | None
    tracker_status: str
    vlm_status: str | None
    policy_latency_s: float = 0.0


@dataclass
class EventResult:
    event: dict[str, Any]
    vlm: dict[str, Any] | None
    attempts: list[AttemptLog] = field(default_factory=list)
    final_status: str = "uncertain"
    escalated: bool = False
    farmer: dict[str, Any] | None = None
    agent_seconds: float = 0.0


class AgentWorker:
    def __init__(self, cfg: Config, buffer: FrameBuffer, alarm: AlarmPlayer, memory: Memory,
                 vlm: VLMChecker | None, retreat: RetreatChecker, policy: Policy,
                 notifier: Notifier, on_result: Callable[[EventResult], None] | None = None):
        self.cfg = cfg
        self.buffer = buffer
        self.alarm = alarm
        self.memory = memory
        self.vlm = vlm
        self.retreat = retreat
        self.policy = policy
        self.notifier = notifier
        self.on_result = on_result
        self.queue: "queue.Queue[IntrusionEvent | None]" = queue.Queue()
        self.results: list[EventResult] = []
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vlm")
        self._thread = threading.Thread(target=self._loop, daemon=True, name="agent")
        self._last_played: dict[str, float] = {}
        self.first_sound_cache: dict[str, str] | None = None  # set by build_system (shared with fast path)
        self._jsonl = cfg.path("logging.events_jsonl") if cfg.get("logging.events_jsonl") else None

    def start(self) -> "AgentWorker":
        self._thread.start()
        return self

    def stop(self, wait: bool = True, timeout: float | None = None) -> None:
        self.queue.put(None)
        if wait:
            # short joins: on Windows a single long join() ignores Ctrl+C until it times out
            deadline = None if timeout is None else time.monotonic() + timeout
            while self._thread.is_alive() and (deadline is None or time.monotonic() < deadline):
                self._thread.join(0.2)

    def _loop(self) -> None:
        while True:
            ev = self.queue.get()
            if ev is None:
                return
            try:
                res = self.handle(ev)
                self.results.append(res)
                if self.on_result:
                    self.on_result(res)
            except Exception:
                log.exception("Agent failed on event %s", ev.event_id)

    # ---- one event ---------------------------------------------------------------------
    def _vlm_task(self, ev: IntrusionEvent) -> VLMReport | None:
        if self.vlm is None:
            return None
        before = self.cfg.get("agent.clip.frames_before", 2)
        after = self.cfg.get("agent.clip.frames_after", 2)
        self.buffer.wait_until(ev.video_time + 1.0, timeout_s=30)
        pkts = self.buffer.around(ev.video_time, before, after)
        trig = min(pkts, key=lambda p: abs(p.video_time - ev.video_time)) if pkts else None
        if trig is None:
            return None
        images, layout = build_event_images(trig, pkts, ev.bbox,
                                            self.cfg.get("agent.clip.crop_padding", 0.25),
                                            self.cfg.get("agent.clip.max_side", 768))
        return self.vlm.check(images, layout, ev.species)

    def handle(self, ev: IntrusionEvent) -> EventResult:
        t_start = time.perf_counter()
        info = self.cfg.species[ev.species]
        label = ev.species.replace("_", " ")
        max_attempts = self.cfg.get("agent.policy.max_attempts", 3)
        bucket = hour_bucket(ev.wall_time)
        vlm_future = self._pool.submit(self._vlm_task, ev)

        tried: list[str] = [ev.first_sound]
        self._last_played[ev.first_sound] = ev.video_time
        attempts: list[AttemptLog] = []
        rr = self.retreat.check(ev.species, label, ev.video_time)
        attempts.append(self._log_attempt(ev, 1, ev.first_sound, "fast_path_default",
                                          "first alarm (no GenAI in the alarm path)", rr, bucket))

        report: VLMReport | None = None
        try:
            report = vlm_future.result(timeout=self.cfg.get("agent.timeout_s", 120) + 30)
        except Exception as exc:
            log.warning("VLM check failed: %s", exc)
        behaviour = report.behaviour if report else "unclear"
        count = report.count if report else 1

        status = rr.status
        escalated = False
        while status not in (RetreatStatus.GONE, RetreatStatus.LEAVING):
            now = self.buffer.latest_time
            ctx = PolicyContext(species=ev.species, species_label=label, allowed=info.allowed_sounds,
                                default_sound=info.default_sound, tried_this_event=list(tried),
                                attempt_no=len(tried), max_attempts=max_attempts, hour_bucket=bucket,
                                behaviour=behaviour, count=count,
                                stats=self.memory.sound_stats(ev.species, bucket),
                                last_played_at=dict(self._last_played), now=now,
                                sound_cooldown_s=self.cfg.get("agent.policy.sound_cooldown_s", 20))
            decision: PolicyDecision = self.policy.decide(ctx)
            if decision.action == ESCALATE:
                escalated = True
                log.info("Event %s: escalate to farmer (%s)", ev.event_id, decision.reason)
                break
            t_play = self.buffer.latest_time
            self.alarm.play(decision.action, reason=f"agent:{decision.source}", event_id=ev.event_id)
            self._last_played[decision.action] = t_play
            tried.append(decision.action)
            rr = self.retreat.check(ev.species, label, t_play)
            attempts.append(self._log_attempt(ev, len(tried), decision.action, decision.source,
                                              decision.reason, rr, bucket, decision.latency_s))
            status = rr.status

        final = status.value
        summary = self.notifier.build_summary(ev.species, ev.wall_time,
                                              [a.sound for a in attempts], final, len(attempts),
                                              count, behaviour)
        farmer = self.notifier.send(summary)
        result = EventResult(event=ev.to_dict(), vlm=asdict(report) if report else None,
                             attempts=attempts, final_status=final, escalated=escalated,
                             farmer=farmer, agent_seconds=time.perf_counter() - t_start)
        self.memory.add_event(event_id=ev.event_id, wall_time=ev.wall_time, video_time=ev.video_time,
                              species=ev.species, label=ev.label, zone=ev.zone,
                              track_id=ev.track_id,
                              vlm_species=report.species if report else None,
                              vlm_behaviour=behaviour,
                              vlm_confidence=report.confidence if report else None,
                              final_status=final, attempts=len(attempts), escalated=int(escalated),
                              farmer_message=farmer["message"],
                              extra={"vlm_note": report.note if report else None})
        self._write_jsonl(result)
        self._update_first_sound(ev, bucket, max_attempts)
        return result

    def _update_first_sound(self, ev: IntrusionEvent, bucket: str, max_attempts: int) -> None:
        """Off the critical path: pick the first sound for this species' NEXT visit."""
        if self.first_sound_cache is None or self.policy.name not in ("llm", "bandit", "random"):
            return
        info = self.cfg.species[ev.species]
        ctx = PolicyContext(species=ev.species, species_label=ev.species.replace("_", " "),
                            allowed=info.allowed_sounds, default_sound=info.default_sound,
                            tried_this_event=[], attempt_no=0, max_attempts=max_attempts,
                            hour_bucket=bucket, stats=self.memory.sound_stats(ev.species, bucket),
                            sound_cooldown_s=0)
        try:
            rec = self.policy.decide(ctx).action
            if rec in info.allowed_sounds:
                self.first_sound_cache[ev.species] = rec
        except Exception:
            log.exception("first-sound update failed")

    def _log_attempt(self, ev: IntrusionEvent, n: int, sound: str, source: str, reason: str,
                     rr: RetreatResult, bucket: str, latency: float = 0.0) -> AttemptLog:
        self.memory.add_attempt(ev.event_id, n, ev.species, sound, source, reason, rr.status.value,
                                rr.time_to_leave_s, rr.tracker_status.value,
                                rr.vlm_status.value if rr.vlm_status else None, bucket=bucket)
        log.info("Event %s attempt %d: %s -> %s (tracker=%s vlm=%s)", ev.event_id, n, sound,
                 rr.status.value, rr.tracker_status.value,
                 rr.vlm_status.value if rr.vlm_status else None)
        return AttemptLog(n, sound, source, reason, rr.status.value, rr.time_to_leave_s,
                          rr.tracker_status.value, rr.vlm_status.value if rr.vlm_status else None,
                          latency)

    def _write_jsonl(self, res: EventResult) -> None:
        if not self._jsonl:
            return
        self._jsonl.parent.mkdir(parents=True, exist_ok=True)
        with self._jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"type": "agent_result", **asdict(res)}, ensure_ascii=False,
                                default=str) + "\n")
