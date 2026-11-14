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
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

from farmguard.agent.clip_builder import MAX_IMAGES, MAX_SIDE, build_event_images
from farmguard.agent.memory import Memory, hour_bucket
from farmguard.agent.notifier import Notifier
from farmguard.agent.policy import LLMPolicy, Policy, PolicyContext
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
    vlm_error: str | None = None


@dataclass
class EventResult:
    event: dict[str, Any]
    vlm: dict[str, Any] | None
    attempts: list[AttemptLog] = field(default_factory=list)
    final_status: str = "uncertain"
    escalated: bool = False
    farmer: dict[str, Any] | None = None
    agent_seconds: float = 0.0
    vlm_failed: bool = False
    detector_species: str = ""
    vlm_species: str | None = None
    vlm_confidence: str | None = None
    effective_species: str = ""          # what sounds, policy, alert and memory used
    overridden: bool = False             # a confident VLM species replaced the detector's
    possible_false_alarm: bool = False   # the VLM saw no animal, with high confidence
    species_uncertain: bool = False      # detector and VLM disagree and one says an immediate-escalation species
    effective_by_threshold: dict[str, str] = field(default_factory=dict)  # none/high/medium -> species (E2)


CONF_RANK = {"low": 0, "medium": 1, "high": 2}


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
        self.queue: queue.Queue[IntrusionEvent | None] = queue.Queue()
        self.results: list[EventResult] = []
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vlm")
        self._thread = threading.Thread(target=self._loop, daemon=True, name="agent")
        self._last_played: dict[str, float] = {}
        self._llm_answers = 0     # in-enum LLM answers, and how many picked the first shown option
        self._first_position = 0
        self.first_sound_cache: dict[str, str] | None = None  # set by build_system (shared with fast path)
        self._jsonl = cfg.path("logging.events_jsonl") if cfg.get("logging.events_jsonl") else None
        if isinstance(policy, LLMPolicy) and not policy.decision_log and cfg.get("agent.policy.decision_log"):
            policy.decision_log = cfg.path("agent.policy.decision_log")

    def start(self) -> AgentWorker:
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

    def vlm_counts(self) -> dict[str, int]:
        """For run results: a failed VLM call falls back quietly, so it has to be counted."""
        attempts = [a for r in self.results for a in r.attempts]
        return {"event_checks": len(self.results), "event_failed": sum(r.vlm_failed for r in self.results),
                "retreat_checks": sum(a.vlm_status is not None or a.vlm_error is not None for a in attempts),
                "retreat_failed": sum(a.vlm_error is not None for a in attempts),
                "species_overridden": sum(r.overridden for r in self.results),
                "possible_false_alarm": sum(r.possible_false_alarm for r in self.results),
                "species_uncertain": sum(r.species_uncertain for r in self.results)}

    def run_counts(self) -> dict[str, Any]:
        """For run results: windows cut short by the end of the clip, and the LLM's position bias."""
        return {"truncated_windows": sum(a.status == RetreatStatus.TRUNCATED.value
                                         for r in self.results for a in r.attempts),
                "llm_answers": self._llm_answers,
                "first_position_rate": self._first_position / self._llm_answers if self._llm_answers else None}

    def _decide(self, ctx: PolicyContext) -> PolicyDecision:
        d = self.policy.decide(ctx)
        if d.shown_position is not None:
            self._llm_answers += 1
            self._first_position += d.shown_position == 0
        return d

    def effective_species(self, detector: str, report: VLMReport | None,
                          min_confidence: str | None = None) -> tuple[str, bool, bool]:
        """(species, overridden, possible_false_alarm). The detector's label was wrong on every event of
        T1's boar clip, so a confident VLM answer wins (agent.vlm.override_*). A failed VLM call,
        'unclassifiable', 'other_animal' or a doubtful answer keeps the detector's label.
        min_confidence given: use that threshold whatever the config (logged per event for E2)."""
        if report is None or report.error:
            return detector, False, False
        if min_confidence is None:
            if not self.cfg.get("agent.vlm.override_species", True):
                return detector, False, False
            min_confidence = str(self.cfg.get("agent.vlm.override_min_confidence", "high"))
        if report.species == "none" and report.confidence == "high":
            return detector, False, True
        need = CONF_RANK.get(min_confidence, 2)
        if (report.species in self.cfg.species and report.species != detector
                and CONF_RANK.get(report.confidence, -1) >= need):
            return report.species, True, False
        return detector, False, False

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
            log.warning("VLM check skipped for %s: no frames in the buffer", ev.event_id)
            return None
        images, layout = build_event_images(trig, pkts, ev.bbox,
                                            self.cfg.get("agent.clip.crop_padding", 0.25),
                                            self.cfg.get("agent.clip.max_side", MAX_SIDE),
                                            max_images=self.cfg.get("agent.clip.max_images", MAX_IMAGES))
        return self.vlm.check(images, layout, ev.species)

    def handle(self, ev: IntrusionEvent) -> EventResult:
        t_start = time.perf_counter()
        max_attempts = self.cfg.get("agent.policy.max_attempts", 3)
        bucket = hour_bucket(ev.wall_time)
        vlm_future = self._pool.submit(self._vlm_task, ev)

        tried: list[str] = [ev.first_sound]
        self._last_played[ev.first_sound] = ev.video_time
        attempts: list[AttemptLog] = []
        rr = self.retreat.check(ev.species.replace("_", " "), ev.video_time)

        report: VLMReport | None = None
        try:
            # the task's own worst case: 30 s waiting for frames + max_total_s for the VLM call
            report = vlm_future.result(timeout=self.cfg.get("agent.max_total_s", 180) + 35)
        except Exception as exc:
            log.warning("VLM check failed: %s", exc)
        behaviour = report.behaviour if report else "unclear"
        count = report.count if report else 1
        species, overridden, false_alarm = self.effective_species(ev.species, report)
        if overridden:
            log.info("Event %s: VLM says %s (%s), detector said %s: using %s", ev.event_id,
                     species, report.confidence, ev.species, species)
        by_threshold = {"none": ev.species, **{c: self.effective_species(ev.species, report, c)[0]
                                               for c in ("high", "medium")}}
        # Smoke test 2: detector "elephant", VLM "wild boar" (medium). Elephant may go straight to the farmer,
        # so a disputed elephant must not do that on the detector's word alone, and the farmer is told both.
        immediate = self._immediate()
        disagree = (report is not None and not report.error
                    and report.species not in (ev.species, "unclassifiable"))
        uncertain = disagree and (ev.species in immediate or report.species in immediate)
        event_immediate = () if uncertain and not overridden else immediate
        info = self.cfg.species[species]
        label = species.replace("_", " ")
        # VLM saw no animal and neither does the retreat check: stop, and record nothing in outcome
        # memory (a sound "working" on an empty field is false evidence)
        quiet = false_alarm and RetreatStatus.STILL_PRESENT not in (rr.tracker_status, rr.vlm_status)
        attempts.append(self._log_attempt(ev, species, 1, ev.first_sound, "fast_path_default",
                                          "first alarm (no GenAI in the alarm path)", rr, bucket,
                                          remember=not quiet))

        status = rr.status
        escalated = False
        # TRUNCATED: the clip or stream ended, so nothing more can be observed (or learned)
        while not quiet and status not in (RetreatStatus.GONE, RetreatStatus.LEAVING, RetreatStatus.TRUNCATED):
            now = self.buffer.latest_time
            ctx = PolicyContext(species=species, species_label=label, allowed=info.allowed_sounds,
                                default_sound=info.default_sound, tried_this_event=list(tried),
                                attempt_no=len(tried), max_attempts=max_attempts, hour_bucket=bucket,
                                behaviour=behaviour, count=count,
                                stats=self._stats(species, bucket),
                                last_played_at=dict(self._last_played), now=now,
                                sound_cooldown_s=self.cfg.get("agent.policy.sound_cooldown_s", 20),
                                event_id=ev.event_id,
                                immediate_escalation_species=event_immediate)
            decision: PolicyDecision = self._decide(ctx)
            if decision.action == ESCALATE:
                escalated = True
                log.info("Event %s: escalate to farmer (%s)", ev.event_id, decision.reason)
                break
            t_play = self.buffer.latest_time
            self.alarm.play(decision.action, reason=f"agent:{decision.source}", event_id=ev.event_id)
            self._last_played[decision.action] = t_play
            tried.append(decision.action)
            rr = self.retreat.check(label, t_play)
            attempts.append(self._log_attempt(ev, species, len(tried), decision.action, decision.source,
                                              decision.reason, rr, bucket, decision.latency_s))
            status = rr.status

        final = status.value
        if quiet:
            note = self.notifier.false_alarm_note(species, ev.wall_time)
            farmer = self.notifier.deliver(note, note, "false_alarm_note")
        else:
            farmer = self.notifier.send(self.notifier.build_summary(
                species, ev.wall_time, [a.sound for a in attempts], final, len(attempts), count, behaviour,
                species_check=(ev.species, report.species) if uncertain else None))
        result = EventResult(event=ev.to_dict(), vlm=asdict(report) if report else None,
                             attempts=attempts, final_status=final, escalated=escalated,
                             farmer=farmer, agent_seconds=time.perf_counter() - t_start,
                             vlm_failed=self.vlm is not None and (report is None or report.error is not None),
                             detector_species=ev.species, vlm_species=report.species if report else None,
                             vlm_confidence=report.confidence if report else None,
                             effective_species=species, overridden=overridden,
                             possible_false_alarm=false_alarm, species_uncertain=uncertain,
                             effective_by_threshold=by_threshold)
        self.memory.add_event(event_id=ev.event_id, wall_time=ev.wall_time, video_time=ev.video_time,
                              species=species, label=ev.label, zone=ev.zone,
                              track_id=ev.track_id,
                              vlm_species=report.species if report else None,
                              vlm_behaviour=behaviour,
                              vlm_confidence=report.confidence if report else None,
                              final_status=final, attempts=len(attempts), escalated=int(escalated),
                              farmer_message=farmer["message"],
                              extra={"vlm_note": report.note if report else None,
                                     "detector_species": ev.species, "overridden": overridden,
                                     "possible_false_alarm": false_alarm, "species_uncertain": uncertain,
                                     "effective_by_threshold": by_threshold})
        self._write_jsonl(result)
        if not quiet:
            self._update_first_sound(ev, species, bucket, max_attempts)
        return result

    def _stats(self, species: str, bucket: str) -> dict[str, dict[str, Any]]:
        return self.memory.sound_stats(species, bucket, window=self.cfg.get("agent.policy.bandit_window", 20))

    def _immediate(self) -> tuple[str, ...]:
        return tuple(self.cfg.get("agent.policy.immediate_escalation_species", ["elephant"]) or ())

    def _update_first_sound(self, ev: IntrusionEvent, species: str, bucket: str, max_attempts: int) -> None:
        """Off the critical path: pick the first sound for this species' NEXT visit."""
        if self.first_sound_cache is None or self.policy.name in ("fixed", "rule_chain"):
            return
        info = self.cfg.species[species]
        ctx = PolicyContext(species=species, species_label=species.replace("_", " "),
                            allowed=info.allowed_sounds, default_sound=info.default_sound,
                            tried_this_event=[], attempt_no=0, max_attempts=max_attempts,
                            hour_bucket=bucket, stats=self._stats(species, bucket),
                            sound_cooldown_s=0, event_id=ev.event_id,
                            immediate_escalation_species=self._immediate())
        try:
            rec = self._decide(ctx).action
            if rec in info.allowed_sounds:
                self.first_sound_cache[species] = rec
        except Exception:
            log.exception("first-sound update failed")

    def _log_attempt(self, ev: IntrusionEvent, species: str, n: int, sound: str, source: str, reason: str,
                     rr: RetreatResult, bucket: str, latency: float = 0.0, remember: bool = True) -> AttemptLog:
        if remember and rr.status != RetreatStatus.TRUNCATED:  # no verdict, so no outcome to learn from
            self.memory.add_attempt(ev.event_id, n, species, sound, source, reason, rr.status.value,
                                    rr.time_to_leave_s, rr.tracker_status.value,
                                    rr.vlm_status.value if rr.vlm_status else None, bucket=bucket)
        log.info("Event %s attempt %d: %s -> %s (tracker=%s vlm=%s)", ev.event_id, n, sound,
                 rr.status.value, rr.tracker_status.value,
                 rr.vlm_status.value if rr.vlm_status else None)
        return AttemptLog(n, sound, source, reason, rr.status.value, rr.time_to_leave_s,
                          rr.tracker_status.value, rr.vlm_status.value if rr.vlm_status else None,
                          latency, rr.vlm_error)

    def _write_jsonl(self, res: EventResult) -> None:
        if not self._jsonl:
            return
        self._jsonl.parent.mkdir(parents=True, exist_ok=True)
        with self._jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"type": "agent_result", **asdict(res)}, ensure_ascii=False,
                                default=str) + "\n")
