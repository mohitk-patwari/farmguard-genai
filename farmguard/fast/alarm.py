"""Threaded alarm player. The frame loop only enqueues a request (microseconds) and never waits
for audio (SentaFito53 plays the gunshot in its own thread for the same reason).

Backends are tried in order for backend="auto": pygame, simpleaudio, aplay (Linux),
afplay (macOS), then "none" (log only, used in tests / headless servers).
Every played alarm records perf_counter timestamps for experiment E1 (latency).
"""
from __future__ import annotations

import logging
import platform
import queue
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class AlarmRecord:
    sound: str
    reason: str
    event_id: str | None
    t_requested: float
    t_started: float = 0.0
    backend: str = ""
    ok: bool = False
    error: str | None = None


class AlarmPlayer:
    def __init__(self, sounds_dir: str | Path, backend: str = "auto",
                 on_played: Callable[[AlarmRecord], None] | None = None):
        self.sounds_dir = Path(sounds_dir)
        self.backend = self._pick_backend(backend)
        self.on_played = on_played
        self.history: list[AlarmRecord] = []
        self._q: queue.Queue[AlarmRecord | None] = queue.Queue()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="alarm")
        self._thread.start()
        log.info("Alarm backend: %s", self.backend)

    # ---- backend selection -----------------------------------------------------------
    def _pick_backend(self, wanted: str) -> str:
        if wanted != "auto":
            return wanted
        try:
            import pygame
            pygame.mixer.init()  # importable is not enough: this raises when there is no audio device
            return "pygame"
        except Exception as exc:
            log.debug("pygame unavailable: %s", exc)
        try:
            import simpleaudio  # noqa: F401
            return "simpleaudio"
        except Exception as exc:
            log.debug("simpleaudio unavailable: %s", exc)
        if platform.system() == "Linux" and shutil.which("aplay"):
            return "aplay"
        if platform.system() == "Darwin" and shutil.which("afplay"):
            return "afplay"
        return "none"

    def path_for(self, sound: str) -> Path:
        return self.sounds_dir / f"{sound}.wav"

    # ---- public API --------------------------------------------------------------------
    def play(self, sound: str, reason: str = "", event_id: str | None = None) -> AlarmRecord:
        """Non-blocking. Returns the record immediately; t_started is filled by the worker."""
        rec = AlarmRecord(sound=sound, reason=reason, event_id=event_id,
                          t_requested=time.perf_counter())
        self._q.put(rec)
        return rec

    def stop(self) -> None:
        self._q.put(None)

    # ---- worker ------------------------------------------------------------------------
    def _loop(self) -> None:
        pg = None
        loaded: dict[str, object] = {}
        if self.backend == "pygame":
            try:  # backend="pygame" set explicitly skips _pick_backend's check
                import pygame
                pygame.mixer.init()
                pg = pygame
                # Decode up front: real recordings are large, and the disk read must not sit on the
                # alarm path.
                loaded = {p.stem: pg.mixer.Sound(str(p)) for p in self.sounds_dir.glob("*.wav")}
            except Exception:
                log.exception("pygame audio unavailable, alarms will be log-only")
                self.backend = "none"
        while True:
            rec = self._q.get()
            if rec is None:
                return
            rec.backend = self.backend
            path = self.path_for(rec.sound)
            try:
                if self.backend == "none":
                    rec.ok = True
                elif not path.exists():
                    raise FileNotFoundError(f"{path} (run: farmguard make-sounds)")
                elif self.backend == "pygame":
                    (loaded.get(rec.sound) or pg.mixer.Sound(str(path))).play()
                    rec.ok = True
                elif self.backend == "simpleaudio":
                    import simpleaudio
                    simpleaudio.WaveObject.from_wave_file(str(path)).play()
                    rec.ok = True
                elif self.backend in ("aplay", "afplay"):
                    subprocess.Popen([self.backend, str(path)], stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
                    rec.ok = True
                # Stamped after the play call so E1 includes it. A failed alarm keeps t_started=0
                # and logs an empty read_to_start_ms, so it can't count as a fast alarm. Mixer and
                # driver output delay are still not included (see docs/HANDOFF.md).
                rec.t_started = time.perf_counter()
            except Exception as exc:  # never crash the alarm thread
                rec.error = str(exc)
                log.error("Alarm failed: %s", exc)
            log.info("ALARM %s (%s) event=%s", rec.sound, rec.reason, rec.event_id)
            with self._lock:
                self.history.append(rec)
            if self.on_played:
                try:
                    self.on_played(rec)
                except Exception:
                    log.exception("on_played callback failed")
