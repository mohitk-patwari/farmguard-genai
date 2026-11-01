"""Regressions found running on Windows; none of them need Windows to run."""
from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from farmguard.agent.orchestrator import AgentWorker
from farmguard.fast.alarm import AlarmPlayer


def test_hindi_print_survives_legacy_stdout_encoding():
    env = {**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"}
    r = subprocess.run([sys.executable, "-c", "import farmguard; print('\\u0916\\u0947\\u0924')"],
                       env=env, capture_output=True, check=False)
    assert r.returncode == 0, r.stderr
    assert r.stdout.decode("utf-8").strip() == "खेत"


@pytest.mark.parametrize("backend", ["auto", "pygame"])
def test_alarm_survives_no_audio_device(monkeypatch, tmp_path, backend):
    pygame = pytest.importorskip("pygame")
    monkeypatch.setattr(pygame.mixer, "init", lambda: (_ for _ in ()).throw(pygame.error("no device")))
    player = AlarmPlayer(tmp_path, backend=backend)
    player.play("siren")
    player.stop()
    player._thread.join(5)
    assert player.history and player.history[0].backend != "pygame"


def test_agent_stop_can_be_interrupted():
    # a stuck agent thread stands in for a slow VLM call while draining on exit
    worker = SimpleNamespace(queue=queue.Queue(),
                             _thread=threading.Thread(target=time.sleep, args=(30,), daemon=True))
    worker._thread.start()
    threading.Timer(0.2, signal.raise_signal, [signal.SIGINT]).start()
    t0 = time.perf_counter()
    with pytest.raises(KeyboardInterrupt):
        AgentWorker.stop(worker, timeout=20)
    assert time.perf_counter() - t0 < 5
