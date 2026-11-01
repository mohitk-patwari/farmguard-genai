"""E1 measurement: the stamps must be ordered, include the play call, and never hide a failure."""
from __future__ import annotations

import csv
import itertools
import queue
import threading
import time

import cv2
import numpy as np

from farmguard.config import load_config
from farmguard.events import Detection
from farmguard.fast.alarm import AlarmPlayer
from farmguard.fast.framebuffer import FrameBuffer
from farmguard.fast.pipeline import FastPath
from farmguard.fast.timing import LatencyLog
from farmguard.fast.trigger import Trigger
from farmguard.fast.video_source import VideoSource


class _Boar:
    """Sees a boar in every frame it is asked about; counts how often it was asked."""
    def __init__(self):
        self.calls = 0

    def detect(self, frame, idx):
        self.calls += 1
        return [Detection((10, 10, 100, 100), .9, "wild boar", "wild_boar", 1)]


def _fast(tmp_path, detector, backend="none", motion=False):
    cfg = load_config(overrides={"logging.events_jsonl": None, "motion_gate.enabled": motion})
    log = LatencyLog(tmp_path / "timing.csv", agent_enabled=False)
    alarm = AlarmPlayer(tmp_path, backend)
    return FastPath(cfg, detector, None, alarm, FrameBuffer(30), queue.Queue(), log), log, alarm


def _rows(tmp_path):
    with (tmp_path / "timing.csv").open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _frames(n):
    return [(i, i / 10, np.zeros((240, 320, 3), np.uint8), time.perf_counter()) for i in range(n)]


def test_stamps_are_ordered_and_include_the_vote(tmp_path):
    fast, _, alarm = _fast(tmp_path, _Boar())
    fast.run(iter(_frames(6)))
    alarm.stop()
    (row,) = _rows(tmp_path)
    t = [float(row[k]) for k in ("t_first_zone", "t_read", "t_detect_done", "t_alarm_requested",
                                 "t_alarm_started")]
    assert t == sorted(t) and t[0] < t[1]  # vote started on an earlier frame than the trigger
    assert float(row["first_zone_to_start_ms"]) >= float(row["read_to_start_ms"]) > 0
    assert row["backend"] == "none" and row["ok"] == "True"


def test_failed_alarm_is_not_logged_as_a_fast_alarm(tmp_path):
    # backend set but the WAV is missing
    fast, _, alarm = _fast(tmp_path, _Boar(), backend="aplay")
    fast.run(iter(_frames(6)))
    alarm.stop()
    (row,) = _rows(tmp_path)
    assert row["ok"] == "False" and row["read_to_start_ms"] == ""


def test_slow_alarm_thread_is_still_logged(tmp_path):
    """The old fixed 0.2 s sleep dropped alarms slower than that: survivorship bias in E1."""
    fast, _, alarm = _fast(tmp_path, _Boar())
    prev = alarm.on_played
    alarm.on_played = lambda rec: (time.sleep(0.5), prev(rec))
    fast.run(iter(_frames(6)))
    assert len(_rows(tmp_path)) == 1


def test_old_csv_header_is_rotated_not_misaligned(tmp_path):
    (tmp_path / "timing.csv").write_text("event_id,sound\na,b\n", encoding="utf-8")
    log = LatencyLog(tmp_path / "timing.csv", False)
    log.add("e", "s", "r", 1.0, 1.1, 1.2, 1.3)
    log.save()
    log.save()  # second save writes nothing new
    assert len(_rows(tmp_path)) == 1 and list(tmp_path.glob("timing.old*.csv"))


def test_carried_forward_boxes_do_not_vote(tmp_path):
    """One false positive on a still scene must not collect 4 votes by being carried forward."""
    det = _Boar()
    fast, _, alarm = _fast(tmp_path, det, motion=True)
    fast.motion.warmup_frames = 1
    fast.motion.keepalive = 1000
    still = np.zeros((240, 320, 3), np.uint8)
    for i in range(10):
        fast.process(i, i / 10, still, time.perf_counter())
    alarm.stop()
    assert det.calls < fast.trigger.min_hits and fast.events == []


def test_trigger_first_stamp_resets_when_votes_clear():
    tr = Trigger(window=3, min_hits=2)
    d = Detection((0, 0, 10, 10), .9, "wild boar", "wild_boar", 1)
    tr.update([(d, "f")], {"track:1"}, 0.0, stamp=100.0)
    for t in (1.0, 2.0, 3.0):
        tr.update([], {"track:1"}, t, stamp=100.0 + t)
    tr.update([(d, "f")], {"track:1"}, 4.0, stamp=104.0)
    (hit,) = tr.update([(d, "f")], {"track:1"}, 5.0, stamp=105.0)
    assert hit.first_stamp == 104.0


def _clip(path, n=30, fps=30):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for _ in range(n):
        vw.write(np.zeros((48, 64, 3), np.uint8))
    vw.release()


def test_live_file_ends_paces_and_counts_drops(tmp_path):
    clip = tmp_path / "c.mp4"
    _clip(clip)
    src = VideoSource(str(clip), live=True)
    t0 = time.perf_counter()
    got = []
    done = threading.Event()

    def consume():
        for item in src.frames():
            got.append(item)
            time.sleep(0.1)  # a detector slower than the camera
        done.set()

    threading.Thread(target=consume, daemon=True).start()
    assert done.wait(10), "live mode on a file must end at EOF"
    src.release()
    assert time.perf_counter() - t0 >= 0.8  # 30 frames at 30 fps: paced like a camera
    assert src.dropped > 0 and len(got) + src.dropped <= 30
    assert all(a[3] < b[3] for a, b in itertools.pairwise(got))
