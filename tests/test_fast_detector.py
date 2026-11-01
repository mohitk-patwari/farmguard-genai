"""UltralyticsDetector against a tiny fake of ultralytics Results, plus the no-network guard.

The fake mirrors what ultralytics 8.4.197 was seen to return: boxes.id is None when untracked, and
names is a dict. Tests marked with real weights are skipped when weights/ is empty (CI).
"""
from __future__ import annotations

import ast
import socket
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from farmguard.fast.detector import UltralyticsDetector, apply_clahe, results_to_detections

ROOT = Path(__file__).resolve().parent.parent


class _T(list):
    """Enough of a tensor: .tolist() and .int()."""
    def tolist(self):
        return list(self)

    def int(self):
        return _T(int(x) for x in self)


class _Boxes(SimpleNamespace):
    def __len__(self):
        return len(self.conf)


def _results(rows, ids=None, names=None):
    boxes = _Boxes(xyxy=_T(r[:4] for r in rows), conf=_T(r[4] for r in rows),
                   cls=_T(r[5] for r in rows), id=None if ids is None else _T(ids))
    return [SimpleNamespace(boxes=boxes, names=names or {0: "boar", 1: "person"})]


def test_tracked_boxes_keep_their_ids():
    out = results_to_detections(_results([[1, 2, 3, 4, .9, 0]], ids=[7]), .25, lambda s: "wild_boar")
    assert [(d.track_id, d.label, d.species, d.xyxy) for d in out] == [(7, "boar", "wild_boar", (1, 2, 3, 4))]


def test_tracking_failure_gives_none_ids_not_a_crash():
    out = results_to_detections(_results([[1, 2, 3, 4, .9, 0], [5, 6, 7, 8, .8, 0]]), .25)
    assert [d.track_id for d in out] == [None, None]


def test_low_score_box_kept_only_when_it_continues_a_track():
    rows = [[1, 2, 3, 4, .15, 0]]
    assert results_to_detections(_results(rows), .25) == []
    assert [d.track_id for d in results_to_detections(_results(rows, ids=[3]), .25)] == [3]


def test_names_as_list_and_ignored_classes():
    res = _results([[1, 2, 3, 4, .9, 0], [1, 2, 3, 4, .9, 1]], names=["Monkey", "person"])
    assert [d.label for d in results_to_detections(res, 0, ignore={"person"})] == ["monkey"]


def test_empty_and_boxless_results():
    assert results_to_detections(_results([])) == []
    assert results_to_detections([SimpleNamespace(boxes=None, names={})]) == []


def test_clahe_lifts_a_dark_frame_and_leaves_the_input_alone():
    rng = np.random.default_rng(0)
    dark = rng.integers(5, 40, (64, 64, 3), dtype=np.uint8)
    before = dark.copy()
    out = apply_clahe(dark)
    assert out.shape == dark.shape and out.std() > dark.std()
    assert np.array_equal(dark, before)


def test_missing_weights_fail_without_downloading(tmp_path):
    with pytest.raises(FileNotFoundError):
        UltralyticsDetector(tmp_path / "nope.pt")


def test_fast_path_imports_nothing_network_or_agent():
    """Invariant 1, statically: fast/ can't import network clients, or the agent layer."""
    banned = {"requests", "urllib", "http", "socket", "aiohttp", "httpx", "google", "anthropic",
              "openai", "ollama"}
    for path in (ROOT / "farmguard" / "fast").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                assert name.split(".")[0] not in banned, f"{path.name} imports {name}"
                assert not name.startswith("farmguard.agent"), f"{path.name} imports {name}"


@pytest.mark.skipif(not (ROOT / "weights" / "yolov8n.pt").exists(), reason="no weights on disk")
def test_real_detector_makes_no_network_attempts(monkeypatch):
    """Invariant 1, at runtime, on the real library. This caught google-analytics on 8.4.197."""
    pytest.importorskip("ultralytics")
    pytest.importorskip("lap")
    attempts = []
    real = socket.socket.connect
    monkeypatch.setattr(socket.socket, "connect",
                        lambda self, addr: (attempts.append(addr), real(self, addr))[1])
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda *a, **k: attempts.append(a[0]) or (_ for _ in ()).throw(OSError("blocked")))
    det = UltralyticsDetector(ROOT / "weights" / "yolov8n.pt", device="cpu")
    for i in range(3):
        assert isinstance(det.detect(np.zeros((320, 320, 3), np.uint8), i), list)
    import time
    time.sleep(1.5)  # ultralytics sends telemetry from a background thread
    assert attempts == []
