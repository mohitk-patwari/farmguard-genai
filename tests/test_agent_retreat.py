"""fuse() rules and the dev-only tuning helper."""
from __future__ import annotations

import pytest

from farmguard.agent.retreat import fuse, fusion_errors
from farmguard.events import RetreatStatus as R


@pytest.mark.parametrize("tracker,vlm,conf,expected", [
    (R.UNCERTAIN, R.GONE, "high", R.GONE),          # no frames for the tracker: trust a confident VLM
    (R.UNCERTAIN, R.STILL_PRESENT, "medium", R.STILL_PRESENT),
    (R.UNCERTAIN, R.GONE, "low", R.UNCERTAIN),
    (R.LEAVING, R.STILL_PRESENT, "high", R.STILL_PRESENT),  # don't record success over "still there"
    (R.LEAVING, R.STILL_PRESENT, "low", R.LEAVING),
    (R.GONE, R.LEAVING, "high", R.GONE),
    (R.LEAVING, R.GONE, "high", R.LEAVING),
    (R.GONE, R.GONE, "low", R.GONE),
    (R.GONE, R.UNCERTAIN, "high", R.GONE),
    (R.GONE, None, None, R.GONE),
])
def test_fuse_table(tracker, vlm, conf, expected):
    assert fuse(tracker, vlm, conf) == expected


def _row(split, status, tracker, vlm="", conf=""):
    return {"split": split, "status": status, "tracker_status": tracker, "vlm_status": vlm, "vlm_confidence": conf}


def test_fusion_errors_never_reads_the_test_split():
    rows = [_row("dev", "gone", "gone"), _row("test", "still_present", "gone")]
    assert fusion_errors(rows) == [{"cell": ("gone", "none", "none"), "fused": "gone", "n": 1, "wrong": 0}]


def test_fusion_errors_requires_a_split_column():
    with pytest.raises(ValueError):
        fusion_errors([{"status": "gone", "tracker_status": "gone"}])


def test_fusion_errors_counts_wrong_and_abstained_worst_first():
    rows = [_row("dev", "gone", "gone"),
            _row("dev", "still_present", "gone"),                    # fused gone: wrong
            _row("dev", "gone", "gone", "still_present", "low"),     # fused uncertain: abstained
            _row("dev", "leaving", "gone", "gone", "high")]          # gone vs leaving: both "left"
    out = fusion_errors(rows)
    assert out[0]["wrong"] == 1 and sum(c["wrong"] for c in out) == 2
    assert {c["cell"]: c["fused"] for c in out}[("gone", "still_present", "low")] == "uncertain"
