"""Unit tests for the pure logic. Run: pytest -q"""
from __future__ import annotations

import numpy as np
import pytest

from farmguard.agent.llm.base import extract_json
from farmguard.agent.llm.providers import MockClient
from farmguard.agent.memory import Memory
from farmguard.agent.policy import BanditPolicy, FixedPolicy, LLMPolicy, PolicyContext, RuleChainPolicy
from farmguard.agent.retreat import fuse, tracker_evidence
from farmguard.agent.validators import ESCALATE, ValidationContext, validate
from farmguard.config import load_config
from farmguard.events import Detection, FramePacket, RetreatStatus
from farmguard.fast.trigger import Trigger
from farmguard.fast.zones import Zone, ZoneSet


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_config_species(cfg):
    assert cfg.species_for_label("wild boar") == "wild_boar"
    assert cfg.species_for_label("a wild boar") == "wild_boar"
    assert cfg.species_for_label("cow") == "stray_cattle"
    assert cfg.species_for_label("person") is None
    for info in cfg.species.values():
        assert info.default_sound in info.allowed_sounds


def test_zone_uses_feet_point():
    zs = ZoneSet([Zone("field", [(0, 100), (200, 100), (200, 200), (0, 200)])])
    head_in_feet_out = Detection((10, 150, 50, 250), 0.9, "wild boar", "wild_boar", 1)
    feet_in = Detection((10, 60, 50, 150), 0.9, "wild boar", "wild_boar", 1)
    assert zs.zone_of(head_in_feet_out) is None
    assert zs.zone_of(feet_in) == "field"


def test_zone_rescale():
    zs = ZoneSet([Zone("f", [(0, 0), (100, 0), (100, 100)])], frame_size=(100, 100))
    assert zs.rescaled(200, 200).zones[0].points[1] == (200, 0)


def test_trigger_vote_and_cooldown():
    tr = Trigger(window=7, min_hits=4, track_cooldown_s=60)
    d = Detection((0, 0, 10, 10), 0.9, "wild boar", "wild_boar", 5)
    fired = [bool(tr.update([(d, "field")], {"track:5"}, t)) for t in range(6)]
    assert fired == [False, False, False, True, False, False]   # fires on the 4th hit only
    assert tr.update([(d, "field")], {"track:5"}, 70.0)           # after cooldown


def test_trigger_flicker_does_not_fire():
    tr = Trigger(window=7, min_hits=4)
    d = Detection((0, 0, 10, 10), 0.9, "wild boar", "wild_boar", 1)
    hits = []
    for t in range(10):
        dets = [(d, "field")] if t % 3 == 0 else []
        hits += tr.update(dets, {"track:1"}, t)
    assert hits == []


def _pkts(flags, species="wild_boar"):
    out = []
    for i, f in enumerate(flags):
        dets = [Detection((0, 0, 10, 10), 0.9, "wild boar", species, 1)] if f else []
        out.append(FramePacket(i, float(i), np.zeros((2, 2, 3), np.uint8), dets, list(dets)))
    return out


def test_tracker_evidence():
    st, ttl = tracker_evidence(_pkts([1, 1, 1, 0, 0, 0, 0, 0]), 0.0, gone_frames=5)
    assert st == RetreatStatus.GONE and ttl == 3.0
    st, _ = tracker_evidence(_pkts([1] * 8), 0.0, gone_frames=5)
    assert st == RetreatStatus.STILL_PRESENT


def test_tracker_evidence_ignores_out_of_zone_detections():
    """T1's question: an untracked box outside the field must not read as "still present"."""
    outside = [Detection((0, 0, 10, 10), 0.9, "wild boar", "wild_boar", None)]
    pkts = [FramePacket(i, float(i), np.zeros((2, 2, 3), np.uint8), outside, []) for i in range(8)]
    assert tracker_evidence(pkts, 0.0, gone_frames=5)[0] == RetreatStatus.GONE
    inside = [FramePacket(i, float(i), np.zeros((2, 2, 3), np.uint8), outside, outside) for i in range(8)]
    assert tracker_evidence(inside, 0.0, gone_frames=5)[0] == RetreatStatus.STILL_PRESENT


def test_tracker_evidence_counts_any_target_animal_in_the_zone():
    """Detector labels are unreliable (T1's boars came out as cattle): any target animal counts."""
    assert tracker_evidence(_pkts([1] * 8, species="stray_cattle"), 0.0, gone_frames=5)[0] == RetreatStatus.STILL_PRESENT
    person = [Detection((0, 0, 10, 10), 0.9, "person", None, 1)]  # not a target species
    pkts = [FramePacket(i, float(i), np.zeros((2, 2, 3), np.uint8), person, person) for i in range(8)]
    assert tracker_evidence(pkts, 0.0, gone_frames=5)[0] == RetreatStatus.GONE


def test_trigger_species_cooldown_stops_fragmented_tracks():
    tr = Trigger(window=7, min_hits=4, track_cooldown_s=60, species_cooldown_s=60)
    fired = []
    for t in range(40):  # a new track id every 5 frames, as on the smoke-test clip
        d = Detection((0, 0, 10, 10), 0.9, "wild boar", "wild_boar", t // 5)
        fired += [(t, h.detection.track_id) for h in tr.update([(d, "field")], {f"track:{t // 5}"}, float(t))]
    assert fired == [(3, 0)]
    d = Detection((0, 0, 10, 10), 0.9, "wild boar", "wild_boar", 99)
    late = [tr.update([(d, "field")], {"track:99"}, float(t)) for t in range(60, 64)]
    assert [bool(x) for x in late] == [False, False, False, True]  # cooldown over: a new event


def test_trigger_species_cooldown_is_per_species():
    tr = Trigger(window=3, min_hits=1, species_cooldown_s=60)
    boar = Detection((0, 0, 10, 10), 0.9, "wild boar", "wild_boar", 1)
    boar2 = Detection((0, 0, 10, 10), 0.9, "wild boar", "wild_boar", 2)
    monkey = Detection((0, 0, 10, 10), 0.9, "monkey", "monkey", 3)
    hits = tr.update([(boar, "f"), (boar2, "f"), (monkey, "f")], set(), 0.0)
    assert [h.detection.track_id for h in hits] == [1, 3]  # two boars in one frame: one event


def test_trigger_cooldown_wait_is_not_counted_as_vote_delay():
    tr = Trigger(window=7, min_hits=2, track_cooldown_s=0, species_cooldown_s=10)
    a = Detection((0, 0, 10, 10), 0.9, "wild boar", "wild_boar", 1)
    b = Detection((0, 0, 10, 10), 0.9, "wild boar", "wild_boar", 2)
    tr.update([(a, "f")], set(), 0.0, stamp=0.0)
    tr.update([(a, "f")], set(), 1.0, stamp=1.0)  # a fires; b waits out the species cooldown below
    hits = []
    for t in range(2, 13):
        hits += tr.update([(b, "f")], set(), float(t), stamp=float(t))
    (hit,) = hits
    assert hit.detection.track_id == 2 and hit.first_stamp >= 10.0


def test_fuse_prefers_vlm_when_tracker_loses_animal():
    assert fuse(RetreatStatus.GONE, RetreatStatus.STILL_PRESENT, "high") == RetreatStatus.STILL_PRESENT
    assert fuse(RetreatStatus.GONE, RetreatStatus.STILL_PRESENT, "low") == RetreatStatus.UNCERTAIN
    assert fuse(RetreatStatus.STILL_PRESENT, None, None) == RetreatStatus.STILL_PRESENT
    assert fuse(RetreatStatus.STILL_PRESENT, RetreatStatus.GONE, "high") == RetreatStatus.LEAVING


def test_validator_blocks_invented_and_repeated_actions():
    ctx = ValidationContext(["a", "b", "c"], ["a"], attempt_no=1, max_attempts=3)
    assert validate("ultrasonic_22khz", ctx)[0] == "b"
    assert validate("a", ctx)[0] == "b"
    assert validate("c", ctx) == ("c", [])
    assert validate("b", ValidationContext(["a", "b"], ["a"], 3, 3))[0] == ESCALATE


def _ctx(cfg, tried, stats=None):
    info = cfg.species["wild_boar"]
    return PolicyContext("wild_boar", "wild boar", info.allowed_sounds, info.default_sound,
                         tried, len(tried), 3, "night", stats=stats or {})


def test_baseline_policies(cfg):
    assert FixedPolicy().decide(_ctx(cfg, ["boar_alarm"])).action == "boar_alarm"
    rc = RuleChainPolicy().decide(_ctx(cfg, ["boar_alarm"]))
    assert rc.action == cfg.species["wild_boar"].allowed_sounds[1]
    stats = {"dog_bark": {"trials": 50, "successes": 49}, "tiger_roar": {"trials": 50, "successes": 1}}
    picks = [BanditPolicy(seed=s).decide(_ctx(cfg, ["boar_alarm"], stats)).action for s in range(20)]
    assert picks.count("dog_bark") >= 15


def test_llm_policy_uses_enum_and_falls_back(cfg):
    good = MockClient(responder=lambda p, s, n: {"action": "gunshot", "reason": "r", "confidence": 0.7})
    d = LLMPolicy(good).decide(_ctx(cfg, ["boar_alarm"]))
    assert d.action == "gunshot" and d.source == "llm"
    bad = MockClient(responder=lambda p, s, n: {"action": "ultrasonic_25khz", "reason": "", "confidence": 1})
    d = LLMPolicy(bad).decide(_ctx(cfg, ["boar_alarm"]))
    assert d.action in cfg.species["wild_boar"].allowed_sounds and "fallback" in d.reason
    assert "location" not in good.calls[0]["prompt"].lower()


def test_memory_stats():
    m = Memory(":memory:")
    for ok in (1, 1, 0):
        m.add_attempt("e", 1, "monkey", "gunshot", "llm", "", "gone" if ok else "still_present", 5.0,
                      bucket="night")
    s = m.sound_stats("monkey", "night")["gunshot"]
    assert s["trials"] == 3 and s["successes"] == 2 and s["recent"][0] == "fail"


def test_extract_json():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": 2} hope this helps') == {"a": 2}
    assert extract_json("no json") is None
