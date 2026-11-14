"""Option shuffle (position bias), llm_bandit, truncated retreat windows, elephant disagreement,
stem-based alert grounding, and the alert provider fallback. Mocks only."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from farmguard.agent.llm.base import LLMResult
from farmguard.agent.llm.providers import MockClient, build_optional_client
from farmguard.agent.memory import Memory
from farmguard.agent.notifier import Notifier
from farmguard.agent.orchestrator import AgentWorker
from farmguard.agent.policy import (
    LLMBanditPolicy,
    LLMPolicy,
    Policy,
    PolicyContext,
    build_policy,
    decision_log_summary,
)
from farmguard.agent.retreat import RetreatChecker, RetreatResult
from farmguard.agent.validators import ESCALATE
from farmguard.config import load_config
from farmguard.events import FramePacket, IntrusionEvent, RetreatStatus, VLMReport
from farmguard.sim.simulator import SimOutcome, summarize

SIX = ["s1", "s2", "s3", "s4", "s5", "s6"]


def _ctx(event_id="ev1", tried=(), allowed=SIX, **kw):
    return PolicyContext("wild_boar", "wild boar", list(allowed), allowed[0], list(tried), len(tried), 3, "night",
                         event_id=event_id, **kw)


def _enum(policy, ctx):
    seen = []
    policy.client = MockClient(responder=lambda p, s, n: (seen.append(s["properties"]["action"]["enum"]),
                                                          {"action": s["properties"]["action"]["enum"][0],
                                                           "reason": "r", "confidence": 0.5})[1])
    policy.decide(ctx)
    return seen[0]


# ---- A. shuffle ----------------------------------------------------------------------------------
def test_shuffle_is_deterministic_per_seed_event_and_attempt():
    a = _enum(LLMPolicy(MockClient(), seed=42), _ctx("ev1", tried=("s1",)))
    assert a == _enum(LLMPolicy(MockClient(), seed=42), _ctx("ev1", tried=("s1",)))
    others = [_enum(LLMPolicy(MockClient(), seed=s), _ctx(e, tried=("s1",))) for s, e in [(7, "ev1"), (42, "ev2")]]
    assert any(o != a for o in others) and all(sorted(o) == sorted(a) for o in others)
    assert "s1" not in a and len(a) == 5  # still the legal set: tried sound out, no escalation at attempt 1


def test_shuffle_off_gives_the_old_order():
    assert _enum(LLMPolicy(MockClient(), shuffle_options=False), _ctx(tried=("s1",))) == SIX[1:]


def test_prompt_stats_follow_the_shown_order():
    policy = LLMPolicy(MockClient(), seed=3)
    ctx = _ctx("ev9", tried=("s1",))
    hist = list(policy.context_json(ctx)["history"])
    assert [s for s in hist if s != "s1"] == policy.enum(ctx) and sorted(hist) == SIX


def test_shown_order_and_position_are_logged(tmp_path):
    path = tmp_path / "d.jsonl"
    policy = LLMPolicy(MockClient(), decision_log=path)
    shown = _enum(policy, _ctx(tried=("s1",)))  # the responder picks the first shown option
    r = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert r["shown_order"] == shown and r["chosen_position"] == 0 and r["shuffle_options"] is True
    assert decision_log_summary(path)["mock"]["first_position_rate"] == 1.0


def test_first_position_rate_in_the_simulator_summary():
    def o(first, answers):
        return SimOutcome("llm", 0, "wild_boar", ["s1"], True, 1, False, 1.0, first_position=first,
                          position_answers=answers)
    assert summarize([o(3, 4), o(0, 4)])["llm"]["first_position_rate"] == 3 / 8
    assert summarize([SimOutcome("bandit", 0, "wild_boar", ["s1"], True, 1, False, 1.0)])["bandit"][
        "first_position_rate"] is None


def test_build_policy_reads_shuffle_and_seed():
    cfg = load_config(overrides={"agent.policy.shuffle_options": False, "agent.seed": 7})
    p = build_policy("llm", MockClient(), cfg=cfg)
    assert p.shuffle_options is False and p.seed == 7
    assert build_policy("llm", MockClient()).shuffle_options is True


# ---- B. llm_bandit -------------------------------------------------------------------------------
def test_llm_bandit_adds_posterior_means_for_legal_sounds():
    p = build_policy("llm_bandit", MockClient())
    assert isinstance(p, LLMBanditPolicy) and p.name == "llm_bandit"
    ctx = _ctx(tried=("s1",), stats={"s2": {"trials": 3, "successes": 2}, "s1": {"trials": 4, "successes": 4}})
    hist = p.context_json(ctx)["history"]
    assert hist["s2"]["posterior_mean"] == 0.6 and hist["s3"]["posterior_mean"] == 0.5
    assert "posterior_mean" not in hist["s1"]  # tried this event: not legal
    client = MockClient()
    p.client = client
    d = p.decide(ctx)
    assert "posterior_mean" in client.calls[0]["prompt"] and d.action in SIX[1:] and d.source == "llm_bandit"


def test_simulator_runs_llm_bandit_by_name():
    from farmguard.sim.simulator import load_sim, run_all
    cfg = load_config()
    sim = load_sim(cfg.root / "config" / "sim.yaml")
    sim["nights"] = 2
    out = summarize(run_all(cfg, sim, ["llm_bandit"], MockClient()))["llm_bandit"]
    assert out["llm_calls_per_event"] > 0 and out["first_position_rate"] is not None


# ---- C. truncated windows ------------------------------------------------------------------------
def _buffer(n, finished):
    pkts = [FramePacket(i, float(i), np.zeros((4, 4, 3), np.uint8)) for i in range(n)]
    return SimpleNamespace(wait_until=lambda t, timeout_s: False, window=lambda a, b: pkts, finished=finished)


def test_short_window_after_the_source_ended_is_truncated():
    client = MockClient()
    r = RetreatChecker(_buffer(3, True), client, min_frames=10).check("wild boar", 0.0)
    assert r.status == RetreatStatus.TRUNCATED and r.frames_seen == 3 and client.calls == []
    assert RetreatChecker(_buffer(3, False), None, min_frames=10).check("wild boar", 0.0).status != RetreatStatus.TRUNCATED


class _Worker(AgentWorker):
    def __init__(self, cfg, report, statuses, policy=None):
        retreat = SimpleNamespace(check=lambda label, t: RetreatResult(*statuses.pop(0), None, 1))
        super().__init__(cfg, SimpleNamespace(latest_time=0.0), SimpleNamespace(play=lambda *a, **k: None),
                         Memory(":memory:"), None, retreat, policy or build_policy("rule_chain"),
                         Notifier(cfg, None))
        self._report = report

    def _vlm_task(self, ev):
        return self._report


def _cfg(tmp_path, **ov):
    return load_config(overrides={"logging.events_jsonl": str(tmp_path / "ev.jsonl"), "notifier.language": "en",
                                  "notifier.channels": [], **ov})


def _event(species):
    return IntrusionEvent("e1", species, species, 1, 0.0, 0, (0, 0, 10, 10), 0.9, "field",
                          load_config().species[species].default_sound, 0.0, 0.0)


STILL = (RetreatStatus.STILL_PRESENT, None, RetreatStatus.STILL_PRESENT, None)
TRUNC = (RetreatStatus.TRUNCATED, None, RetreatStatus.TRUNCATED, None)
GONE = (RetreatStatus.GONE, 1.0, RetreatStatus.GONE, None)


def test_truncated_window_writes_no_memory_and_is_counted(tmp_path):
    w = _Worker(_cfg(tmp_path), VLMReport("wild_boar", 1, "eating", "high", "", ""), [STILL, TRUNC])
    res = w.handle(_event("wild_boar"))
    w.results.append(res)
    assert [a.status for a in res.attempts] == ["still_present", "truncated"] and res.final_status == "truncated"
    stats = w.memory.sound_stats("wild_boar")
    assert list(stats) == [res.attempts[0].sound] and stats[res.attempts[0].sound]["trials"] == 1
    assert w.run_counts()["truncated_windows"] == 1
    row = json.loads((tmp_path / "ev.jsonl").read_text(encoding="utf-8"))
    assert row["attempts"][1]["status"] == "truncated"


# ---- D. elephant disagreement --------------------------------------------------------------------
class _AlwaysEscalate(Policy):
    name = "escalate_test"

    def propose(self, ctx):
        return ESCALATE, "test", 1.0


@pytest.mark.parametrize("conf", ["low", "medium"])
def test_disputed_elephant_cannot_escalate_on_the_detector_label(tmp_path, conf):
    w = _Worker(_cfg(tmp_path), VLMReport("wild_boar", 1, "eating", conf, "", ""), [STILL, GONE],
                policy=_AlwaysEscalate())
    res = w.handle(_event("elephant"))
    assert res.species_uncertain and res.effective_species == "elephant"
    assert not res.escalated and len(res.attempts) == 2  # the escalation became a sound at attempt 2
    assert "Species uncertain (detector: elephant, camera check: wild boar)" in res.farmer["message"]


def test_agreed_elephant_still_escalates_at_once(tmp_path):
    w = _Worker(_cfg(tmp_path), VLMReport("elephant", 1, "eating", "medium", "", ""), [STILL],
                policy=_AlwaysEscalate())
    res = w.handle(_event("elephant"))
    assert res.escalated and len(res.attempts) == 1 and not res.species_uncertain
    assert "uncertain" not in res.farmer["message"]


def test_vlm_elephant_against_detector_boar_is_also_flagged(tmp_path):
    res = _Worker(_cfg(tmp_path), VLMReport("elephant", 1, "eating", "low", "", ""), [GONE]).handle(_event("wild_boar"))
    assert res.species_uncertain and "detector: wild boar, camera check: elephant" in res.farmer["message"]


def test_effective_species_is_logged_at_each_threshold(tmp_path):
    w = _Worker(_cfg(tmp_path), VLMReport("wild_boar", 1, "eating", "medium", "", ""), [GONE])
    res = w.handle(_event("stray_cattle"))
    assert res.effective_by_threshold == {"none": "stray_cattle", "high": "stray_cattle", "medium": "wild_boar"}
    assert res.effective_species == "stray_cattle"  # default threshold is high
    row = json.loads((tmp_path / "ev.jsonl").read_text(encoding="utf-8"))
    assert row["effective_by_threshold"]["medium"] == "wild_boar"
    medium = _Worker(_cfg(tmp_path, **{"agent.vlm.override_min_confidence": "medium"}),
                     VLMReport("wild_boar", 1, "eating", "medium", "", ""), [GONE]).handle(_event("stray_cattle"))
    assert medium.overridden and medium.effective_species == "wild_boar"


# ---- E. grounding on the 6 real smoke-test-2 alerts ------------------------------------------------
# (id, species, sounds, time, LLM text, expected source). A3/B3 never name the boar alarm ("उसकी आवाज़" =
# "its sound"), so they stay rejected; the others only reword a sound name.
SMOKE2 = [
    ("A1", "stray_cattle", ["human_shout", "dog_bark"], "23:05",
     "आवारा पशु आया था। आदमी ने चिल्लाने की आवाज़ और कुत्ते की आवाज़ खेत से जा रहा था। यह खेत से जा रहा था।", "llm"),
    ("A2", "elephant", ["bee_swarm", "tiger_roar"], "23:05",
     ("एक हाथी आया था और आवाज़ मधुमक्खियों और बाघ की आवाज़ सुनी. खेत से जा रहा था। यदि यह नहीं जा रहा है, तो फ़रमान "
     "की जाना चाहिए।"), "llm"),
    ("A3", "wild_boar", ["boar_alarm", "gunshot"], "23:05",
     ("जंगली सूअर आया था और उसकी आवाज़ और पटाखे की आवाज़ बोली. यह खेत से जा रहा था। यदि यह खेत से नहीं निकल गया तो "
     "फ़रमान करें।"), "template_ungrounded"),
    ("B1", "stray_cattle", ["human_shout", "dog_bark"], "23:08",
     ("आवारा पशु 23:08 को आया था। आदमी के चिल्लाने की आवाज़ और कुत्ते की आवाज़ सुनी। खेत से जा रहा था। यदि यह नहीं जा "
     "रहा है, तो फ़रमान की जाने लायक है।"), "llm"),
    ("B2", "elephant", ["bee_swarm", "tiger_roar"], "23:08",
     ("एक हाथी आया था और आवाज़ मधुमक्खियों और बाघ की आवाज़ सुनी. यह खेत से जा रहा था। यदि यह खेत से नहीं निकल गया तो "
     "फ़रमान करें।"), "llm"),
    ("B3", "wild_boar", ["boar_alarm", "gunshot"], "23:08",
     ("जंगली सूअर आया था और उसकी आवाज़ और पटाखे की आवाज़ बोली. यह खेत से जा रहा था। यदि यह खेत से नहीं निकल गया तो "
     "फ़रमान करें।"), "template_ungrounded"),
]


def _compose(text, species, sounds, time, status="leaving"):
    cfg = load_config(overrides={"notifier.language": "hi", "notifier.channels": []})
    nt = Notifier(cfg, MockClient(responder=lambda p, s, n: {"message": text, "voice_text": text}))
    summary = nt.build_summary(species, 0, sounds, status, len(sounds), 1, "moving_through")
    summary["time"] = time
    return nt.compose(summary)[2]


@pytest.mark.parametrize("case", SMOKE2, ids=[c[0] for c in SMOKE2])
def test_smoke_test_2_alerts(case):
    _id, species, sounds, time, text, expected = case
    assert _compose(text, species, sounds, time) == expected


@pytest.mark.parametrize("text", [
    SMOKE2[0][4].replace("आवारा पशु", "हाथी"),                    # wrong species
    SMOKE2[0][4] + " ढोल भी बजा।",                               # a sound that was not played
    SMOKE2[0][4].replace("कुत्ते की आवाज़", "सायरन"),               # wrong sound instead of a played one
    SMOKE2[0][4].replace("जा रहा था", "अभी भी खेत में है"),          # wrong outcome
    SMOKE2[0][4] + " dog",                                       # Latin script
])
def test_paraphrase_tolerance_still_rejects_wrong_facts(text):
    assert _compose(text, "stray_cattle", ["human_shout", "dog_bark"], "23:05") == "template_ungrounded"


def test_nilgai_is_not_read_as_cattle():
    text = "23:05 बजे नीलगाय आई। ढोल बजाया। नीलगाय खेत से चला गया।"
    assert _compose(text, "nilgai", ["drum_beat"], "23:05", status="gone") == "llm"


# ---- F. alert provider ---------------------------------------------------------------------------
class _Down(MockClient):
    name = "gemini"

    def generate_json(self, *a, **k):
        return LLMResult(None, "", 0.1, error="HTTP 503", model=self.model)


def test_failed_alert_provider_falls_back_to_the_agent_model_then_template():
    cfg = load_config(overrides={"notifier.language": "hi", "notifier.channels": []})
    good = SMOKE2[3][4]
    nt = Notifier(cfg, _Down(), fallback=MockClient(responder=lambda p, s, n: {"message": good, "voice_text": good}))
    summary = nt.build_summary("stray_cattle", 0, ["human_shout", "dog_bark"], "leaving", 2, 1, "eating")
    summary["time"] = "23:08"
    assert nt.compose(summary) == (good, good, "llm")
    nt.fallback = _Down()
    assert nt.compose(summary)[2] == "template"


def test_alert_provider_without_key_is_skipped(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert build_optional_client(load_config(), "gemini") is None


def test_system_wires_the_alert_provider(monkeypatch, tmp_path):
    from farmguard.fast.detector import MockDetector
    from farmguard.system import build_system
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    cfg = load_config(overrides={"agent.provider": "mock", "agent.notifier.provider": "gemini", "alarm.backend": "none",
                                 "memory.db_path": str(tmp_path / "m.sqlite"), "logging.timing_csv": str(tmp_path / "t.csv"),
                                 "logging.events_jsonl": None, "zones.file": str(tmp_path / "none.json")})
    system = build_system(cfg, detector=MockDetector({}))
    try:  # no key: alerts stay on the agent's own (mock) client
        assert system.agent.notifier.client.name == "mock" and system.agent.notifier.fallback is None
    finally:
        system.agent.stop(timeout=5)
