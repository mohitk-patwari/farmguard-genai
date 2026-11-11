"""Constrained enum, early-escalation rule, VLM species override, bandit_sw, Ollama seed."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from farmguard.agent.llm.providers import MockClient, build_client
from farmguard.agent.memory import Memory
from farmguard.agent.notifier import Notifier
from farmguard.agent.orchestrator import AgentWorker
from farmguard.agent.policy import LLMPolicy, PolicyContext, RuleChainPolicy, build_policy
from farmguard.agent.retreat import RetreatResult
from farmguard.agent.validators import ESCALATE, validate
from farmguard.config import load_config
from farmguard.events import IntrusionEvent, RetreatStatus, VLMReport

ALLOWED = ["a", "b", "c"]


def _ctx(species="wild_boar", tried=("a",), **kw):
    return PolicyContext(species, species, ALLOWED, "a", list(tried), len(tried), 3, "night",
                         event_id="ev1", **kw)


def _always(action):
    """Ignores the schema, like a model the provider doesn't constrain."""
    return MockClient(responder=lambda p, s, n: {"action": action, "reason": "r", "confidence": 0.5})


def _enum_sent(constrain):
    enums = []
    client = MockClient(responder=lambda p, s, n: (enums.append(s["properties"]["action"]["enum"]),
                                                   {"action": "c", "reason": "r", "confidence": 0.5})[1])
    LLMPolicy(client, constrain_enum=constrain).decide(_ctx(tried=("a",), last_played_at={"b": 95.0}, now=100.0))
    return enums[0]


# ---- 1. constrained enum ---------------------------------------------------------------------
def test_constrained_enum_is_legal_actions_only():
    assert _enum_sent(True) == ["c"]  # a tried, b on cooldown, escalation blocked at attempt 1


def test_unconstrained_enum_is_the_old_one():
    assert _enum_sent(False) == ALLOWED + [ESCALATE]


def test_constrained_never_lets_a_tried_sound_reach_the_validator(tmp_path, monkeypatch):
    import farmguard.agent.policy as pol
    seen = []
    monkeypatch.setattr(pol, "validate", lambda a, v: (seen.append(a), validate(a, v))[1])
    path = tmp_path / "d.jsonl"
    d = LLMPolicy(_always("a"), decision_log=path).decide(_ctx(tried=("a",)))
    r = json.loads(path.read_text(encoding="utf-8"))
    assert "a" not in seen and d.action in ("b", "c")
    assert r["out_of_enum"] is True and r["constrain_enum"] is True and r["enum"] == ["b", "c"]


def test_no_legal_sound_means_escalate_without_a_call():
    client = MockClient()
    d = LLMPolicy(client).decide(_ctx(tried=("a",), last_played_at={"b": 99.0, "c": 99.0}, now=100.0))
    assert d.action == ESCALATE and client.calls == []


def test_mode_seed_and_model_in_every_decision_record(tmp_path):
    path = tmp_path / "d.jsonl"
    for constrain in (True, False):
        LLMPolicy(_always("b"), decision_log=path, constrain_enum=constrain).decide(_ctx())
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["constrain_enum"] for r in rows] == [True, False]
    assert all("seed" in r and r["model"] == "mock" for r in rows)


def test_build_policy_reads_the_flag():
    cfg = load_config(overrides={"agent.policy.constrain_enum": False})
    assert build_policy("llm", MockClient(), cfg=cfg).constrain_enum is False
    assert build_policy("llm", MockClient()).constrain_enum is True


# ---- 3. early escalation ---------------------------------------------------------------------
def test_escalation_blocked_at_attempt_1_counts_as_invalid():
    final, warnings = validate(ESCALATE, _ctx(tried=("a",)).validation())
    assert final == "b" and warnings


def test_escalation_allowed_for_elephant_at_attempt_1():
    assert validate(ESCALATE, _ctx("elephant", tried=("a",)).validation()) == (ESCALATE, [])


def test_escalation_allowed_from_attempt_2_or_when_no_sound_is_left():
    assert validate(ESCALATE, _ctx(tried=("a", "b")).validation()) == (ESCALATE, [])
    ctx = _ctx(tried=("a",), last_played_at={"b": 99.0, "c": 99.0}, now=100.0)
    assert validate(ESCALATE, ctx.validation()) == (ESCALATE, [])


def test_rule_chain_still_escalates_when_exhausted():
    assert RuleChainPolicy().decide(_ctx(tried=("a", "b", "c"))).action == ESCALATE


# ---- 4. sliding-window bandit ----------------------------------------------------------------
def test_bandit_sw_forgets_outcomes_older_than_the_window():
    mem = Memory(":memory:")
    for i in range(30):  # old: a always works, b always fails
        mem.add_attempt(f"o{i}", 1, "wild_boar", "a" if i % 2 else "b", "x", "", "gone" if i % 2 else "still_present",
                        None, bucket="night")
    for i in range(10):  # newest 10: a always fails now (habituated)
        mem.add_attempt(f"n{i}", 1, "wild_boar", "a", "x", "", "still_present", None, bucket="night")
    stats = mem.sound_stats("wild_boar", "night", window=10)
    assert stats["a"]["window_trials"] == 10 and stats["a"]["window_successes"] == 0
    assert stats["b"]["window_trials"] == 0 and stats["a"]["successes"] == 15
    picks = [build_policy("bandit_sw", seed=s).decide(_ctx(tried=(), stats=stats)).action for s in range(30)]
    assert picks.count("a") <= 3  # 0/10 recent; the full-history bandit sees 15/25
    full = [build_policy("bandit", seed=s).decide(_ctx(tried=(), stats=stats)).action for s in range(30)]
    assert full.count("a") > picks.count("a")


def test_bandit_sw_never_proposes_a_tried_sound():
    for s in range(20):
        assert build_policy("bandit_sw", seed=s).propose(_ctx(tried=("a", "b")))[0] == "c"


# ---- 5. seed ---------------------------------------------------------------------------------
def test_seed_is_in_the_ollama_request_body(monkeypatch):
    import farmguard.agent.llm.providers as prov
    bodies = []
    monkeypatch.setattr(prov, "_post_json", lambda url, body, timeout, headers=None:
                        (bodies.append(body), {"message": {"content": "{}"}})[1])
    client = build_client(load_config(overrides={"agent.provider": "ollama"}), "llm")
    client.generate_json("p", {"type": "object"})
    assert bodies[0]["options"]["seed"] == 42 and client.seed == 42


# ---- 2. VLM species override -----------------------------------------------------------------
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


def _event(species="stray_cattle"):
    return IntrusionEvent("e1", species, species, 1, 0.0, 0, (0, 0, 10, 10), 0.9, "field",
                          "human_shout", 0.0, 0.0)


def _report(species, confidence):
    return VLMReport(species, 2, "eating", confidence, "obs", "note")


STILL = (RetreatStatus.STILL_PRESENT, None, RetreatStatus.STILL_PRESENT, None)
GONE = (RetreatStatus.GONE, 1.0, RetreatStatus.GONE, None)


def test_high_confidence_vlm_species_overrides_the_detector(tmp_path):
    w = _Worker(_cfg(tmp_path), _report("wild_boar", "high"), [STILL, GONE])
    res = w.handle(_event())
    assert res.overridden and res.effective_species == "wild_boar" and res.detector_species == "stray_cattle"
    assert res.attempts[1].sound in w.cfg.species["wild_boar"].allowed_sounds
    assert "wild boar" in res.farmer["message"]
    assert set(w.memory.sound_stats("wild_boar")) == {"human_shout", res.attempts[1].sound}
    assert w.memory.sound_stats("stray_cattle") == {}
    row = json.loads((tmp_path / "ev.jsonl").read_text(encoding="utf-8"))
    assert row["vlm_species"] == "wild_boar" and row["vlm_confidence"] == "high" and row["overridden"] is True
    w.results.append(res)
    assert w.vlm_counts()["species_overridden"] == 1


@pytest.mark.parametrize("species,conf", [("wild_boar", "medium"), ("wild_boar", "low"),
                                          ("unclassifiable", "high"), ("other_animal", "high")])
def test_doubtful_vlm_keeps_the_detector_species(tmp_path, species, conf):
    res = _Worker(_cfg(tmp_path), _report(species, conf), [GONE]).handle(_event())
    assert not res.overridden and res.effective_species == "stray_cattle"


def test_override_can_be_switched_off(tmp_path):
    cfg = _cfg(tmp_path, **{"agent.vlm.override_species": False})
    assert _Worker(cfg, _report("wild_boar", "high"), [GONE]).handle(_event()).effective_species == "stray_cattle"


def test_failed_vlm_call_keeps_the_detector_species(tmp_path):
    report = VLMReport("wild_boar", 1, "unclear", "high", "", "", error="timeout")
    assert not _Worker(_cfg(tmp_path), report, [GONE]).handle(_event()).overridden


def test_possible_false_alarm_stops_and_sends_a_low_priority_note(tmp_path):
    uncertain = (RetreatStatus.UNCERTAIN, None, RetreatStatus.UNCERTAIN, None)
    w = _Worker(_cfg(tmp_path), _report("none", "high"), [uncertain])
    res = w.handle(_event())
    assert res.possible_false_alarm and len(res.attempts) == 1 and not res.escalated
    assert res.farmer["source"] == "false_alarm_note" and "Low priority" in res.farmer["message"]
    assert w.memory.sound_stats("stray_cattle") == {}  # no outcome learned from an empty field


def test_false_alarm_keeps_deterring_if_the_retreat_check_sees_an_animal(tmp_path):
    w = _Worker(_cfg(tmp_path), _report("none", "high"), [STILL, GONE])
    res = w.handle(_event())
    assert res.possible_false_alarm and len(res.attempts) == 2 and res.farmer["source"] != "false_alarm_note"
