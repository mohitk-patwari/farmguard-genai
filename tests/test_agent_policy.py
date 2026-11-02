"""LLMPolicy context, decision log and invalid-action measurement."""
from __future__ import annotations

import json

import pytest

from farmguard.agent.llm.providers import MockClient
from farmguard.agent.policy import LLMPolicy, PolicyContext, decision_log_summary
from farmguard.agent.validators import ESCALATE

ALLOWED = ["a", "b", "c"]


def _ctx(tried=("a",), **kw):
    return PolicyContext("wild_boar", "wild boar", ALLOWED, "a", list(tried), len(tried), 3, "night",
                         event_id="ev1", **kw)


def _answer(action):
    return MockClient(responder=lambda p, s, n: {"action": action, "reason": "r", "confidence": 0.5})


def _policy(tmp_path, action="b"):
    return LLMPolicy(_answer(action), decision_log=tmp_path / "d.jsonl"), tmp_path / "d.jsonl"


def test_context_lists_every_sound_in_order_even_with_no_trials():
    ctx = _ctx(stats={"c": {"trials": 2, "successes": 1, "success_rate": 0.5, "recent": ["ok", "fail"]}})
    hist = LLMPolicy(MockClient()).context_json(ctx)["history"]
    assert list(hist) == ALLOWED
    assert hist["b"] == {"trials": 0, "success_rate": None, "recent": []}
    assert hist["c"]["success_rate"] == 0.5
    assert "successes" not in hist["c"]  # redundant with the rate: keep the prompt compact


def test_context_reports_cooldown_the_validator_will_enforce():
    ctx = _ctx(tried=(), last_played_at={"b": 95.0}, now=100.0, sound_cooldown_s=20.0)
    assert LLMPolicy(MockClient()).context_json(ctx)["on_cooldown"] == ["b"]


def test_prompt_matches_situation_and_is_compact_json():
    client = MockClient()
    policy = LLMPolicy(client)
    policy.decide(_ctx(tried=()))
    policy.decide(_ctx(tried=("a",)))
    first, retry = client.calls[0]["prompt"], client.calls[1]["prompt"]
    assert "next time this species enters" in first and "animal is in" not in first
    assert "just played" in retry
    assert "habituate" not in first and "decisively" not in retry
    assert '"attempt":2' in retry and "location" not in retry.lower()


def test_full_prompt_and_response_are_logged(tmp_path):
    policy, path = _policy(tmp_path)
    d = policy.decide(_ctx())
    r = json.loads(path.read_text(encoding="utf-8"))
    assert d.action == "b" and d.source == "llm"
    assert "wild boar" in r["prompt"] and json.loads(r["raw_response"])["action"] == "b"
    assert r["event_id"] == "ev1" and r["proposed"] == "b" and r["final_action"] == "b"
    assert r["model"] == "mock" and r["provider"] == "mock" and "llm_latency_s" in r


@pytest.mark.parametrize("action,out_of_enum,changed", [
    ("b", False, False),
    ("a", False, True),                  # in the enum, but already failed in this event
    ("ultrasonic_25khz", True, False),   # outside the enum: never reaches the validator or the speaker
])
def test_invalid_actions_are_measured_before_the_validator(tmp_path, action, out_of_enum, changed):
    policy, path = _policy(tmp_path, action)
    d = policy.decide(_ctx())
    r = json.loads(path.read_text(encoding="utf-8"))
    assert r["proposed"] == action and r["out_of_enum"] is out_of_enum and r["validator_changed"] is changed
    assert d.action in ALLOWED and d.action != "a"


def test_llm_failure_is_logged_and_credited_to_the_fallback(tmp_path):
    class Down(MockClient):
        def _call(self, *a):
            raise ConnectionError("offline")

    path = tmp_path / "d.jsonl"
    d = LLMPolicy(Down(), decision_log=path).decide(_ctx())
    r = json.loads(path.read_text(encoding="utf-8"))
    assert d.source == "llm_fallback_bandit" and d.action in ALLOWED
    assert r["llm_failed"] is True and r["out_of_enum"] is False


def test_no_llm_call_when_the_validator_would_force_escalation():
    client = MockClient()
    d = LLMPolicy(client).decide(_ctx(tried=("a", "b", "c")))
    assert d.action == ESCALATE and client.calls == []


def test_summary_rate_counts_answers_not_failed_calls(tmp_path):
    path = tmp_path / "d.jsonl"
    rows = [{"model": "m", "llm_failed": True, "out_of_enum": False, "validator_changed": False},
            {"model": "m", "llm_failed": False, "out_of_enum": True, "validator_changed": False},
            {"model": "m", "llm_failed": False, "out_of_enum": False, "validator_changed": True},
            {"model": "m", "llm_failed": False, "out_of_enum": False, "validator_changed": False},
            {"model": "m", "llm_failed": False, "out_of_enum": False, "validator_changed": False},
            {"model": "down", "llm_failed": True, "out_of_enum": False, "validator_changed": False}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    s = decision_log_summary(path)
    assert s["m"]["decisions"] == 5 and s["m"]["llm_failed"] == 1 and s["m"]["invalid_action_rate"] == 0.5
    assert s["down"]["invalid_action_rate"] is None


def test_unwritable_log_does_not_cost_the_decision(tmp_path):
    (tmp_path / "file").write_text("x")
    d = LLMPolicy(_answer("b"), decision_log=tmp_path / "file" / "d.jsonl").decide(_ctx())
    assert d.action == "b"
