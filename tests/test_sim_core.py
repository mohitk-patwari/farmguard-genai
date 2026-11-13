"""Simulator and experiment-statistics tests. No network, no real models."""
from __future__ import annotations

import json
import random

import pytest

from farmguard.agent.llm.providers import MockClient
from farmguard.agent.policy import build_policy
from farmguard.config import load_config
from farmguard.eval import metrics as M
from farmguard.sim.simulator import (
    SimEvent,
    flee_probability,
    grid_cells,
    learning_curve,
    load_sim,
    make_events,
    make_world,
    run_all,
    run_policy,
    summarize,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture()
def sim(cfg):
    s = load_sim(cfg.root / "config" / "sim.yaml")
    s["nights"] = 15
    return s


def test_events_keep_nights_in_time_order(sim):
    """A 02:00 event belongs after the same night's 22:00, not before it."""
    sim["nights"] = 40
    evs = make_events(sim, random.Random(1))
    assert [e.night for e in evs] == sorted(e.night for e in evs)
    assert all(e.night * 24 + 19 <= e.t_hours < e.night * 24 + 31 for e in evs)


def test_outcome_noise_is_common_across_policies(cfg, sim):
    """Common random numbers: forcing every flee to be decided by the draw alone, the first attempt
    of each event must succeed or fail identically under policies that play the same first sound."""
    sim["habituation"]["strength"] = 0.0
    events = make_events(sim, random.Random(3))
    a = run_policy(cfg, sim, build_policy("fixed"), events, 4, 5, adaptive_first=False)
    b = run_policy(cfg, sim, build_policy("rule_chain"), events, 4, 5, adaptive_first=False)
    assert [o.attempts == 1 and o.success for o in a] == [o.attempts == 1 and o.success for o in b]


def test_fixed_repeats_and_others_never_repeat_within_event(cfg, sim):
    out = run_all(cfg, sim, ["fixed", "random", "rule_chain", "bandit"])
    assert any(len(set(o.sounds)) < len(o.sounds) for o in out if o.policy == "fixed")
    for o in out:
        if o.policy != "fixed":
            assert len(set(o.sounds)) == len(o.sounds)
        assert o.attempts <= sim["timing"]["max_attempts"]
        assert all(s in cfg.species[o.species].allowed_sounds for s in o.sounds)
        assert o.success != o.escalated  # every event ends one way or the other


def test_llm_failure_is_counted_not_hidden(cfg, sim):
    """LLMPolicy silently uses the bandit on failure; the summary must expose that."""
    def dead(prompt, schema, n):
        raise ConnectionError("backend down")
    summ = summarize(run_all(cfg, sim, ["llm"], MockClient(responder=dead)))["llm"]
    assert summ["llm_calls_per_event"] > 0
    assert summ["llm_fallback_rate"] == 1.0


def test_llm_enum_violation_is_an_invalid_action(cfg, sim):
    bad = MockClient(responder=lambda p, s, n: {"action": "ultrasonic_25khz", "reason": "", "confidence": 1})
    out = run_all(cfg, sim, ["llm"], bad)
    assert sum(o.invalid_actions for o in out) == sum(o.llm_calls for o in out) > 0
    assert all(s in cfg.species[o.species].allowed_sounds for o in out for s in o.sounds)


def test_llm_never_called_when_validator_would_force_escalation(cfg, sim):
    """Attempts are capped, so the final failed attempt must not spend an LLM call."""
    client = MockClient()
    out = run_all(cfg, sim, ["llm"], client)
    max_a = sim["timing"]["max_attempts"]
    assert all(o.llm_calls <= (max_a - 1) + 1 for o in out)  # next-sound calls + first-sound update


def test_sim_context_uses_the_live_agent_knobs(sim):
    """bandit_window and immediate_escalation_species reach the simulator like the AgentWorker."""
    cfg2 = load_config(overrides={"agent.policy.immediate_escalation_species": ["monkey"],
                                  "agent.policy.bandit_window": 3})
    pol, seen = build_policy("bandit_sw", seed=1), []
    orig = pol.decide
    pol.decide = lambda ctx: (seen.append(ctx), orig(ctx))[1]
    run_policy(cfg2, sim, pol, make_events(sim, random.Random(2)), 4, 5)
    assert seen and all(c.immediate_escalation_species == ("monkey",) for c in seen)
    assert all(sum(s["window_trials"] for s in c.stats.values()) <= 3 for c in seen)


def test_early_escalation_is_tracked(cfg, sim):
    quitter = MockClient(responder=lambda p, s, n: {"action": "escalate_farmer", "reason": "", "confidence": 1})
    summ = summarize(run_all(cfg, sim, ["llm"], quitter))["llm"]
    assert summ["early_escalation_rate"] > 0


def test_cross_habituation_lowers_flee_probability(cfg, sim):
    world = make_world(cfg, sim, random.Random(1))
    ev = SimEvent(0, 10.0, "wild_boar", False, 1, "morning")
    world.expose("wild_boar", "dog_bark", 10.0, 72)
    alone = flee_probability(world, sim, ev, "siren")
    sim["habituation"]["cross"] = 0.5
    assert flee_probability(world, sim, ev, "siren") < alone


def test_grid_is_full_factorial_and_does_not_touch_base(sim):
    cells = grid_cells(sim)
    assert len(cells) == 3 * 2 * 3 * 2
    levels, scenario = cells[-1]
    assert scenario["habituation"]["strength"] == levels["habituation_strength"]
    assert scenario["modifiers"]["eating_penalty"] == levels["eating_penalty"]
    assert sim["habituation"]["strength"] == 0.35


def test_llm_runs_resume_from_cache_and_key_on_scenario(cfg, sim, tmp_path):
    client = MockClient()
    first = run_all(cfg, sim, ["llm"], client, cache_dir=tmp_path)
    n = len(client.calls)
    assert run_all(cfg, sim, ["llm"], client, cache_dir=tmp_path) == first and len(client.calls) == n
    logged = next(tmp_path.rglob("*.decisions.jsonl")).read_text(encoding="utf-8").splitlines()
    assert len(logged) == sum(o.llm_calls for o in first) == n
    sim["habituation"]["strength"] = 0.6
    run_all(cfg, sim, ["llm"], client, cache_dir=tmp_path)
    assert len(client.calls) > n


def test_e4_primary_family_and_run_level_stats(cfg, sim):
    from farmguard.eval.experiments import E4_PRIMARY, e4_paired
    t = e4_paired(run_all(cfg, sim, ["fixed", "random", "bandit", "bandit_sw", "llm"], MockClient(), repeats=4))
    assert set(E4_PRIMARY) <= set(t)
    for k, v in t.items():
        assert v["holm_family"] == ("primary" if k in E4_PRIMARY else "secondary")
        assert v["sign_test_p_holm"] >= v["sign_test_p"] and v["mcnemar_p_holm"] >= v["mcnemar_p"]
        lo, hi = v["success_diff_ci95"]
        assert lo <= v["mean_success_diff"] <= hi and v["runs_won"] + v["runs_lost"] <= 4
    assert M.sign_test([1, 1, 1, 1, 1]) == 0.0625 and M.sign_test([0, 0]) == 1.0


def test_unknown_policy_fails_before_any_llm_call(cfg, sim):
    client = MockClient()
    with pytest.raises(ValueError):
        run_all(cfg, sim, ["llm", "no_such_policy"], client)
    assert client.calls == []


def test_learning_curve_covers_all_nights(cfg, sim):
    rows = learning_curve(run_all(cfg, sim, ["fixed"], repeats=2), bin_nights=5)
    assert [r["night_start"] for r in rows] == [0, 5, 10]
    assert all(r["runs"] == 2 and 0 <= r["success_rate"] <= 1 for r in rows)


def test_paired_tests_refuse_unaligned_runs(cfg, sim):
    from farmguard.eval.experiments import e4_paired
    out = run_all(cfg, sim, ["fixed", "bandit"], repeats=3)
    tests = e4_paired(out)
    assert set(tests) == {"bandit_vs_fixed"}
    assert tests["bandit_vs_fixed"]["runs"] == 3
    other = [o for o in out if o.policy == "bandit"]
    other[0].event = 999
    with pytest.raises(ValueError):
        e4_paired(out)


# Reference values come from scipy 1.14.1 (not a project dependency), computed once by hand.
SCIPY = {"mwu_u": 7.0, "mwu_p": 0.052470159539504374, "wilcoxon_p": 0.625}


def test_mann_whitney_matches_scipy():
    a, b = [1.1, 2.2, 3.3, 4.4, 5.5, 2.2], [2.2, 3.0, 6.1, 7.2, 8.3, 9.4, 10.5]
    r = M.mann_whitney(a, b)
    assert r["u"] == pytest.approx(SCIPY["mwu_u"])
    assert r["p"] == pytest.approx(SCIPY["mwu_p"], abs=1e-9)


def test_wilcoxon_matches_scipy_exact():
    no_ties = [125, 116, 130, 140, 141, 115, 140, 125, 140, 135], [110, 122, 125, 120, 140, 124, 123, 137, 135.5, 145]
    assert M.wilcoxon_signed_rank(*no_ties)["p"] == pytest.approx(SCIPY["wilcoxon_p"])
    # with tied differences scipy's exact method is only approximate; this value is from
    # scipy.stats.permutation_test on the same signed ranks
    tied = [125, 115, 130, 140, 141, 115, 140, 125, 140, 135], [110, 122, 125, 120, 140, 124, 123, 137, 135, 145]
    assert M.wilcoxon_signed_rank(*tied)["p"] == pytest.approx(0.607421875)


def test_permutation_floor_at_five_runs():
    assert M.paired_permutation([1, 1, 1, 1, 1]) == pytest.approx(2 / 32)
    assert M.paired_permutation([0, 0]) == 1.0


def test_kappa_and_holm():
    assert M.cohen_kappa(list("aabb"), list("aabb")) == 1.0
    assert M.cohen_kappa(list("aabb"), list("abab")) == pytest.approx(0.0)
    assert M.holm({"x": 0.01, "y": 0.04, "z": 0.03}) == pytest.approx({"x": 0.03, "z": 0.06, "y": 0.06})


def test_median_diff_ci_brackets_a_clear_shift():
    r = M.median_diff_ci([10.0 + i for i in range(20)], [i * 1.0 for i in range(20)])
    assert r["diff"] == 10.0 and r["ci_low"] <= 10.0 <= r["ci_high"]
    assert json.dumps(r)

