"""One command per experiment: `python -m farmguard.eval <exp> ...` from the repo root.

  e1        latency, agent on vs off      --clips data/clips/*.mp4 --provider ollama --runs 3
  e2        VLM species + behaviour       --provider ollama --split test
  e3        retreat check                 --provider ollama --split test [--with-detector --zones z.json]
  e4        simulator, one or both modes  --policies fixed,random,rule_chain,bandit,llm --repeats 5
  e4-grid   sensitivity grid              --policies fixed,random,rule_chain,bandit --repeats 30
  e5        false alarms                  --clips data/clips/*.mp4 --provider ollama --split test
  e6        blind rating sheet + key      --provider ollama
  e6-score  analyse filled rating sheets  --ratings results/ratings/*.csv

Every non-mock run is appended to paper/RESULTS.md. Mock runs only check the pipeline: they are
never logged and scripts/figures_all.py refuses them.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import shlex
import sys
from datetime import date
from pathlib import Path
from typing import Any

from farmguard.config import load_config
from farmguard.eval import experiments as E

RESULTS_LOG = Path("paper/RESULTS.md")


def _headline(exp: str, out: dict[str, Any]) -> str:
    """The numbers that go in the log, short enough for a table cell."""
    if exp == "e1":
        d = out["read_to_start_ms"]
        return (f"read->start ms on {d['agent_on'].get('median', float('nan')):.1f} / "
                f"off {d['agent_off'].get('median', float('nan')):.1f} (median); "
                f"MWU p={d['mann_whitney']['p']:.3g}")
    if exp == "e2":
        return "; ".join(f"{k}: species {v['species']['accuracy']:.3f}, behaviour {v['behaviour']['accuracy']:.3f}"
                         for k, v in out["conditions"].items())
    if exp == "e3":
        return "; ".join(f"{k}: left/stayed acc {v['left_vs_stayed']['accuracy']:.3f}"
                         for k, v in out["conditions"].items())
    if exp == "e4":
        return f"SIMULATED, {out['sim_config']['effectiveness']['mode']}: " + " ".join(
                        f"{p}: {s['success_rate']:.3f} succ, {s['mean_time_in_field_s']:.0f}s"
                        + (f", fallback {s['llm_fallback_rate']:.1%}, invalid {s['invalid_action_rate']:.1%}, "
                           f"early escalation {s['early_escalation_rate']:.1%}"
                           if s["llm_calls_per_event"] else "") + ";"
                        for p, s in out["summary"].items())
    if exp == "e4-grid":
        return f"{out['cells']} cells x {len(out['policies'])} policies x {out['repeats']} repeats: see e4_grid.csv"
    if exp == "e5":
        return (f"FA/h {out['fa_per_hour_before']:.2f} -> {out['fa_per_hour_after']:.2f}; "
                f"true rejected {out['true_events_rejected_by_gate']} of {out['triggers']} triggers")
    if exp == "e6-score":
        return json.dumps({"raters": out["raters"], **out["by_variant"], "wilcoxon": out["clarity_wilcoxon"]})
    return "rating sheet written"


def _log_result(exp: str, out: dict[str, Any], client) -> None:
    prov = E.provenance(client)
    commit = prov["commit"] + ("+dirty" if prov["dirty"] else "")
    cmd = "python -m farmguard.eval " + shlex.join(sys.argv[1:])
    text = _headline(exp, out).replace("|", "/").replace("\n", " ")
    RESULTS_LOG.parent.mkdir(exist_ok=True)
    with RESULTS_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"| {date.today()} | {commit} | {prov['model'] or '-'} | `{cmd}` | {text} |\n")


def _preflight(client) -> None:
    """A dead backend is otherwise invisible: LLM decisions fall back to the bandit and VLM answers
    look like abstentions, and the run still finishes with plausible numbers."""
    res = client.generate_json('Reply with JSON only: {"ok": true}',
                               {"type": "object", "properties": {"ok": {"type": "boolean"}}})
    if res.data is None:
        raise SystemExit(f"LLM backend {client.name}/{client.model} is not usable: {res.error}")


def _clips(patterns: list[str]) -> list[str]:
    clips = sorted({c for p in patterns for c in glob.glob(p)})
    if not clips:
        raise SystemExit(f"no clips match {patterns}")
    return clips


def main(argv: list[str] | None = None) -> dict[str, Any] | str:
    p = argparse.ArgumentParser(prog="python -m farmguard.eval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("exp", choices=["e1", "e2", "e3", "e4", "e4-grid", "e5", "e6", "e6-score"])
    p.add_argument("--settings", help="settings.yaml path")
    p.add_argument("--provider", help="ollama|gemini|anthropic|openai_compat|mock")
    p.add_argument("--vlm-model")
    p.add_argument("--llm-model", help="text model for the policy and alerts (E4, E6)")
    p.add_argument("--clips", nargs="*", default=[], help="E1/E5 video files or globs")
    p.add_argument("--runs", type=int, default=1, help="E1: passes over the clips")
    p.add_argument("--labels", help="label CSV, default from data/labels/")
    p.add_argument("--split", default="test", choices=["dev", "test", "all"],
                   help="default test: tune on dev only, never on test")
    p.add_argument("--lighting", help="only clips with this lighting in splits.csv, e.g. night")
    p.add_argument("--limit", type=int, help="first N samples (smoke test only)")
    p.add_argument("--with-detector", action="store_true", help="E3: add tracker-only and fused")
    p.add_argument("--zones", help="zones JSON")
    p.add_argument("--mock-annotations", help="use MockDetector with this annotation JSON")
    p.add_argument("--sim", default="config/sim.yaml")
    p.add_argument("--policies", default="fixed,random,rule_chain,bandit,llm")
    p.add_argument("--repeats", type=int, help="E4 default 5, grid default 30")
    p.add_argument("--mode", default="both", choices=["random", "informed", "both"], help="E4 effectiveness")
    p.add_argument("--n", type=int, default=20, help="E6: cases per language")
    p.add_argument("--ratings", nargs="*", default=[], help="E6 filled sheets, one per rater")
    p.add_argument("--no-log", action="store_true", help="do not append to paper/RESULTS.md")
    p.add_argument("--results-dir", help="write here instead of results/, e.g. for a timing probe "
                                         "that must not replace the numbers the paper's figures read")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.results_dir:
        E.RESULTS = Path(args.results_dir)

    ov: dict[str, Any] = {}
    if args.provider:
        ov["agent.provider"] = args.provider
    if args.vlm_model:
        ov["agent.vlm_model"] = args.vlm_model
    if args.llm_model:
        ov["agent.llm_model"] = args.llm_model
    if args.zones:
        ov["zones.file"] = args.zones
    if args.mock_annotations:
        ov["detector.backend"] = "mock"
        ov["detector.mock_annotations"] = args.mock_annotations
    cfg = load_config(args.settings, overrides=ov)

    from farmguard.agent.llm.providers import build_client
    policies = args.policies.split(",")
    client = None
    if args.exp in ("e1", "e2", "e3", "e5"):
        client = build_client(cfg, "vlm")
    elif args.exp == "e6" or (args.exp in ("e4", "e4-grid") and "llm" in policies):
        client = build_client(cfg, "llm")
    if client is not None:
        _preflight(client)
    # e1 builds its own client per system unless this mock is passed in
    mock = client if client is not None and client.name == "mock" else None

    if args.exp == "e1":
        out = E.e1_run(_clips(args.clips), ov, args.runs, args.settings, mock)
    elif args.exp == "e2":
        out = E.e2_vlm(cfg, client, args.labels or "data/labels/vlm_samples.csv", limit=args.limit,
                       split=args.split, lighting=args.lighting)
    elif args.exp == "e3":
        factory, zones = None, None
        if args.with_detector:
            from farmguard.fast.detector import build_detector
            from farmguard.fast.zones import ZoneSet

            def factory():
                return build_detector(cfg)
            zones = ZoneSet.load(args.zones) if args.zones else None
        out = E.e3_retreat(client, args.labels or "data/labels/retreat_samples.csv", limit=args.limit,
                           detector_factory=factory, zones=zones, split=args.split, lighting=args.lighting,
                           gone_frames=cfg.get("agent.retreat.gone_frames", 15))
    elif args.exp == "e4":
        out = {}
        for mode in (("random", "informed") if args.mode == "both" else (args.mode,)):
            out = E.e4_simulation(cfg, args.sim, policies, client, args.repeats or 5, mode)
            print(json.dumps({"mode": mode, "summary": out["summary"], "paired_tests": out["paired_tests"],
                              "warnings": out["warnings"]}, indent=2))
            for p, s in out["summary"].items():
                if s["llm_calls_per_event"]:
                    print(f"{mode} {p}: invalid {s['invalid_action_rate']:.1%}, fallback {s['llm_fallback_rate']:.1%}, "
                          f"early escalation {s['early_escalation_rate']:.1%}")
            if not args.no_log and (client is None or client.name != "mock"):
                _log_result(args.exp, out, client)
        return out
    elif args.exp == "e4-grid":
        out = E.e4_grid(cfg, args.sim, policies, client, args.repeats or 30)
    elif args.exp == "e5":
        out = E.e5_run(_clips(args.clips), args.labels or "data/labels/intrusions.csv", ov, client,
                       args.settings, split=args.split)
    elif args.exp == "e6":
        out = {"sheet": str(E.e6_alert_samples(cfg, client, args.n))}
    else:
        out = E.e6_score(args.ratings, args.labels or E.RESULTS / "e6_key.csv")

    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "paired_tests")}, indent=2,
                     ensure_ascii=False, default=str))
    if not args.no_log and (client is None or client.name != "mock"):
        _log_result(args.exp, out, client)
    return out


if __name__ == "__main__":
    main()
