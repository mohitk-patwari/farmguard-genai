"""E4 position-bias ablation and hybrid, one Holm family (SIMULATED):

  python scripts/e4_ablation.py --shuffled results/e4_shuffle --ordered results \
      --ordered-log results/e4_llm_runs/<hash> --out results/e4_ablation.json

Comparisons: llm vs bandit, llm vs random, llm_bandit vs bandit (shuffled run), and llm shuffled vs
llm ordered (the 773ad66 run, on the repeats both have). Primary: run-level sign test and bootstrap CI
of the mean per-run difference. McNemar is descriptive (events in a run share habituation state).
first_position_rate is compared with chance, the mean of 1/options shown; calls that showed one
option are reported apart because they inflate it.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

from farmguard.eval.experiments import holm_family, paired_stats

PAIRS = [("llm", "bandit"), ("llm", "random"), ("llm_bandit", "bandit")]


def load(path: Path) -> dict[str, list]:
    by: dict[str, list] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by.setdefault(r["policy"], []).append(SimpleNamespace(
                run=int(r["run"]), event=int(r["event"]), species=r["species"], night=r["night"],
                success=int(r["success"]), time_in_field_s=float(r["time_in_field_s"])))
    return by


def position_stats(log_dir: Path, policy: str) -> dict[str, float | int]:
    first = n = multi = multi_first = 0
    chance = multi_chance = 0.0
    for f in sorted(log_dir.glob(f"{policy}_r*.decisions.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            shown = r.get("shown_order") or r.get("enum")  # ordered logs predate shown_order: enum order was shown
            if not shown or r["proposed"] not in shown:
                continue
            top = shown.index(r["proposed"]) == 0
            first, n, chance = first + top, n + 1, chance + 1 / len(shown)
            if len(shown) > 1:
                multi, multi_first, multi_chance = multi + 1, multi_first + top, multi_chance + 1 / len(shown)
    return {"answers": n, "first_position_rate": first / n if n else None, "chance": chance / n if n else None,
            "multi_option_answers": multi, "multi_first_position_rate": multi_first / multi if multi else None,
            "multi_chance": multi_chance / multi if multi else None}


def ablation(shuffled: dict[str, list], ordered: dict[str, list]) -> dict[str, dict]:
    tests = {f"{p}_vs_{b}": paired_stats(shuffled[p], shuffled[b], f"{p} and {b}")
             for p, b in PAIRS if p in shuffled and b in shuffled}
    runs = {o.run for o in ordered["llm"]}
    sh = [o for o in shuffled["llm"] if o.run in runs]
    # same seeds and scenario => same event stream; check it, not just the indices
    if [(o.species, o.night) for o in sh] != [(o.species, o.night) for o in ordered["llm"]]:
        raise ValueError("shuffled and ordered runs saw different events")
    tests["llm_shuffled_vs_llm_ordered"] = paired_stats(sh, ordered["llm"], "shuffled and ordered llm")
    holm_family(tests, list(tests), "ablation")
    return tests


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shuffled", type=Path, default=Path("results/e4_shuffle"))
    ap.add_argument("--ordered", type=Path, default=Path("results"))
    ap.add_argument("--ordered-log", type=Path, required=True, help="decision-log dir of the ordered llm run")
    ap.add_argument("--mode", default="random")
    ap.add_argument("--out", type=Path, default=Path("results/e4_ablation.json"))
    a = ap.parse_args(argv)
    shuffled = load(a.shuffled / f"e4_{a.mode}_outcomes.csv")
    ordered = load(a.ordered / f"e4_{a.mode}_outcomes.csv")
    shuffled_logs = sorted(a.shuffled.glob("e4_llm_runs/*"))
    if len(shuffled_logs) != 1:
        raise SystemExit(f"expected one shuffled decision-log dir, found {shuffled_logs}")
    out = {"label": "SIMULATED", "mode": a.mode, "tests": ablation(shuffled, ordered),
           "position": {"llm_ordered": position_stats(a.ordered_log, "llm"),
                        **{f"{p}_shuffled": position_stats(shuffled_logs[0], p) for p in ("llm", "llm_bandit")}},
           "success": {p: statistics.fmean(o.success for o in os_) for p, os_ in shuffled.items()}}
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, t in out["tests"].items():
        print(f"{k}: diff {t['mean_success_diff']:+.3f} {t['success_diff_ci95']}, runs {t['runs_won']}/"
              f"{t['runs_lost']} of {t['runs']}, sign p {t['sign_test_p']:.4f} holm {t['sign_test_p_holm']:.4f}")
    for k, s in out["position"].items():
        print(k, s)


if __name__ == "__main__":
    main()
