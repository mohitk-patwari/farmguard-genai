"""results/ -> paper/figures/*.pdf and paper/tables/*.tex. Experiments with no results are skipped.

  python scripts/figures_all.py

Results from the mock client are refused, as are E4 results whose LLM column is really the bandit:
neither may ever reach the paper. E4 text always says "simulated" (CLAUDE.md invariant 6).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger("figures")

# Categorical slots in validated order (node validate_palette.js, light mode, surface #ffffff).
# Markers carry identity too: IEEE proceedings are often read in greyscale.
POLICY_STYLE = {"fixed": ("#2a78d6", "o"), "random": ("#eb6834", "s"), "rule_chain": ("#1baf7a", "^"),
                "bandit": ("#eda100", "D"), "llm": ("#e87ba4", "v"),
                # ponytail: shares the bandit's hue (no validated 6th slot yet); the marker tells them apart
                "bandit_sw": ("#eda100", "X")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
SINGLE_COL = (3.5, 2.3)
DOUBLE_COL = (7.16, 2.5)
# docs/RESEARCH_NOTES.md section 1: 79 on-topic repos containing code, regex scan.
FEATURES = [("Ultralytics YOLOv8/11/12", 25), ("Recorded sound playback", 23), ("Keras/CNN classifier", 19),
            ("Buzzer", 14), ("Twilio SMS", 14), ("Email", 9), ("Firebase/app push", 8), ("Tracking", 7),
            ("WhatsApp", 7), ("Voice call", 4), ("Escalation by risk", 3), ("Local-language alert", 2),
            ("LLM/VLM in the loop", 2), ("Rotates deterrents", 1), ("Checks the animal left", 0),
            ("Logs detection-to-alarm latency", 0)]


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "lines.linewidth": 1.2, "lines.markersize": 4, "legend.frameon": False, "pdf.fonttype": 42,
        "figure.constrained_layout.use": True})


def _load(path: Path) -> dict[str, Any] | None:
    """A result the paper may use, or None."""
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("provenance", {}).get("provider") == "mock":
        log.warning("refused %s: mock client, pipeline check only", path)
        return None
    if data.get("warnings"):
        log.warning("refused %s: %s", path, data["warnings"])
        return None
    return data


def _esc(s: Any) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("|", r"\,/\,")


def _table(path: Path, caption: str, label: str, header: list[str], rows: list[list[Any]],
           note: str = "") -> None:
    """One booktabs table; \\resizebox keeps it inside an IEEE column."""
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\resizebox{\columnwidth}{!}{%", rf"\begin{{tabular}}{{l{'r' * (len(header) - 1)}}}", r"\toprule",
             " & ".join(_esc(h) for h in header) + r" \\", r"\midrule"]
    lines += [" & ".join(_esc(c) for c in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}}"]
    if note:
        lines.append(rf"\\[2pt]{{\footnotesize {note}}}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s", path)


def _save(fig, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)
    log.info("wrote %s", path)


def _pol_label(p: str) -> str:
    return {"rule_chain": "rule chain", "llm": "LLM", "bandit_sw": "bandit (window)"}.get(p, p)


def _p(v: float) -> str:
    return "<0.001" if v < 0.001 else f"{v:.3f}"


# ------------------------------------------------------------------ landscape (Figure 3)
def fig_features(figs: Path) -> None:
    names = [n for n, _ in FEATURES][::-1]
    vals = [v for _, v in FEATURES][::-1]
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.barh(names, vals, height=0.6, color=POLICY_STYLE["fixed"][0])
    for y, v in enumerate(vals):
        ax.text(v + 0.4, y, str(v), va="center", color=INK, fontsize=7)
    ax.set_xlabel("Repositories (of 79)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    _save(fig, figs / "feature_gap.pdf")


# ------------------------------------------------------------------ E4
def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def e4_summary_table(data: dict[str, Any], path: Path, label: str, mode: str) -> None:
    summ = data["summary"]
    runs = next(iter(summ.values()))["runs"]
    _table(path, f"SIMULATED deterrence, ``{mode}'' effectiveness, {runs} independent runs of "
           f"{data['sim_config']['nights']} nights. All parameters are assumptions, not field data.", label,
           ["Policy", "Success (run mean)", "95% CI", "Run SD", "First sound", "Late", "Time (s)", "LLM calls/event"],
           [[_pol_label(p), f"{s['run_mean_success']:.3f}",
             f"{s['run_mean_success_ci95'][0]:.3f}-{s['run_mean_success_ci95'][1]:.3f}",
             f"{s['success_rate_run_sd']:.3f}", f"{s['first_sound_success']:.3f}", f"{s['late_success_rate']:.3f}",
             f"{s['mean_time_in_field_s']:.0f}", f"{s['llm_calls_per_event']:.2f}"] for p, s in summ.items()],
           "CI: percentile bootstrap over runs. Escalation rate is 1 - success by construction.")


def e4(results: Path, figs: Path, tables: Path) -> None:
    modes = [m for m in ("random", "informed") if _load(results / f"e4_{m}.json")]
    if not modes:
        return
    fig, axes = plt.subplots(1, len(modes), figsize=DOUBLE_COL, sharey=True, squeeze=False)
    fig2, axes2 = plt.subplots(1, len(modes), figsize=DOUBLE_COL, sharex=True, squeeze=False)
    for i, mode in enumerate(modes):
        data = _load(results / f"e4_{mode}.json")
        summ = data["summary"]
        runs = next(iter(summ.values()))["runs"]
        e4_summary_table(data, tables / f"e4_{mode}.tex", f"tab:e4_{mode}", mode)
        b30 = _load(results / "e4_baselines30" / f"e4_{mode}.json")
        if b30:
            e4_summary_table(b30, tables / f"e4_{mode}_b30.tex", f"tab:e4_{mode}_b30", mode)
        prim = {k: t for k, t in data["paired_tests"].items() if t.get("holm_family") == "primary"}
        _table(tables / f"e4_{mode}_tests.tex",
               f"Is the LLM better? Simulated ``{mode}'' setting, {runs} runs. Holm-adjusted within these "
               f"comparisons. Smallest possible run-level p with {runs} runs: {2 / 2 ** runs:.2g}.",
               f"tab:e4_{mode}_tests",
               ["Comparison", "Success diff [95% CI]", "Runs won/lost", "Sign p", "Events b/c", "McNemar p",
                "Time diff (s)"],
               [[k.replace("_vs_", " vs ").replace("_", " "),
                 f"{t['mean_success_diff']:+.3f} [{t['success_diff_ci95'][0]:+.3f}, {t['success_diff_ci95'][1]:+.3f}]",
                 f"{t['runs_won']}/{t['runs_lost']}", _p(t["sign_test_p_holm"]), f"{t['b']}/{t['c']}",
                 _p(t["mcnemar_p_holm"]), f"{t['mean_time_diff_s']:+.0f}"] for k, t in prim.items()],
               "Diff and CI: mean of per-run differences, percentile bootstrap over runs (too narrow with few runs). "
               "McNemar pools events, which share habituation state within a run, so it overstates the evidence.")
        llm_rows = [(f"{mode}, constrained enum", s) for p, s in summ.items() if s["llm_calls_per_event"]]
        probe = _load(results / "probe_3b" / f"e4_{mode}.json")
        if probe:
            llm_rows.append((f"{mode}, unconstrained enum (1-run probe)", probe["summary"]["llm"]))
        if llm_rows:
            _table(tables / f"e4_{mode}_llm.tex", f"LLM decisions, simulated ``{mode}'' setting.",
                   f"tab:e4_{mode}_llm", ["LLM run", "Calls/event", "Invalid", "Fallback", "Early escalation"],
                   [[name, f"{s['llm_calls_per_event']:.2f}", f"{s['invalid_action_rate']:.1%}",
                     f"{s['llm_fallback_rate']:.1%}", f"{s['early_escalation_rate']:.1%}"] for name, s in llm_rows],
                   "Invalid = the validator changed the LLM's choice. Fallback = the call failed and the bandit decided.")

        ax = axes[0][i]
        by: dict[str, list[dict]] = defaultdict(list)
        for row in data["learning_curve"]:
            by[row["policy"]].append(row)
        for p in (q for q in POLICY_STYLE if q in by):
            rows = by[p]
            color, marker = POLICY_STYLE[p]
            xs = [r["night_start"] for r in rows]
            ys = [r["success_rate"] for r in rows]
            se = [r["se"] for r in rows]
            ax.fill_between(xs, [y - s for y, s in zip(ys, se)], [y + s for y, s in zip(ys, se)],
                            color=color, alpha=0.12, linewidth=0)
            ax.plot(xs, ys, color=color, marker=marker, label=_pol_label(p))
        ax.set_title(f"{mode} effectiveness (simulated)", color=INK, fontsize=8)
        ax.set_xlabel("Night (start of 5-night block)")
        if i == 0:
            ax.set_ylabel("Success rate")
        ax.set_ylim(0, 1)

        # time in field: same per-event mean as the table, interval from the spread of run means
        per_run: dict[tuple[str, str], list[float]] = defaultdict(list)
        for o in _read_csv(results / f"e4_{mode}_outcomes.csv"):
            per_run[(o["policy"], o["run"])].append(float(o["time_in_field_s"]))
        ax2 = axes2[0][i]
        names = list(summ)[::-1]
        for y, p in enumerate(names):
            means = [statistics.fmean(v) for (q, _), v in per_run.items() if q == p]
            m = summ[p]["mean_time_in_field_s"]
            half = 1.96 * statistics.stdev(means) / len(means) ** 0.5 if len(means) > 1 else 0.0
            color, marker = POLICY_STYLE[p]
            ax2.plot([m - half, m + half], [y, y], color=color, solid_capstyle="round")
            ax2.plot([m], [y], color=color, marker=marker, markersize=5)
            ax2.annotate(f"{m:.0f}", (m + half, y), xytext=(4, 0), textcoords="offset points", va="center",
                         fontsize=7, color=INK)
        ax2.set_yticks(range(len(names)), [_pol_label(p) for p in names])
        ax2.set_title(f"{mode} effectiveness (simulated)", color=INK, fontsize=8)
        ax2.set_xlabel("Mean time in field per event (s)")
        ax2.grid(axis="y", visible=False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center")
    _save(fig, figs / "e4_learning_curves.pdf")
    _save(fig2, figs / "e4_time_in_field.pdf")


def e4_grid(results: Path, figs: Path, tables: Path) -> None:
    data = _load(results / "e4_grid.json")
    if not data:
        return
    rows, factors = data["rows"], ["habituation_strength", "effectiveness", "eating_penalty", "cross_habituation"]
    titles = {"habituation_strength": "Habituation strength", "effectiveness": "Effectiveness",
              "eating_penalty": "Eating penalty", "cross_habituation": "Cross-habituation"}
    fig, axes = plt.subplots(1, len(factors), figsize=DOUBLE_COL, sharey=True)
    for ax, f in zip(axes, factors):
        levels = sorted({r[f] for r in rows}, key=str)
        for p in data["policies"]:
            ys = [statistics.fmean(r["success_rate"] for r in rows if r["policy"] == p and r[f] == lv)
                  for lv in levels]
            color, marker = POLICY_STYLE[p]
            ax.plot(range(len(levels)), ys, color=color, marker=marker, label=_pol_label(p))
        ax.set_xticks(range(len(levels)), [str(lv) for lv in levels])
        ax.set_xlabel(titles[f])
        ax.set_xlim(-0.3, len(levels) - 0.7)
    axes[0].set_ylabel("Mean success rate (simulated)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center")
    _save(fig, figs / "e4_sensitivity.pdf")

    tests = defaultdict(list)
    for t in data["paired_tests"]:
        tests[t["comparison"]].append(t)
    _table(tables / "e4_grid.tex",
           f"Sensitivity over {data['cells']} simulated scenarios ({data['repeats']} runs each). "
           "Win = higher success with Holm-adjusted run-level p < 0.05. Assumptions, not field data.",
           "tab:e4_grid", ["Comparison", "Mean diff", "Min diff", "Max diff", "Wins", "Losses"],
           [[k.replace("_vs_", " vs ").replace("_", " "),
             f"{statistics.fmean(t['mean_success_diff'] for t in v):+.3f}",
             f"{min(t['mean_success_diff'] for t in v):+.3f}", f"{max(t['mean_success_diff'] for t in v):+.3f}",
             sum(t["mean_success_diff"] > 0 and t["run_level_success_p_holm"] < 0.05 for t in v),
             sum(t["mean_success_diff"] < 0 and t["run_level_success_p_holm"] < 0.05 for t in v)]
            for k, v in tests.items()])


# ------------------------------------------------------------------ E1, E2, E3, E5, E6
def e1(results: Path, figs: Path, tables: Path) -> None:
    data = _load(results / "e1_latency.json")
    if not data:
        return
    keys = {"read_to_request_ms": "Read to alarm request (ms)", "read_to_start_ms": "Read to audio start (ms)",
            "detect_ms_per_frame": "Detector per frame (ms)"}
    rows = []
    for k, title in keys.items():
        d = data.get(k)
        if d:
            on, off, diff = d["agent_on"], d["agent_off"], d["median_diff_on_minus_off"]
            mw = d.get("mann_whitney", {}).get("p")
            rows.append([title, f"{off['median']:.1f} / {off['p95']:.1f} (n={off['n']})",
                         f"{on['median']:.1f} / {on['p95']:.1f} (n={on['n']})",
                         f"{diff['diff']:+.1f} [{diff['ci_low']:+.1f}, {diff['ci_high']:+.1f}]",
                         _p(mw) if mw is not None else "-"])
    _table(tables / "e1.tex", "Alarm latency with the agent off and on (median / p95, same clips).", "tab:e1",
           ["Measure", "Agent off", "Agent on", "Median diff [95% CI]", "MWU p"], rows)
    alarms = _read_csv(results / "e1_alarms.csv") if (results / "e1_alarms.csv").exists() else []
    first = [r for r in alarms if r["reason"] == "first_alarm" and r["read_to_start_ms"]]
    if first:
        fig, ax = plt.subplots(figsize=SINGLE_COL)
        groups = [[float(r["read_to_start_ms"]) for r in first if r["agent_enabled"] == c] for c in ("False", "True")]
        ax.boxplot(groups, orientation="horizontal", tick_labels=["Agent off", "Agent on"], widths=0.5,
                   medianprops={"color": POLICY_STYLE["fixed"][0], "linewidth": 1.5},
                   flierprops={"markersize": 3})
        ax.set_xlabel("Frame read to alarm start (ms)")
        ax.grid(axis="y", visible=False)
        _save(fig, figs / "e1_latency.pdf")


def e2(results: Path, figs: Path, tables: Path) -> None:
    for path in sorted(results.glob("e2_*_summary.json")):
        data = _load(path)
        if not data:
            continue
        name = path.stem.removeprefix("e2_").removesuffix("_summary")
        cond = data["conditions"]
        _table(tables / f"e2_{name}.tex",
               f"VLM on the {data['split']} split ({_esc(data['provenance']['model'])}).", f"tab:e2_{name}",
               ["Condition", "n", "Species acc", "Committed", "Macro-F1", "Abstain", "Behaviour acc", "Latency (s)", "Errors"],
               [[k, c["species"]["n"], f"{c['species']['accuracy']:.3f}", f"{c['species']['accuracy_when_committed']:.3f}",
                 f"{c['species']['macro_f1']:.3f}", f"{c['species']['abstention_rate']:.3f}",
                 f"{c['behaviour']['accuracy']:.3f}", f"{c['latency_s'].get('median', float('nan')):.1f}", c["n_errors"]]
                for k, c in cond.items()])
        modes = list(dict.fromkeys(k.split("|")[0] for k in cond))
        fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL, sharey=True)
        for ax, field, title in zip(axes, ("species", "behaviour"), ("Species accuracy", "Behaviour accuracy")):
            for hint, color, marker, lab in (("no_hint", POLICY_STYLE["fixed"][0], "o", "no location hint"),
                                             ("hint", POLICY_STYLE["random"][0], "s", "location hint")):
                ys = [cond[f"{m}|{hint}"][field]["accuracy"] for m in modes if f"{m}|{hint}" in cond]
                if ys:
                    ax.plot(modes[:len(ys)], ys, color=color, marker=marker, label=lab)
            ax.set_title(title, color=INK, fontsize=8)
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", labelsize=7)
        axes[0].legend()
        _save(fig, figs / f"e2_{name}.pdf")


def e3(results: Path, tables: Path) -> None:
    for path in sorted(p for p in results.glob("e3_*.json")):
        data = _load(path)
        if not data:
            continue
        name = path.stem.removeprefix("e3_")
        _table(tables / f"e3_{name}.tex",
               f"Did the animal leave? Hand labels, {data['split']} split ({_esc(data['provenance']['model'])}).",
               f"tab:e3_{name}", ["Method", "n", "Accuracy", "Macro-F1", "Abstain", "Committed", "4-way acc"],
               [[k, c["left_vs_stayed"]["n"], f"{c['left_vs_stayed']['accuracy']:.3f}",
                 f"{c['left_vs_stayed']['macro_f1']:.3f}", f"{c['left_vs_stayed']['abstention_rate']:.3f}",
                 f"{c['left_vs_stayed']['accuracy_when_committed']:.3f}", f"{c['four_way']['accuracy']:.3f}"]
                for k, c in data["conditions"].items()])


def e5(results: Path, figs: Path, tables: Path) -> None:
    for path in sorted(p for p in results.glob("e5_*.json")):
        data = _load(path)
        if not data or "fa_per_hour_before" not in data:
            continue
        name = path.stem.removeprefix("e5_")
        _table(tables / f"e5_{name}.tex", f"False alarms over {data['video_hours']:.2f} h of video.", f"tab:e5_{name}",
               ["Condition", "False alarms", "Per hour", "Intervals found"],
               [["Fast path", data["false_alarms_before"], f"{data['fa_per_hour_before']:.2f}",
                 f"{data['intervals_found_before']}/{data['intervals']}"],
                ["+ VLM gate", data["false_alarms_after"], f"{data['fa_per_hour_after']:.2f}",
                 f"{data['intervals_found_after']}/{data['intervals']}"]],
               f"True events rejected by the gate: {data['true_events_rejected_by_gate']}.")
        fig, ax = plt.subplots(figsize=(3.5, 1.3))
        ax.barh(["+ VLM gate", "Fast path"], [data["fa_per_hour_after"], data["fa_per_hour_before"]], height=0.5,
                color=POLICY_STYLE["fixed"][0])
        ax.set_xlabel("False alarms per hour")
        ax.grid(axis="y", visible=False)
        _save(fig, figs / f"e5_{name}.pdf")


def e6(results: Path, tables: Path) -> None:
    data = _load(results / "e6_scores.json")
    if not data:
        return
    v = data["by_variant"]
    _table(tables / "e6.tex", f"Blind alert ratings ({data['raters']} raters).", "tab:e6",
           ["Alert", "Ratings", "Understood %", "Facts correct %", "Clarity (1-5)"],
           [[k, v[k]["n"], f"{v[k]['understood_pct']:.1f}", f"{v[k]['facts_correct_pct']:.1f}",
             f"{v[k]['clarity_mean']:.2f}"] for k in ("llm", "template")],
           f"Clarity Wilcoxon on per-rater means: p = {_p(data['clarity_wilcoxon']['p'])}.")


def main(argv: list[str] | None = None, results: str = "results", paper: str = "paper") -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=results)
    p.add_argument("--paper", default=paper)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    res, figs, tables = Path(a.results), Path(a.paper) / "figures", Path(a.paper) / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    _style()
    fig_features(figs)
    e1(res, figs, tables)
    e2(res, figs, tables)
    e3(res, tables)
    e4(res, figs, tables)
    e4_grid(res, figs, tables)
    e5(res, figs, tables)
    e6(res, tables)


if __name__ == "__main__":
    main()
