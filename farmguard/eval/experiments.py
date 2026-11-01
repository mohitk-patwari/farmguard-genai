"""Experiments E1-E6 (see docs/PAPER_PLAN.md). Each function writes JSON + CSV into results/.
One command per experiment: `python -m farmguard.eval --help`.

Label files (formats in data/labels/README.md, made with scripts/label.py):
  data/labels/vlm_samples.csv     E2: sample_id, frames (';'-separated paths, time order),
                                      trigger_index, bbox ("x1,y1,x2,y2"), species, behaviour
  data/labels/retreat_samples.csv E3: sample_id, frames (';' paths after the sound), species, status
  data/labels/intrusions.csv      E5: clip, start_s, end_s, species   (true intrusions per clip)
  data/labels/splits.csv          clip, split (dev|test), lighting. Report on test; tune on dev only.
"""
from __future__ import annotations

import csv
import json
import logging
import random
import re
import statistics
import subprocess
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from farmguard.agent.clip_builder import build_event_images, sample_frames, to_jpeg
from farmguard.agent.llm.base import LLMClient
from farmguard.agent.prompts import SYSTEM_VISION, retreat_prompt, retreat_schema
from farmguard.agent.retreat import fuse, tracker_evidence
from farmguard.agent.vlm_check import VLMChecker
from farmguard.config import ROOT, Config, load_config
from farmguard.eval.labels import read_rows, select
from farmguard.eval.metrics import (
    classification_report,
    holm,
    latency_summary,
    mann_whitney,
    mcnemar,
    median_diff_ci,
    paired_permutation,
    wilcoxon_signed_rank,
    wilson_interval,
)
from farmguard.events import FramePacket, RetreatStatus

log = logging.getLogger(__name__)
RESULTS = Path("results")
SPLITS = Path("data/labels/splits.csv")
GATE_REJECTS = ("none", "other_animal", "unclassifiable")


def provenance(client: LLMClient | None = None) -> dict[str, Any]:
    """Written into every result so a number can be traced. provider == 'mock' means a pipeline
    check only: scripts/figures_all.py refuses to put those results in the paper."""
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], check=False, cwd=ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return ""
    return {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "commit": git("rev-parse", "--short", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--", ".", ":(exclude)paper")),
            "provider": client.name if client else None, "model": client.model if client else None}


def _write(name: str, obj: dict[str, Any], client: LLMClient | None = None) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps({"provenance": provenance(client), **obj}, indent=2, ensure_ascii=False,
                            default=str), encoding="utf-8")
    log.info("wrote %s", p)
    return p


def _write_csv(name: str, rows: list[dict[str, Any]], encoding: str = "utf-8") -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    fields = list(rows[0]) if rows else ["empty"]
    with p.open("w", newline="", encoding=encoding) as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %s", p)
    return p


def _slug(client: LLMClient) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", f"{client.name}_{client.model}")


def _paired_correct(truth: list[str], a: list[str], b: list[str]) -> dict[str, float]:
    """McNemar on per-sample correctness of two conditions over the same samples."""
    x = sum(p == t and q != t for t, p, q in zip(truth, a, b))
    y = sum(q == t and p != t for t, p, q in zip(truth, a, b))
    return {"a_only_correct": x, "b_only_correct": y, "mcnemar_p": mcnemar(x, y)}


# ---------------------------------------------------------------------------- E1 latency
def e1_latency(timing_csv: str | Path) -> dict[str, Any]:
    """Compare frame-to-alarm latency with the agent on vs off from one timing CSV (run
    `farmguard run` twice, once with --no-agent). e1_run does the whole thing in one command."""
    rows = read_rows(timing_csv)
    on = [r for r in rows if r["agent_enabled"] == "True" and r["reason"] == "first_alarm"]
    off = [r for r in rows if r["agent_enabled"] == "False" and r["reason"] == "first_alarm"]
    return _e1_report(on, off, {})


def _e1_report(on: list[dict], off: list[dict], detect: dict[str, list[float]],
               client: LLMClient | None = None) -> dict[str, Any]:
    def col(rows: list[dict], key: str) -> list[float]:
        return [float(r[key]) for r in rows if r[key] not in ("", "None", None)]

    out: dict[str, Any] = {}
    for key in ("read_to_request_ms", "read_to_start_ms"):
        a, b = col(on, key), col(off, key)
        out[key] = {"agent_on": latency_summary(a), "agent_off": latency_summary(b),
                    "mann_whitney": mann_whitney(a, b), "median_diff_on_minus_off": median_diff_ci(a, b)}
    if detect.get("agent_on") and detect.get("agent_off"):
        # Per-frame detector time: alarms are rare, frames are not, and a VLM sharing the GPU would
        # show up here first. If this rises, "unchanged" is not supported even with a clean alarm test.
        out["detect_ms_per_frame"] = {"agent_on": latency_summary(detect["agent_on"]),
                                      "agent_off": latency_summary(detect["agent_off"]),
                                      "median_diff_on_minus_off":
                                          median_diff_ci(detect["agent_on"], detect["agent_off"])}
    _write("e1_latency.json", out, client)
    return out


def e1_run(clips: list[str], overrides: dict[str, Any], runs: int = 1, settings: str | None = None,
           llm_client: LLMClient | None = None) -> dict[str, Any]:
    """Same clips with the agent on and off, order swapped every run, paced to real time."""
    from farmguard.agent.llm.providers import build_client
    from farmguard.fast.video_source import VideoSource
    from farmguard.system import build_system, run_video
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    detect: dict[str, list[float]] = {"agent_on": [], "agent_off": []}
    for r in range(runs):
        for clip in clips:
            for agent_on in ((True, False) if r % 2 == 0 else (False, True)):
                cfg = load_config(settings, overrides={
                    **overrides, "agent.enabled": agent_on,
                    "memory.db_path": str((RESULTS / "e1_memory.sqlite").resolve()),
                    "logging.events_jsonl": str((RESULTS / "e1_events.jsonl").resolve()),
                    "logging.timing_csv": str((RESULTS / "e1_timing_raw.csv").resolve())})
                system = build_system(cfg, llm_client=llm_client)
                src = VideoSource(clip, target_fps=cfg.get("video.target_fps", 10),
                                  resize_width=cfg.get("video.resize_width", 960))
                run_video(system, src.frames(), realtime=True)
                src.release()
                system.memory.conn.close()
                rows += [{"run": r, "clip": clip, **row} for row in system.latency.rows]
                detect["agent_on" if agent_on else "agent_off"] += system.fast.timer.samples["detect"]
    _write_csv("e1_alarms.csv", rows)
    first = [r for r in rows if r["reason"] == "first_alarm"]
    client = llm_client or build_client(load_config(settings, overrides=overrides), "vlm")
    return _e1_report([r for r in first if r["agent_enabled"]], [r for r in first if not r["agent_enabled"]],
                      detect, client)


# ---------------------------------------------------------------------------- E2 VLM
def _load_pkts(paths: list[str]) -> list[FramePacket]:
    pkts = []
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(p)
        pkts.append(FramePacket(index=i, video_time=float(i), frame=img))
    return pkts


def e2_vlm(cfg: Config, client: LLMClient, samples_csv: str | Path,
           modes: tuple[str, ...] = ("crop", "crop+full", "crop+full+burst"),
           location_hints: tuple[str | None, ...] = (None, "a farm in Maharashtra, India"),
           limit: int | None = None, split: str = "all", lighting: str | None = None) -> dict[str, Any]:
    rows = select(read_rows(samples_csv), SPLITS, split, lighting)[:limit]
    checker = VLMChecker(cfg, client)
    agent_cfg = cfg.raw.setdefault("agent", {})
    before = agent_cfg.get("include_location_in_prompt", False)
    agent_cfg["include_location_in_prompt"] = True  # ablation only
    per_sample, summary, preds = [], {}, {}
    try:
        for mode in modes:
            for hint in location_hints:
                key = f"{mode}|{'hint' if hint else 'no_hint'}"
                ys, yp, bs, bp, lat, errs = [], [], [], [], [], 0
                for r in rows:
                    pkts = _load_pkts([p for p in r["frames"].split(";") if p])
                    trig = pkts[int(r.get("trigger_index") or 0)]
                    bbox = tuple(float(v) for v in r["bbox"].split(",")) if r.get("bbox") else (
                        0, 0, trig.frame.shape[1], trig.frame.shape[0])
                    images, layout = build_event_images(trig, pkts, bbox, mode=mode)
                    rep = checker.check(images, layout, None, location_hint=hint)
                    ys.append(r["species"])
                    yp.append(rep.species)
                    bs.append(r["behaviour"])
                    bp.append(rep.behaviour)
                    lat.append(rep.latency_s)
                    errs += rep.error is not None
                    per_sample.append({"condition": key, "sample_id": r["sample_id"],
                                       "species_true": r["species"], "species_pred": rep.species,
                                       "behaviour_true": r["behaviour"], "behaviour_pred": rep.behaviour,
                                       "confidence": rep.confidence, "latency_s": rep.latency_s,
                                       "error": rep.error})
                # an error reads as an abstention, so a dead backend would look like a shy model
                summary[key] = {"species": classification_report(ys, yp),
                                "behaviour": classification_report(bs, bp),
                                "latency_s": latency_summary(lat), "n_errors": errs,
                                "model": client.model}
                preds[key] = (ys, yp, bs, bp)
    finally:
        agent_cfg["include_location_in_prompt"] = before
    tests: dict[str, Any] = {}
    keys = list(preds)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            pa, pb = preds[a], preds[b]
            tests[f"{a} vs {b}"] = {"species": _paired_correct(pa[0], pa[1], pb[1]),
                                    "behaviour": _paired_correct(pa[2], pa[3], pb[3])}
    out = {"split": split, "lighting": lighting, "conditions": summary, "paired_tests": tests}
    _write(f"e2_{_slug(client)}_summary.json", out, client)
    _write_csv(f"e2_{_slug(client)}_samples.csv", per_sample)
    return out


# ---------------------------------------------------------------------------- E3 retreat
def _binary(status: str) -> str:
    """The decision that matters: did it leave (gone/leaving) or not?"""
    if status in ("gone", "leaving"):
        return "left"
    return "stayed" if status == "still_present" else "uncertain"


def _tracker_status(pkts: list[FramePacket], species: str, detector: Any, zones: Any,
                    gone_frames: int) -> str:
    """Same evidence as RetreatChecker: detector + zone membership -> tracker_evidence()."""
    from farmguard.fast.zones import ZoneSet
    for i, p in enumerate(pkts):
        h, w = p.frame.shape[:2]
        zs = zones.rescaled(w, h) if zones else ZoneSet.full_frame(w, h)
        p.detections = detector.detect(p.frame, i)
        p.in_zone_track_ids = {d.track_id for d in p.detections
                               if d.track_id is not None and zs.zone_of(d)}
    return tracker_evidence(pkts, species, 0.0, gone_frames)[0].value


def e3_retreat(client: LLMClient, samples_csv: str | Path, n_frames: int = 4, limit: int | None = None,
               detector_factory: Callable[[], Any] | None = None, zones: Any = None,
               gone_frames: int = 15, split: str = "all", lighting: str | None = None) -> dict[str, Any]:
    """VLM-only, and with a detector also tracker-only and fused (agent/retreat.fuse), all against
    hand labels. detector_factory gives each window a fresh tracker, so ids never carry over."""
    rows = select(read_rows(samples_csv), SPLITS, split, lighting)[:limit]
    samples, lat = [], []
    for r in rows:
        pkts = _load_pkts([p for p in r["frames"].split(";") if p])
        chosen = sample_frames(pkts, n_frames, last_heavy=True)
        images = [to_jpeg(p.frame, 640) for p in chosen]
        res = client.generate_json(retreat_prompt(len(images), r["species"].replace("_", " ")),
                                   retreat_schema(), images, SYSTEM_VISION)
        data = res.data or {}
        vlm = data.get("status", "uncertain")
        row = {"sample_id": r["sample_id"], "status_true": r["status"], "vlm": vlm,
               "vlm_confidence": data.get("confidence"), "error": res.error}
        if detector_factory:
            tracker = _tracker_status(pkts, r["species"], detector_factory(), zones, gone_frames)
            vlm_status = RetreatStatus(vlm) if vlm in {s.value for s in RetreatStatus} else None
            row["tracker"] = tracker
            row["fused"] = fuse(RetreatStatus(tracker), vlm_status, data.get("confidence")).value
        samples.append(row)
        lat.append(res.latency_s)
    conditions = ["vlm"] + (["tracker", "fused"] if detector_factory else [])
    truth4 = [s["status_true"] for s in samples]
    truth2 = [_binary(t) for t in truth4]
    preds2 = {c: [_binary(s[c]) for s in samples] for c in conditions}
    out: dict[str, Any] = {
        "split": split, "lighting": lighting,
        "conditions": {c: {"four_way": classification_report(truth4, [s[c] for s in samples]),
                           "left_vs_stayed": classification_report(truth2, preds2[c])}
                       for c in conditions},
        "n_errors": sum(s["error"] is not None for s in samples),
        "latency_s": latency_summary(lat), "model": client.model}
    if detector_factory:
        out["paired_tests"] = {"fused vs tracker": _paired_correct(truth2, preds2["fused"], preds2["tracker"]),
                               "fused vs vlm": _paired_correct(truth2, preds2["fused"], preds2["vlm"])}
    _write(f"e3_{_slug(client)}.json", out, client)
    _write_csv(f"e3_{_slug(client)}_samples.csv", samples)
    return out


# ---------------------------------------------------------------------------- E4 simulator
def e4_paired(outcomes: list, baselines: tuple[str, ...] = ("fixed", "bandit")) -> dict[str, Any]:
    """McNemar on events is the plan's test, but events within a run share habituation state and are
    not independent. The run-level sign-flip test is the one that respects that: treat McNemar as
    descriptive. Holm-adjusted across this family. With few runs the run-level p has a floor."""
    by: dict[str, list] = {}
    for o in outcomes:
        by.setdefault(o.policy, []).append(o)
    tests: dict[str, Any] = {}
    for base in baselines:
        ref = by.get(base)
        if ref is None:
            continue
        for p, os_ in by.items():
            if p == base or len(os_) != len(ref) or f"{base}_vs_{p}" in tests:
                continue
            if any((o.run, o.event) != (f.run, f.event) for o, f in zip(os_, ref)):
                raise ValueError(f"{p} and {base} are not paired on the same events")
            b = sum(o.success and not f.success for o, f in zip(os_, ref))
            c = sum(f.success and not o.success for o, f in zip(os_, ref))
            runs: dict[int, list[list[float]]] = {}
            for o, f in zip(os_, ref):
                d = runs.setdefault(o.run, [[], []])
                d[0].append(o.success - f.success)
                d[1].append(o.time_in_field_s - f.time_in_field_s)
            succ_d = [statistics.fmean(v[0]) for v in runs.values()]
            time_d = [statistics.fmean(v[1]) for v in runs.values()]
            tests[f"{p}_vs_{base}"] = {
                "b": b, "c": c, "mcnemar_p": mcnemar(b, c), "runs": len(runs),
                "mean_success_diff": statistics.fmean(succ_d),
                "run_level_success_p": paired_permutation(succ_d),
                "mean_time_diff_s": statistics.fmean(time_d),
                "run_level_time_p": paired_permutation(time_d)}
    adjusted = holm({k: v["run_level_success_p"] for k, v in tests.items()})
    for k, v in tests.items():
        v["run_level_success_p_holm"] = adjusted[k]
    return tests


def _llm_warnings(summ: dict[str, dict[str, float]], client: LLMClient | None) -> list[str]:
    """LLMPolicy falls back to the bandit on any failure and still reports itself as "llm", so a
    dead backend would produce a clean-looking but false LLM result. Say so in the output."""
    out = []
    llm = summ.get("llm")
    if llm and llm["llm_fallback_rate"] > 0.01:
        out.append(f"llm_fallback_rate={llm['llm_fallback_rate']:.1%}: these decisions were the bandit's")
    if llm and client is not None and client.name == "mock":
        out.append("llm column is the mock client (first enum value): pipeline check, not a result")
    return out


def e4_simulation(cfg: Config, sim_path: str | Path, policies: list[str], llm_client=None,
                  repeats: int = 3, mode: str | None = None) -> dict[str, Any]:
    from farmguard.sim.simulator import learning_curve, load_sim, run_all, save_outcomes, summarize
    sim = load_sim(sim_path)
    if mode:
        sim["effectiveness"]["mode"] = mode
    mode = sim["effectiveness"]["mode"]
    outcomes = run_all(cfg, sim, policies, llm_client, repeats, cache_dir=RESULTS / "e4_llm_runs")
    save_outcomes(outcomes, RESULTS / f"e4_{mode}_outcomes.csv")
    summ = summarize(outcomes)
    for s in summ.values():
        n = int(s["events"])
        s["success_ci95"] = wilson_interval(round(s["success_rate"] * n), n)
    curve = learning_curve(outcomes)
    _write_csv(f"e4_{mode}_learning_curve.csv", curve)
    _write_csv(f"e4_{mode}_summary.csv", [{"policy": p, **s} for p, s in summ.items()])
    out = {"summary": summ, "paired_tests": e4_paired(outcomes), "learning_curve": curve,
           "warnings": _llm_warnings(summ, llm_client), "assumption_note":
           "Simulated deterrence. Every parameter in sim_config is an assumption, not field data.",
           "sim_config": sim}
    _write(f"e4_{mode}.json", out, llm_client)
    return out


def e4_grid(cfg: Config, sim_path: str | Path, policies: list[str], llm_client=None,
            repeats: int = 30) -> dict[str, Any]:
    """Full factorial sensitivity grid from sim.yaml; one row per cell x policy."""
    from farmguard.sim.simulator import grid_cells, load_sim, run_all, summarize
    sim = load_sim(sim_path)
    rows, tests = [], []
    for levels, scenario in grid_cells(sim):
        outcomes = run_all(cfg, scenario, policies, llm_client, repeats, cache_dir=RESULTS / "e4_llm_runs")
        for p, s in summarize(outcomes).items():
            rows.append({**levels, "policy": p, "runs": s["runs"], "events": s["events"],
                         "success_rate": s["success_rate"], "run_sd": s["success_rate_run_sd"],
                         "mean_time_in_field_s": s["mean_time_in_field_s"],
                         "late_success_rate": s["late_success_rate"],
                         "llm_fallback_rate": s["llm_fallback_rate"]})
        for name, t in e4_paired(outcomes).items():
            tests.append({**levels, "comparison": name, "mean_success_diff": t["mean_success_diff"],
                          "run_level_success_p": t["run_level_success_p"],
                          "run_level_success_p_holm": t["run_level_success_p_holm"],
                          "mean_time_diff_s": t["mean_time_diff_s"]})
    _write_csv("e4_grid.csv", rows)
    _write_csv("e4_grid_tests.csv", tests)
    out = {"cells": len(rows) // max(1, len(policies)), "repeats": repeats, "policies": policies,
           "rows": rows, "paired_tests": tests, "assumption_note":
           "Simulated deterrence. Grid levels are assumptions, not field data."}
    _write("e4_grid.json", out, llm_client)
    return out


# ---------------------------------------------------------------------------- E5 false alarms
def _e5_score(triggers: list[dict[str, Any]], truth: list[dict[str, str]], hours: float,
              tolerance_s: float) -> dict[str, Any]:
    """triggers: {clip, video_time, vlm_species}. Intervals only count within their own clip."""
    def covers(tr: dict[str, Any], t: dict[str, str]) -> bool:
        return (Path(t["clip"]).stem == Path(tr["clip"]).stem and
                float(t["start_s"]) - tolerance_s <= tr["video_time"] <= float(t["end_s"]) + tolerance_s)

    for tr in triggers:
        tr["is_true"] = any(covers(tr, t) for t in truth)
        tr["kept"] = tr.get("vlm_species") not in GATE_REJECTS
    kept = [tr for tr in triggers if tr["kept"]]
    fa_before = sum(not tr["is_true"] for tr in triggers)
    fa_after = sum(not tr["is_true"] for tr in kept)
    found_before = sum(any(covers(tr, t) for tr in triggers) for t in truth)
    found_after = sum(any(covers(tr, t) for tr in kept) for t in truth)
    return {"triggers": len(triggers), "video_hours": hours, "intervals": len(truth),
            "false_alarms_before": fa_before, "false_alarms_after": fa_after,
            "fa_per_hour_before": fa_before / hours if hours else float("nan"),
            "fa_per_hour_after": fa_after / hours if hours else float("nan"),
            "true_events_rejected_by_gate": sum(tr["is_true"] and not tr["kept"] for tr in triggers),
            "intervals_found_before": found_before, "intervals_found_after": found_after}


def e5_false_alarms(events_jsonl: str | Path, intrusions_csv: str | Path, video_hours: float,
                    tolerance_s: float = 5.0, clip: str | None = None) -> dict[str, Any]:
    """Scores an existing events.jsonl from one clip. Its events carry no clip name, so a labels
    file with several clips needs `clip`. e5_run does several clips in one command."""
    truth = read_rows(intrusions_csv)
    clips = {Path(t["clip"]).stem for t in truth}
    if clip is None and len(clips) > 1:
        raise ValueError(f"{intrusions_csv} covers {len(clips)} clips; pass clip= or use e5_run")
    name = clip or (next(iter(clips)) if clips else "clip")
    truth = [t for t in truth if Path(t["clip"]).stem == Path(name).stem]
    evs = [json.loads(l) for l in Path(events_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    agent = {e["event"]["event_id"]: e for e in evs if e.get("type") == "agent_result"}
    triggers = [{"clip": name, "video_time": e["video_time"],
                 "vlm_species": (agent.get(e["event_id"], {}).get("vlm") or {}).get("species")}
                for e in evs if e.get("type") == "intrusion"]
    out = _e5_score(triggers, truth, video_hours, tolerance_s)
    _write("e5_false_alarms.json", out)
    return out


def _e5_clip(cfg: Config, clip: str, checker: VLMChecker) -> tuple[list[dict[str, Any]], float]:
    """Fast path over one clip, then the VLM gate on each trigger's frames (same image builder and
    neighbours as the agent). Retreat checks and alerts are skipped: only the gate is measured."""
    from farmguard.fast.video_source import VideoSource
    from farmguard.system import build_system, run_video
    before, after = cfg.get("agent.clip.frames_before", 2), cfg.get("agent.clip.frames_after", 2)
    recent: deque[FramePacket] = deque(maxlen=before + 1)
    pending: list[tuple[Any, list[FramePacket]]] = []

    def on_frame(pkt: FramePacket, new_events: list) -> None:
        recent.append(pkt)
        for _, pkts in pending:
            if len(pkts) < before + 1 + after:
                pkts.append(pkt)
        pending.extend((ev, list(recent)) for ev in new_events)

    system = build_system(cfg, on_frame=on_frame)
    src = VideoSource(clip, target_fps=cfg.get("video.target_fps", 10),
                      resize_width=cfg.get("video.resize_width", 960))
    hours = src.cap.get(cv2.CAP_PROP_FRAME_COUNT) / src.native_fps / 3600
    run_video(system, src.frames(), realtime=False)
    src.release()
    system.memory.conn.close()
    triggers = []
    for ev, pkts in pending:
        trig = min(pkts, key=lambda p: abs(p.video_time - ev.video_time))
        images, layout = build_event_images(trig, pkts, ev.bbox, cfg.get("agent.clip.crop_padding", 0.25),
                                            cfg.get("agent.clip.max_side", 768))
        rep = checker.check(images, layout, ev.species)
        triggers.append({"clip": clip, "video_time": ev.video_time, "detector_species": ev.species,
                         "vlm_species": rep.species, "vlm_confidence": rep.confidence, "error": rep.error})
    return triggers, hours


def e5_run(clips: list[str], intrusions_csv: str | Path, overrides: dict[str, Any], client: LLMClient,
           settings: str | None = None, tolerance_s: float = 5.0, split: str = "all") -> dict[str, Any]:
    """Every clip listed counts as watched video: a clip with no row in intrusions.csv is a clean
    clip, so any trigger on it is a false alarm. List them explicitly so none is missed."""
    if split != "all":
        clips = [r["clip"] for r in select([{"clip": c} for c in clips], SPLITS, split)]
    stems = {Path(c).stem for c in clips}
    truth = [t for t in read_rows(intrusions_csv) if Path(t["clip"]).stem in stems]
    triggers, hours = [], 0.0
    RESULTS.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        cfg = load_config(settings, overrides={
            **overrides, "agent.enabled": False, "alarm.backend": "none",
            "memory.db_path": str((RESULTS / "e5_memory.sqlite").resolve()),
            "logging.events_jsonl": "", "logging.timing_csv": str((RESULTS / "e5_timing_raw.csv").resolve())})
        tr, h = _e5_clip(cfg, clip, VLMChecker(cfg, client))
        triggers += tr
        hours += h
    out = _e5_score(triggers, truth, hours, tolerance_s)
    out.update(split=split, n_errors=sum(t["error"] is not None for t in triggers), clips=len(clips))
    _write(f"e5_{_slug(client)}.json", out, client)
    _write_csv(f"e5_{_slug(client)}_triggers.csv", triggers)
    return out


# ---------------------------------------------------------------------------- E6 alerts
def e6_alert_samples(cfg: Config, client: LLMClient | None, n: int = 20, seed: int = 1) -> Path:
    """LLM and template alerts for the same facts. Raters get e6_rating_sheet.csv (shuffled, source
    hidden); e6_key.csv maps message ids back to the source and must not reach them."""
    from farmguard.agent.notifier import Notifier
    rng = random.Random(seed)
    base_ts = datetime(2026, 1, 15, 2, 0).timestamp()  # fixed, so the facts are reproducible
    notifier_cfg = cfg.raw.setdefault("notifier", {})
    before = notifier_cfg.get("language")
    rows = []
    try:
        for lang in ("hi", "mr"):
            notifier_cfg["language"] = lang
            nt = Notifier(cfg, client)
            for i in range(n):
                sp = rng.choice(list(cfg.species))
                info = cfg.species[sp]
                k = rng.randint(1, 3)
                sounds = rng.sample(info.allowed_sounds, k)
                status = rng.choice(["gone", "gone", "leaving", "still_present"])
                summ = nt.build_summary(sp, base_ts - rng.randint(0, 36000), sounds, status, k,
                                        rng.randint(1, 3), rng.choice(["eating", "moving_through"]))
                llm_msg, _, src = nt.compose(summ)
                tmpl = Notifier(cfg, None).compose(summ)[0]
                # a failed LLM call leaves two identical template messages: unusable as a pair
                for variant, msg in (("llm" if src == "llm" else "template_fallback", llm_msg),
                                     ("template", tmpl)):
                    rows.append({"lang": lang, "case": i, "variant": variant, "message": msg,
                                 "facts": json.dumps(summ, ensure_ascii=False)})
    finally:
        notifier_cfg["language"] = before
    rng.shuffle(rows)
    for mid, r in enumerate(rows):
        r["message_id"] = mid
    # utf-8-sig: otherwise Excel shows Hindi and Marathi as mojibake
    path = _write_csv("e6_rating_sheet.csv", [
        {"message_id": r["message_id"], "message": r["message"], "understood_yes_no": "",
         "facts": r["facts"], "facts_correct_yes_no": "", "clarity_1_5": ""} for r in rows],
        encoding="utf-8-sig")
    _write_csv("e6_key.csv", [{k: r[k] for k in ("message_id", "lang", "case", "variant")} for r in rows])
    return path


def e6_score(rating_csvs: list[str | Path], key_csv: str | Path) -> dict[str, Any]:
    """One filled sheet per rater (rater = file stem). Clarity: Wilcoxon on per-rater mean clarity,
    LLM vs template, so n = raters; ratings by one person are not independent."""
    key = {r["message_id"]: r for r in read_rows(key_csv)}
    usable = {(k["lang"], k["case"]) for k in key.values() if k["variant"] == "llm"}
    rows = []
    for path in rating_csvs:
        for r in read_rows(path):
            k = key[r["message_id"]]
            if (k["lang"], k["case"]) not in usable or not r.get("clarity_1_5"):
                continue
            rows.append({"rater": Path(path).stem, "variant": k["variant"], "lang": k["lang"],
                         "clarity": float(r["clarity_1_5"]),
                         "understood": r["understood_yes_no"].strip().lower().startswith("y"),
                         "facts_correct": r["facts_correct_yes_no"].strip().lower().startswith("y")})

    def stats(sel: list[dict]) -> dict[str, float]:
        n = len(sel)
        return {"n": n, "understood_pct": 100 * sum(r["understood"] for r in sel) / n if n else float("nan"),
                "facts_correct_pct": 100 * sum(r["facts_correct"] for r in sel) / n if n else float("nan"),
                "clarity_mean": statistics.fmean(r["clarity"] for r in sel) if n else float("nan")}

    raters = sorted({r["rater"] for r in rows})
    means = {v: [statistics.fmean(r["clarity"] for r in rows if r["rater"] == rt and r["variant"] == v)
                 for rt in raters] for v in ("llm", "template")} if rows else {"llm": [], "template": []}
    out = {"raters": len(raters), "cases_with_llm_alert": len(usable),
           "cases_lost_to_llm_failure": len({(k["lang"], k["case"]) for k in key.values()} - usable),
           "by_variant": {v: stats([r for r in rows if r["variant"] == v]) for v in ("llm", "template")},
           "by_variant_lang": {f"{v}_{lang}": stats([r for r in rows if r["variant"] == v and r["lang"] == lang])
                               for v in ("llm", "template") for lang in ("hi", "mr")},
           "clarity_wilcoxon": wilcoxon_signed_rank(means["llm"], means["template"])}
    _write("e6_scores.json", out)
    return out
