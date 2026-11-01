"""Experiments E1-E6 (see docs/PAPER_PLAN.md). Each function writes CSV/JSON into results/.

Label files (create them while labelling; formats are documented in data/labels/README.md):
  data/labels/vlm_samples.csv     E2: sample_id, frames (';'-separated paths, time order),
                                      trigger_index, bbox ("x1,y1,x2,y2"), species, behaviour
  data/labels/retreat_samples.csv E3: sample_id, frames (';' paths after the sound), species, status
  data/labels/intrusions.csv      E5: clip, start_s, end_s, species   (true intrusions per clip)
"""
from __future__ import annotations

import csv
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import cv2

from farmguard.agent.clip_builder import build_event_images, sample_frames, to_jpeg
from farmguard.agent.llm.base import LLMClient
from farmguard.agent.prompts import SYSTEM_VISION, retreat_prompt, retreat_schema
from farmguard.agent.vlm_check import VLMChecker
from farmguard.config import Config
from farmguard.eval.metrics import classification_report, latency_summary, mcnemar, wilson_interval
from farmguard.events import FramePacket

log = logging.getLogger(__name__)
RESULTS = Path("results")


def _write(name: str, obj: Any) -> Path:
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log.info("wrote %s", p)
    return p


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------- E1 latency
def e1_latency(timing_csv: str | Path) -> dict[str, Any]:
    """Compare frame-to-alarm latency with the agent on vs off (run `farmguard run` twice,
    once with --no-agent, on the same clips; both append to the same timing CSV)."""
    rows = _read_csv(timing_csv)
    out = {}
    for cond in ("True", "False"):
        sel = [r for r in rows if r["agent_enabled"] == cond and r["reason"] == "first_alarm"]
        out["agent_on" if cond == "True" else "agent_off"] = {
            "read_to_request_ms": latency_summary([float(r["read_to_request_ms"]) for r in sel]),
            "read_to_start_ms": latency_summary([float(r["read_to_start_ms"]) for r in sel
                                                 if r["read_to_start_ms"] not in ("", "None")])}
    _write("e1_latency.json", out)
    return out


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
           limit: int | None = None) -> dict[str, Any]:
    rows = _read_csv(samples_csv)[:limit]
    checker = VLMChecker(cfg, client)
    cfg.raw.setdefault("agent", {})["include_location_in_prompt"] = True  # ablation only
    per_sample, summary = [], {}
    for mode in modes:
        for hint in location_hints:
            key = f"{mode}|{'hint' if hint else 'no_hint'}"
            ys, yp, bs, bp, lat = [], [], [], [], []
            for r in rows:
                pkts = _load_pkts([p for p in r["frames"].split(";") if p])
                trig = pkts[int(r.get("trigger_index") or 0)]
                bbox = tuple(float(v) for v in r["bbox"].split(",")) if r.get("bbox") else (
                    0, 0, trig.frame.shape[1], trig.frame.shape[0])
                images, layout = build_event_images(trig, pkts, bbox, mode=mode)
                rep = checker.check(images, layout, None, location_hint=hint)
                ys.append(r["species"]); yp.append(rep.species)
                bs.append(r["behaviour"]); bp.append(rep.behaviour)
                lat.append(rep.latency_s)
                per_sample.append({"condition": key, "sample_id": r["sample_id"],
                                   "species_true": r["species"], "species_pred": rep.species,
                                   "behaviour_true": r["behaviour"], "behaviour_pred": rep.behaviour,
                                   "confidence": rep.confidence, "latency_s": rep.latency_s,
                                   "error": rep.error})
            summary[key] = {"species": classification_report(ys, yp),
                            "behaviour": classification_report(bs, bp),
                            "latency_s": latency_summary(lat), "model": client.model}
    cfg.raw["agent"]["include_location_in_prompt"] = False
    _write("e2_vlm_summary.json", summary)
    _write("e2_vlm_samples.json", per_sample)
    return summary


# ---------------------------------------------------------------------------- E3 retreat
def e3_retreat(client: LLMClient, samples_csv: str | Path, n_frames: int = 4,
               limit: int | None = None) -> dict[str, Any]:
    """VLM retreat judgement vs hand labels. For the tracker-only baseline, run the detector on
    the same frames (see `farmguard eval e3 --with-detector`) and compare."""
    rows = _read_csv(samples_csv)[:limit]
    yt, yp, lat = [], [], []
    for r in rows:
        pkts = _load_pkts([p for p in r["frames"].split(";") if p])
        chosen = sample_frames(pkts, n_frames, last_heavy=True)
        images = [to_jpeg(p.frame, 640) for p in chosen]
        res = client.generate_json(retreat_prompt(len(images), r["species"].replace("_", " ")),
                                   retreat_schema(), images, SYSTEM_VISION)
        yt.append(r["status"])
        yp.append((res.data or {}).get("status", "uncertain"))
        lat.append(res.latency_s)
    # collapse to the decision that matters: did it leave (gone/leaving) or not?
    bin_t = ["left" if s in ("gone", "leaving") else "stayed" for s in yt]
    bin_p = ["left" if s in ("gone", "leaving") else ("stayed" if s == "still_present" else "uncertain")
             for s in yp]
    out = {"four_way": classification_report(yt, yp), "left_vs_stayed": classification_report(bin_t, bin_p),
           "latency_s": latency_summary(lat), "model": client.model}
    _write("e3_retreat.json", out)
    return out


# ---------------------------------------------------------------------------- E4 simulator
def e4_simulation(cfg: Config, sim_path: str | Path, policies: list[str], llm_client=None,
                  repeats: int = 3) -> dict[str, Any]:
    from farmguard.sim.simulator import load_sim, run_all, save_outcomes, summarize
    sim = load_sim(sim_path)
    outcomes = run_all(cfg, sim, policies, llm_client, repeats)
    save_outcomes(outcomes, RESULTS / "e4_outcomes.csv")
    summ = summarize(outcomes)
    # paired comparison of each policy vs fixed on identical events (McNemar on success)
    by = {p: [o for o in outcomes if o.policy == p] for p in policies}
    tests = {}
    base = by.get("fixed")
    for p, os_ in by.items():
        if base is None or p == "fixed" or len(os_) != len(base):
            continue
        b = sum(o.success and not f.success for o, f in zip(os_, base))
        c = sum(f.success and not o.success for o, f in zip(os_, base))
        tests[f"{p}_vs_fixed"] = {"b": b, "c": c, "mcnemar_p": mcnemar(b, c)}
    for p, s in summ.items():
        n = int(s["events"])
        s["success_ci95"] = wilson_interval(round(s["success_rate"] * n), n)
    out = {"summary": summ, "paired_tests": tests, "sim_config": sim}
    _write("e4_simulation.json", out)
    return out


# ---------------------------------------------------------------------------- E5 false alarms
def e5_false_alarms(events_jsonl: str | Path, intrusions_csv: str | Path, video_hours: float,
                    tolerance_s: float = 5.0) -> dict[str, Any]:
    """False alarms per hour before (every fast-path trigger) and after the VLM gate (triggers
    whose VLM species is none / other_animal / unclassifiable are counted as rejected)."""
    truth = _read_csv(intrusions_csv)
    evs = [json.loads(l) for l in Path(events_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    intr = {e["event_id"]: e for e in evs if e.get("type") == "intrusion"}
    agent = {e["event"]["event_id"]: e for e in evs if e.get("type") == "agent_result"}

    def is_true(e: dict) -> bool:
        return any(float(t["start_s"]) - tolerance_s <= e["video_time"] <= float(t["end_s"]) + tolerance_s
                   for t in truth)

    fa_before = sum(not is_true(e) for e in intr.values())
    kept = [e for eid, e in intr.items()
            if not (agent.get(eid, {}).get("vlm") or {}).get("species") in ("none", "other_animal",
                                                                              "unclassifiable")]
    fa_after = sum(not is_true(e) for e in kept)
    missed_after = sum(is_true(e) for e in intr.values()) - sum(is_true(e) for e in kept)
    out = {"triggers": len(intr), "false_alarms_before": fa_before, "false_alarms_after": fa_after,
           "fa_per_hour_before": fa_before / video_hours, "fa_per_hour_after": fa_after / video_hours,
           "true_events_rejected_by_gate": missed_after}
    _write("e5_false_alarms.json", out)
    return out


# ---------------------------------------------------------------------------- E6 alerts
def e6_alert_samples(cfg: Config, client: LLMClient | None, n: int = 20, seed: int = 1) -> Path:
    """Generates LLM alerts and template alerts for the same facts, shuffled into a CSV for a
    blind farmer / volunteer rating (understood? correct facts? 1-5 clarity)."""
    from farmguard.agent.notifier import Notifier
    rng = random.Random(seed)
    rows = []
    for lang in ("hi", "mr"):
        cfg.raw.setdefault("notifier", {})["language"] = lang
        nt = Notifier(cfg, client)
        for i in range(n):
            sp = rng.choice(list(cfg.species))
            info = cfg.species[sp]
            k = rng.randint(1, 3)
            sounds = rng.sample(info.allowed_sounds, k)
            status = rng.choice(["gone", "gone", "leaving", "still_present"])
            summ = nt.build_summary(sp, time.time() - rng.randint(0, 36000), sounds, status, k,
                                    rng.randint(1, 3), rng.choice(["eating", "moving_through"]))
            llm_msg, _, src = nt.compose(summ)
            tmpl = Notifier(cfg, None).compose(summ)[0]
            pair = [("llm" if src == "llm" else "template_fallback", llm_msg), ("template", tmpl)]
            rng.shuffle(pair)
            for variant, msg in pair:
                rows.append({"lang": lang, "case": i, "variant_hidden": variant, "message": msg,
                             "facts": json.dumps(summ, ensure_ascii=False),
                             "understood_yes_no": "", "facts_correct_yes_no": "", "clarity_1_5": ""})
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "e6_alert_rating_sheet.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path
