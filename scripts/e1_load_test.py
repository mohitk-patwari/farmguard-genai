"""E1 load test: does a real VLM running in parallel delay the first alarm?

Plays a clip like a camera (live mode: paced, stale frames dropped) through the real detector, in
two arms: fast path alone, and fast path + agent with the real VLM. Run on the same machine and GPU:

    python scripts/e1_load_test.py --clip data/clips/x.mp4 --repeats 40 --provider ollama

An alarm counts as "under load" only if a VLM call overlapped its first-zone-to-alarm interval, and
that count is reported. If it is near zero the agent arm has not tested the claim.

Without Ollama, `--provider mock --stand-in-load` holds each mock call open while a separate process
loads the GPU. That checks this script and shows sensitivity to GPU contention, but it is NOT the E1
result: the JSON says real_vlm=false, and the output goes to e1_load_standin.json.
Writes results/e1_load.json and one CSV of alarms per arm.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from farmguard.agent.llm.providers import build_client
from farmguard.config import load_config
from farmguard.fast.trigger import Trigger
from farmguard.fast.video_source import VideoSource
from farmguard.system import build_system

log = logging.getLogger("e1_load")


class SpyClient:
    """Wraps the real client and records when each call ran. Decides nothing."""

    def __init__(self, client, hold_s: float = 0.0):
        self._c = client
        self.hold_s = hold_s
        self.spans: list[tuple[float, float, bool]] = []
        self._lock = threading.Lock()
        self.active = 0

    def __getattr__(self, name):
        return getattr(self._c, name)

    def generate_json(self, *a, **k):
        t0 = time.perf_counter()
        with self._lock:
            self.active += 1
        try:
            time.sleep(self.hold_s)
            res = self._c.generate_json(*a, **k)
            ok = res.error is None
            return res
        finally:
            with self._lock:
                self.active -= 1
                self.spans.append((t0, time.perf_counter(), locals().get("ok", False)))


# Stand-in only: keeps about 3 GB of VRAM and the GPU busy, in its own process like Ollama would be.
GPU_STRESS = """
import torch
x = torch.rand(9000, 9000, device="cuda", dtype=torch.float32)
y = torch.rand(9000, 9000, device="cuda", dtype=torch.float32)
while True:
    x = (x @ y).clamp(0, 1)
"""


def overlaps(spans, start: float, end: float) -> bool:
    return any(a <= end and b >= start for a, b, _ in spans)


def pct(vals: list[float], q: float) -> float | None:
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(q * len(s)))], 2) if s else None


def summarise(vals: list[float]) -> dict:
    return {"n": len(vals), "median_ms": pct(vals, .5), "p95_ms": pct(vals, .95),
            "max_ms": round(max(vals), 2) if vals else None}


def gpu_snapshot() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                               "--format=csv,noheader"], check=False, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except OSError:
        return "n/a"


def run_arm(name: str, args, agent_on: bool) -> dict:
    timing = ROOT / "results" / f"e1_load_{name}.csv"
    timing.unlink(missing_ok=True)
    cfg = load_config(overrides={
        "agent.enabled": agent_on, "agent.provider": args.provider, "alarm.backend": args.alarm_backend,
        "logging.timing_csv": str(timing), "logging.events_jsonl": None,
        "memory.db_path": str(ROOT / "results" / f"e1_load_{name}.sqlite"),
        "notifier.channels": [], "detector.model": args.model,
        "detector.open_vocabulary": False, "zones.file": str(ROOT / "config" / "does_not_exist.json")})
    spy = SpyClient(build_client(cfg, "vlm"), 8.0 if args.stand_in_load else 0.0) if agent_on else None
    stress = (subprocess.Popen([sys.executable, "-c", GPU_STRESS])
              if agent_on and args.stand_in_load else None)
    if stress:
        time.sleep(8)  # let it reach full load before the first frame
    system = build_system(cfg, llm_client=spy)
    fast = system.fast
    frames = gpu_samples = 0
    dropped = 0
    offset = 0.0
    gpu: list[str] = []
    t_start = time.perf_counter()
    for rep in range(args.repeats):
        fast.trigger = Trigger(window=cfg.get("trigger.window"), min_hits=cfg.get("trigger.min_hits"),
                               track_cooldown_s=cfg.get("trigger.track_cooldown_s"),
                               species_cooldown_s=cfg.get("trigger.species_cooldown_s"))
        src = VideoSource(args.clip, live=True, target_fps=cfg.get("video.target_fps", 10),
                          resize_width=cfg.get("video.resize_width", 960))
        last = 0.0
        for idx, vt, frame, t_read in src.frames():
            fast.process(idx, offset + vt, frame, t_read)
            frames += 1
            last = vt
            if frames % 50 == 0 and gpu_samples < 40:
                gpu.append(gpu_snapshot())
                gpu_samples += 1
        dropped += src.dropped
        src.release()
        offset += last + 1.0
        log.info("%s rep %d/%d: %d events so far", name, rep + 1, args.repeats, len(fast.events))
    elapsed = time.perf_counter() - t_start
    if stress:
        stress.kill()
    fast.run(iter(()))  # finishes the buffer, waits for the alarm thread, saves the CSV
    if system.agent:
        system.agent.stop(wait=True, timeout=args.drain_timeout)
    system.alarm.stop()
    rows = []
    if timing.exists():
        import csv
        with timing.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    usable = [r for r in rows if r["read_to_start_ms"]]
    def under_load(r):
        return spy is not None and overlaps(spy.spans, float(r["t_first_zone"]), float(r["t_alarm_started"]))
    loaded = [r for r in usable if under_load(r)]
    detect = fast.timer.summary().get("detect", {})
    return {
        "arm": name, "frames_processed": frames, "frames_dropped_live": dropped,
        "effective_fps": round(frames / elapsed, 2), "alarms": len(rows), "alarms_failed": len(rows) - len(usable),
        "backend": sorted({r["backend"] for r in rows}), "detect_stage_ms": detect,
        "read_to_start_ms": summarise([float(r["read_to_start_ms"]) for r in usable]),
        "first_zone_to_start_ms": summarise([float(r["first_zone_to_start_ms"]) for r in usable]),
        "under_vlm_load": {
            "alarms": len(loaded),
            "read_to_start_ms": summarise([float(r["read_to_start_ms"]) for r in loaded]),
            "first_zone_to_start_ms": summarise([float(r["first_zone_to_start_ms"]) for r in loaded])},
        "vlm_calls": {"count": len(spy.spans), "ok": sum(s[2] for s in spy.spans),
                      "median_s": pct([b - a for a, b, _ in spy.spans], .5)} if spy else None,
        "gpu_samples_util_memused_memtotal": gpu[:40],
        "_read": [float(r["read_to_start_ms"]) for r in usable],
        "_first": [float(r["first_zone_to_start_ms"]) for r in usable],
        "_read_loaded": [float(r["read_to_start_ms"]) for r in loaded],
        "_first_loaded": [float(r["first_zone_to_start_ms"]) for r in loaded]}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("farmguard").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--repeats", type=int, default=40)
    ap.add_argument("--provider", default="ollama")
    ap.add_argument("--model", default=None, help="detector weights (default: settings.yaml)")
    ap.add_argument("--alarm-backend", default="auto", help="auto plays real audio; set none for a silent run")
    ap.add_argument("--drain-timeout", type=float, default=600)
    ap.add_argument("--stand-in-load", action="store_true", help="GPU stress + held mock calls; not a real VLM")
    args = ap.parse_args()
    if args.stand_in_load and args.provider != "mock":
        ap.error("--stand-in-load is only for --provider mock")
    if args.model is None:
        args.model = load_config().get("detector.model")
    (ROOT / "results").mkdir(exist_ok=True)
    base = run_arm("no_agent", args, False)
    agent = run_arm("agent", args, True)
    from scipy.stats import mannwhitneyu
    tests = {}
    for key, a, b in (("read_to_start_ms_all", base["_read"], agent["_read"]),
                      ("first_zone_to_start_ms_all", base["_first"], agent["_first"]),
                      ("read_to_start_ms_under_load", base["_read"], agent["_read_loaded"]),
                      ("first_zone_to_start_ms_under_load", base["_first"], agent["_first_loaded"])):
        if a and b:
            tests[key] = {"p": float(mannwhitneyu(a, b, alternative="two-sided").pvalue), "n": [len(a), len(b)]}
    arms = [{k: v for k, v in x.items() if not k.startswith("_")} for x in (base, agent)]
    real_vlm = args.provider != "mock"
    out = ROOT / "results" / ("e1_load.json" if real_vlm else "e1_load_standin.json")
    result = {"real_vlm": real_vlm, "clip": args.clip, "repeats": args.repeats, "arms": arms,
              "mann_whitney": tests}
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
