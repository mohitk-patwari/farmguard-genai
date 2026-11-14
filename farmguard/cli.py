"""Command line: `farmguard <command>` (or `python -m farmguard.cli <command>`)."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from farmguard.config import load_config


def _cfg(args, extra: dict | None = None):
    ov = dict(extra or {})
    if getattr(args, "provider", None):
        ov["agent.provider"] = args.provider
    if getattr(args, "policy", None) and isinstance(args.policy, str):
        ov["agent.policy.type"] = args.policy
    return load_config(args.config, overrides=ov)


def cmd_run(args) -> None:
    from farmguard.fast.video_source import VideoSource
    from farmguard.system import build_system, run_video
    ov = {}
    if args.no_agent:
        ov["agent.enabled"] = False
    if args.mock_annotations:
        ov["detector.backend"] = "mock"
        ov["detector.mock_annotations"] = args.mock_annotations
    if args.zones:
        ov["zones.file"] = args.zones
    if args.language:
        ov["notifier.language"] = args.language
    if args.mute:
        ov["alarm.backend"] = "none"
    cfg = _cfg(args, ov)
    src = VideoSource(args.source or cfg.get("video.source"), live=args.live,
                      target_fps=cfg.get("video.target_fps", 10),
                      resize_width=cfg.get("video.resize_width", 960))
    system = build_system(cfg)
    run_video(system, src.frames(), realtime=not args.fast, max_frames=args.max_frames)
    src.release()
    print(json.dumps({"events": len(system.fast.events),
                      "vlm": system.agent.vlm_counts() if system.agent else None,
                      "agent": system.agent.run_counts() if system.agent else None,
                      "stage_timing_ms": system.fast.timer.summary(),
                      "results": [{"event": r.event["event_id"], "final": r.final_status,
                                   "detector_species": r.detector_species, "vlm_species": r.vlm_species,
                                   "vlm_confidence": r.vlm_confidence, "effective_species": r.effective_species,
                                   "overridden": r.overridden, "possible_false_alarm": r.possible_false_alarm,
                                   "species_uncertain": r.species_uncertain,
                                   "effective_by_threshold": r.effective_by_threshold,
                                   "sounds": [a.sound for a in r.attempts],
                                   "farmer": (r.farmer or {}).get("message")}
                                  for r in (system.agent.results if system.agent else [])]},
                     indent=2, ensure_ascii=False))


def cmd_simulate(args) -> None:
    from farmguard.agent.llm.providers import build_client
    from farmguard.eval.experiments import e4_simulation
    cfg = _cfg(args)
    policies = args.policies.split(",")
    client = build_client(cfg, "llm") if "llm" in policies else None
    out = e4_simulation(cfg, args.sim, policies, client, args.repeats)
    print(json.dumps(out["summary"], indent=2))
    print(json.dumps(out["paired_tests"], indent=2))


def cmd_eval(args) -> None:
    from farmguard.agent.llm.providers import build_client
    from farmguard.eval import experiments as E
    cfg = _cfg(args)
    if args.exp == "e1":
        out = E.e1_latency(args.timing or cfg.path("logging.timing_csv"))
    elif args.exp == "e2":
        out = E.e2_vlm(cfg, build_client(cfg, "vlm"), args.labels or "data/labels/vlm_samples.csv",
                       limit=args.limit)
    elif args.exp == "e3":
        out = E.e3_retreat(build_client(cfg, "vlm"), args.labels or "data/labels/retreat_samples.csv",
                           limit=args.limit)
    elif args.exp == "e5":
        out = E.e5_false_alarms(cfg.path("logging.events_jsonl"),
                                args.labels or "data/labels/intrusions.csv", args.hours)
    elif args.exp == "e6":
        out = str(E.e6_alert_samples(cfg, build_client(cfg, "llm"), args.limit or 20))
    else:
        raise SystemExit(f"unknown experiment {args.exp}")
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="farmguard", description=__doc__)
    p.add_argument("--config", default=None, help="settings.yaml path")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the full system on a video / webcam / RTSP stream")
    r.add_argument("--source")
    r.add_argument("--live", action="store_true", help="drop stale frames (cameras)")
    r.add_argument("--fast", action="store_true", help="do not pace a file to real time")
    r.add_argument("--no-agent", action="store_true", help="fast path only (E1 baseline)")
    r.add_argument("--provider", help="ollama|gemini|anthropic|openai_compat|mock")
    r.add_argument("--policy", help="llm|bandit|rule_chain|random|fixed")
    r.add_argument("--mock-annotations", help="use MockDetector with this annotation JSON")
    r.add_argument("--zones")
    r.add_argument("--language", help="hi|mr|en")
    r.add_argument("--max-frames", type=int)
    r.add_argument("--mute", action="store_true",
                   help="no audio; alarms still logged and timed (same as FARMGUARD_MUTE=1)")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("simulate", help="E4: compare deterrent policies in the simulator")
    s.add_argument("--sim", default="config/sim.yaml")
    s.add_argument("--policies", default="fixed,random,rule_chain,bandit")
    s.add_argument("--repeats", type=int, default=3)
    s.add_argument("--provider")
    s.set_defaults(fn=cmd_simulate)

    e = sub.add_parser("eval", help="run an experiment: e1 e2 e3 e5 e6")
    e.add_argument("exp")
    e.add_argument("--labels")
    e.add_argument("--timing")
    e.add_argument("--limit", type=int)
    e.add_argument("--hours", type=float, default=1.0, help="hours of video (E5)")
    e.add_argument("--provider")
    e.set_defaults(fn=cmd_eval)

    m = sub.add_parser("make-sounds", help="write placeholder deterrent WAVs into sounds/")
    m.add_argument("--out", default="sounds")
    m.set_defaults(fn=lambda a: print("\n".join(map(str, __import__("farmguard.tools", fromlist=["x"]).make_sounds(a.out)))))

    z = sub.add_parser("draw-zone", help="click the field polygon on the first frame")
    z.add_argument("video")
    z.add_argument("--out", default="config/zones.json")
    z.set_defaults(fn=lambda a: __import__("farmguard.tools", fromlist=["x"]).draw_zone(a.video, a.out))

    d = sub.add_parser("make-demo", help="synthetic demo video + mock annotations + zones")
    d.add_argument("--out", default="data/demo")
    d.set_defaults(fn=lambda a: print(json.dumps(__import__("farmguard.tools", fromlist=["x"]).make_demo(a.out), indent=2)))

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s")
    try:
        args.fn(args)
    except KeyboardInterrupt:
        # os._exit: the agent's VLM pool threads are not daemons, and a normal exit would join them
        # mid-LLM call (uninterruptible on Windows). Memory and logs are already written per event.
        logging.getLogger(__name__).info("Interrupted")
        sys.stdout.flush()
        os._exit(130)


if __name__ == "__main__":
    main(sys.argv[1:])
