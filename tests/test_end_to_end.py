"""Full system on a synthetic video with MockDetector + MockClient (no GPU, no network)."""
from __future__ import annotations


from farmguard.agent.llm.providers import MockClient
from farmguard.config import load_config
from farmguard.fast.video_source import VideoSource
from farmguard.sim.simulator import load_sim, run_all, summarize
from farmguard.system import build_system, run_video
from farmguard.tools import make_demo, make_sounds


def test_demo_end_to_end(tmp_path):
    make_sounds(tmp_path / "sounds", seconds=0.5)
    demo = make_demo(tmp_path / "demo", seconds=30, fps=10, w=320, h=180, ignores_first=True)
    cfg = load_config(overrides={
        "detector.backend": "mock", "detector.mock_annotations": demo["annotations"],
        "zones.file": demo["zones"], "alarm.backend": "none",
        "alarm.sounds_dir": str(tmp_path / "sounds"), "agent.provider": "mock",
        "agent.retreat.observe_seconds": 6, "agent.retreat.gone_frames": 10,
        "video.buffer_seconds": 120, "video.resize_width": 0, "motion_gate.enabled": False,
        "memory.db_path": str(tmp_path / "m.sqlite"), "logging.events_jsonl": str(tmp_path / "ev.jsonl"),
        "logging.timing_csv": str(tmp_path / "t.csv"), "notifier.language": "en"})
    system = build_system(cfg, llm_client=MockClient())
    src = VideoSource(demo["video"], live=False, target_fps=10)
    run_video(system, src.frames(), realtime=False)
    src.release()
    assert len(system.fast.events) == 1
    res = system.agent.results[0]
    sounds = [a.sound for a in res.attempts]
    assert len(sounds) >= 2, "animal ignored the first alarm, so the agent must retry"
    assert len(set(sounds)) == len(sounds), "never repeat a failed sound within an event"
    assert all(s in cfg.species["wild_boar"].allowed_sounds for s in sounds)
    assert res.farmer and res.farmer["message"]
    assert (tmp_path / "t.csv").exists()


def test_simulator_runs():
    cfg = load_config()
    sim = load_sim(cfg.root / "config" / "sim.yaml")
    sim["nights"] = 10
    out = run_all(cfg, sim, ["fixed", "rule_chain", "bandit"], repeats=1)
    s = summarize(out)
    assert set(s) == {"fixed", "rule_chain", "bandit"}
    assert len({v["events"] for v in s.values()}) == 1  # paired: identical event streams
