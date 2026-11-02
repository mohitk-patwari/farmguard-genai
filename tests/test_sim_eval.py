"""Experiment runners and labelling kit, on synthetic frames with mock backends."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from farmguard.agent.llm.providers import MockClient
from farmguard.config import ROOT, load_config
from farmguard.eval import experiments as E
from farmguard.eval import labels as L
from farmguard.fast.detector import MockDetector
from farmguard.tools import make_demo


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(E, "SPLITS", tmp_path / "splits.csv")
    return tmp_path


def _frames(root: Path, clip: str, n: int) -> list[str]:
    d = root / "frames" / clip
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        p = d / f"{i:06d}.jpg"
        cv2.imwrite(str(p), np.full((90, 160, 3), 40 + i, np.uint8))
        out.append(str(p))
    return out


def _csv(path: Path, rows: list[dict]) -> Path:
    L.write_rows(path, rows, list(rows[0]))
    return path


# ------------------------------------------------------------------ splits and labels
def test_split_is_by_clip_and_sticky():
    first = L.assign_splits([f"c{i}" for i in range(10)], test_fraction=0.3)
    assert sum(v["split"] == "test" for v in first.values()) == 3
    second = L.assign_splits([f"c{i}" for i in range(20)], first, test_fraction=0.3)
    assert all(second[k]["split"] == v["split"] for k, v in first.items())


def test_select_filters_by_split_and_rejects_unknown_clip(tmp_path):
    splits = _csv(tmp_path / "s.csv", [{"clip": "a", "split": "dev", "lighting": "day"},
                                       {"clip": "b", "split": "test", "lighting": "night"}])
    rows = [{"frames": "data/frames/a/1.jpg;data/frames/a/2.jpg"}, {"clip": "data/clips/b.mp4"}]
    assert L.select(rows, splits, "dev") == rows[:1]
    assert L.select(rows, splits, "test") == rows[1:]
    assert L.select(rows, splits, "all", "night") == rows[1:]
    with pytest.raises(ValueError):
        L.select([{"frames": "data/frames/ghost/1.jpg"}], splits, "test")


def test_agreement_and_merge(tmp_path):
    a = _csv(tmp_path / "a.csv", [{"sample_id": "1", "species": "x", "behaviour": "e"},
                                  {"sample_id": "2", "species": "y", "behaviour": "e"}])
    b = _csv(tmp_path / "b.csv", [{"sample_id": "1", "species": "x", "behaviour": "e"},
                                  {"sample_id": "2", "species": "x", "behaviour": "e"}])
    assert L.agreement(a, b, ["species"])["species"]["agree"] == 0.5
    assert L.merge(a, b, ["species", "behaviour"], tmp_path / "m.csv", tmp_path / "c.csv") == (1, 1)
    assert L.read_rows(tmp_path / "c.csv")[0]["species_b"] == "x"


def test_extract_frames_matches_pipeline_sampling(tmp_path):
    demo = make_demo(tmp_path / "demo", seconds=3, fps=10, w=160, h=90)
    out = _script("extract_frames").extract(demo["video"], tmp_path / "frames", fps=5, width=0)
    rows = L.read_rows(out / "manifest.csv")
    assert [int(r["frame"]) for r in rows[:3]] == [0, 2, 4]
    assert float(rows[1]["time_s"]) == pytest.approx(0.2)
    assert Path(rows[0]["file"]).exists()


def test_burst_keeps_time_order_at_clip_edges():
    label = _script("label")
    files = [str(i) for i in range(6)]
    assert label.burst(files, 2) == (["0", "2", "4"], 1)
    assert label.burst(files, 0) == (["0", "2"], 0)
    assert label.burst(files, 5) == (["3", "5"], 1)


def test_label_session_writes_documented_format(tmp_path, monkeypatch):
    """Drives the GUI loop with scripted keys. Real window behaviour still needs a human check."""
    label = _script("label")
    monkeypatch.setattr(label, "FRAMES", tmp_path / "frames")
    monkeypatch.setattr(label, "LABELS", tmp_path / "labels")
    files = _frames(tmp_path, "c01", 6)
    _csv(tmp_path / "frames" / "c01" / "manifest.csv",
         [{"frame": i, "file": f, "time_s": i / 10, "clip": "data/clips/c01.mp4"} for i, f in enumerate(files)])
    keys = iter([ord("d"), ord("d"), ord("l"), ord("1"), ord("1"), ord("q")])
    monkeypatch.setattr(cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(cv2, "destroyWindow", lambda *a: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *a: next(keys))
    monkeypatch.setattr(cv2, "selectROI", lambda *a, **k: (10, 20, 30, 40))
    label.label_clip("vlm", "c01", "a", ["wild_boar", "nilgai"])
    row = L.read_rows(tmp_path / "labels" / "vlm_samples.a.csv")[0]
    assert row["sample_id"] == "c01_f2" and row["bbox"] == "10,20,40,60"
    assert row["species"] == "wild_boar" and row["behaviour"] == "eating"
    assert row["frames"].split(";")[int(row["trigger_index"])] == files[2]

    keys2 = iter([ord("["), ord("d"), ord("d"), ord("d"), ord("]"), ord("2"), ord("3"), ord("q")])
    monkeypatch.setattr(cv2, "waitKey", lambda *a: next(keys2))
    label.label_clip("retreat", "c01", "a", ["wild_boar", "nilgai"])
    row = L.read_rows(tmp_path / "labels" / "retreat_samples.a.csv")[0]
    assert row["species"] == "nilgai" and row["status"] == "still_present"
    assert row["frames"].split(";") == files[:4]


# ------------------------------------------------------------------ E2, E3
def test_e2_respects_split_and_restores_location_flag(tmp_path, results):
    cfg = load_config()
    a, b = _frames(tmp_path, "dev1", 3), _frames(tmp_path, "test1", 3)
    samples = _csv(tmp_path / "v.csv", [
        {"sample_id": "d", "frames": ";".join(a), "trigger_index": 1, "bbox": "1,1,50,50",
         "species": "wild_boar", "behaviour": "eating"},
        {"sample_id": "t", "frames": ";".join(b), "trigger_index": 1, "bbox": "",
         "species": "nilgai", "behaviour": "moving_through"}])
    _csv(tmp_path / "splits.csv", [{"clip": "dev1", "split": "dev", "lighting": ""},
                                   {"clip": "test1", "split": "test", "lighting": ""}])
    client = MockClient()
    out = E.e2_vlm(cfg, client, samples, split="test", location_hints=(None, "somewhere"), modes=("crop",))
    assert out["conditions"]["crop|no_hint"]["species"]["n"] == 1  # dev sample stayed out
    assert "crop|no_hint vs crop|hint" in out["paired_tests"]
    assert "somewhere" in client.calls[-1]["prompt"] and "somewhere" not in client.calls[0]["prompt"]
    assert cfg.get("agent.include_location_in_prompt") is False
    saved = json.loads((tmp_path / "results" / "e2_mock-mock_summary.json").read_text(encoding="utf-8"))
    assert saved["provenance"]["provider"] == "mock"


def test_e2_counts_backend_errors(tmp_path, results):
    def dead(p, s, n):
        raise ConnectionError("down")
    frames = _frames(tmp_path, "c", 3)
    samples = _csv(tmp_path / "v.csv", [{"sample_id": "1", "frames": ";".join(frames), "trigger_index": 0,
                                         "bbox": "", "species": "nilgai", "behaviour": "eating"}])
    out = E.e2_vlm(load_config(), MockClient(responder=dead), samples, modes=("crop",), location_hints=(None,))
    assert out["conditions"]["crop|no_hint"]["n_errors"] == 1


def test_e3_scores_tracker_vlm_and_fused_on_labels(tmp_path, results):
    left = _frames(tmp_path, "c1", 6)
    stayed = _frames(tmp_path, "c2", 6)
    samples = _csv(tmp_path / "r.csv", [{"sample_id": "left", "frames": ";".join(left), "species": "wild_boar", "status": "gone"},
                                        {"sample_id": "stay", "frames": ";".join(stayed), "species": "wild_boar", "status": "still_present"}])
    box = [{"xyxy": [10, 10, 50, 50], "label": "wild boar", "track_id": 1}]
    cfg = load_config()
    scripts = iter([{"0": box}, {str(i): box for i in range(6)}])  # left: box in frame 0 only; stay: all

    def factory():
        return MockDetector({"frames": next(scripts)}, cfg.species_for_label)

    vlm = MockClient(responder=lambda p, s, n: {"observation": "", "status": "still_present",
                                                "animals_visible": 1, "confidence": "low"})
    out = E.e3_retreat(vlm, samples, detector_factory=factory, gone_frames=3)
    c = out["conditions"]
    assert c["tracker"]["left_vs_stayed"]["accuracy"] == 1.0
    assert c["vlm"]["left_vs_stayed"]["accuracy"] == 0.5     # always says still_present
    assert c["fused"]["four_way"]["prediction_counts"].get("uncertain") == 1  # gone + weak VLM = uncertain
    assert "fused vs tracker" in out["paired_tests"]


# ------------------------------------------------------------------ E1, E5 through the real system
def _demo_overrides(tmp_path, demo) -> dict:
    return {"detector.backend": "mock", "detector.mock_annotations": demo["annotations"],
            "zones.file": demo["zones"], "alarm.backend": "none", "agent.provider": "mock",
            "agent.retreat.observe_seconds": 1, "agent.retreat.gone_frames": 2, "video.buffer_seconds": 120,
            "video.resize_width": 0, "motion_gate.enabled": False, "notifier.language": "en"}


def test_e1_runs_both_conditions(tmp_path, results, monkeypatch):
    demo = make_demo(tmp_path / "demo", seconds=8, fps=10, w=160, h=90, ignores_first=False)
    out = E.e1_run([demo["video"]], _demo_overrides(tmp_path, demo), llm_client=MockClient())
    assert out["read_to_request_ms"]["agent_on"]["n"] == 1 == out["read_to_request_ms"]["agent_off"]["n"]
    assert out["detect_ms_per_frame"]["agent_on"]["n"] > 0
    assert (tmp_path / "results" / "e1_alarms.csv").exists()


def test_e5_counts_false_alarm_only_within_its_own_clip(tmp_path, results):
    demo = make_demo(tmp_path / "demo", seconds=8, fps=10, w=160, h=90, ignores_first=False)
    # the truth interval belongs to ANOTHER clip, so the demo's trigger is a false alarm
    truth = _csv(tmp_path / "i.csv", [{"clip": "data/clips/other.mp4", "start_s": 0, "end_s": 99, "species": "wild_boar"}])
    client = MockClient(responder=lambda p, s, n: {"observation": "", "species": "none", "count": 0,
                                                   "behaviour": "absent", "confidence": "high", "note": ""})
    out = E.e5_run([demo["video"]], truth, _demo_overrides(tmp_path, demo), client)
    assert (out["triggers"], out["false_alarms_before"], out["false_alarms_after"]) == (1, 1, 0)
    assert out["intervals"] == 0 and out["video_hours"] == pytest.approx(8 / 3600, rel=0.05)


def test_e5_score_measures_recall_cost_of_gate():
    truth = [{"clip": "a.mp4", "start_s": "10", "end_s": "20", "species": "x"}]
    trig = [{"clip": "a.mp4", "video_time": 12.0, "vlm_species": "unclassifiable"},
            {"clip": "b.mp4", "video_time": 12.0, "vlm_species": "wild_boar"}]
    out = E._e5_score(trig, truth, 2.0, 5.0)
    assert out["true_events_rejected_by_gate"] == 1 and out["intervals_found_before"] == 1
    assert out["intervals_found_after"] == 0 and out["fa_per_hour_after"] == 0.5


def test_e5_single_file_mode_refuses_ambiguous_clips(tmp_path):
    t = _csv(tmp_path / "i.csv", [{"clip": "a.mp4", "start_s": 0, "end_s": 1, "species": "x"},
                                  {"clip": "b.mp4", "start_s": 0, "end_s": 1, "species": "x"}])
    with pytest.raises(ValueError):
        E.e5_false_alarms(tmp_path / "none.jsonl", t, 1.0)


# ------------------------------------------------------------------ E6
def _alert(prompt, schema, n):
    f = json.loads(prompt[prompt.index("{"):prompt.rindex("}") + 1])  # echoes the facts, so it passes the grounding check
    return {"message": " ".join([f["time"], f["animal"], str(f["count"]), *f["sounds_played"], f["result"]]),
            "voice_text": "v"}


def test_e6_sheet_is_blind_and_scoring_is_paired(tmp_path, results):
    cfg = load_config()
    sheet = E.e6_alert_samples(cfg, MockClient(responder=_alert), n=3)
    assert cfg.get("notifier.language") == "hi"
    assert sheet.read_bytes().startswith(b"\xef\xbb\xbf")  # BOM so Excel renders Devanagari
    head = L.read_rows(sheet)[0]
    assert "variant" not in head and "variant_hidden" not in head and "case" not in head
    key = {r["message_id"]: r for r in L.read_rows(tmp_path / "results" / "e6_key.csv")}
    assert {r["variant"] for r in key.values()} == {"llm", "template"}

    files = []
    for rater, boost in (("r1", 1), ("r2", 1), ("r3", 2)):
        rows = []
        for r in L.read_rows(sheet):
            llm = key[r["message_id"]]["variant"] == "llm"
            rows.append({**r, "understood_yes_no": "yes" if llm else "no", "facts_correct_yes_no": "y",
                         "clarity_1_5": 2 + (boost if llm else 0)})
        files.append(_csv(tmp_path / f"{rater}.csv", rows))
    out = E.e6_score(files, tmp_path / "results" / "e6_key.csv")
    assert out["raters"] == 3
    assert out["by_variant"]["llm"]["understood_pct"] == 100.0
    assert out["by_variant"]["template"]["understood_pct"] == 0.0
    assert out["by_variant"]["llm"]["clarity_mean"] > out["by_variant"]["template"]["clarity_mean"]
    assert out["clarity_wilcoxon"]["n"] == 3


def test_e6_llm_failure_drops_the_pair(tmp_path, results):
    sheet = E.e6_alert_samples(load_config(), MockClient(), n=2)  # default mock gives no message
    key = L.read_rows(tmp_path / "results" / "e6_key.csv")
    assert {r["variant"] for r in key} == {"template_fallback", "template"}
    rows = [{**r, "understood_yes_no": "y", "facts_correct_yes_no": "y", "clarity_1_5": 3} for r in L.read_rows(sheet)]
    out = E.e6_score([_csv(tmp_path / "r.csv", rows)], tmp_path / "results" / "e6_key.csv")
    assert out["by_variant"]["llm"]["n"] == 0 and out["cases_lost_to_llm_failure"] == 4


# ------------------------------------------------------------------ figures and tables
def test_figures_make_paper_inputs_and_refuse_mock_or_fallback(tmp_path, results):
    pytest.importorskip("matplotlib")
    figures = _script("figures_all")
    cfg = load_config()
    sim = tmp_path / "sim.yaml"
    sim.write_text((ROOT / "config" / "sim.yaml").read_text(encoding="utf-8").replace("nights: 60", "nights: 10"),
                   encoding="utf-8")
    res, paper = tmp_path / "results", tmp_path / "paper"
    figs, tables = paper / "figures", paper / "tables"
    figs.mkdir(parents=True), tables.mkdir(parents=True)

    E.e4_simulation(cfg, sim, ["fixed", "bandit", "llm"], MockClient(), 2, "random")   # mock LLM
    figures.main(["--results", str(res), "--paper", str(paper)])
    assert (figs / "feature_gap.pdf").exists() and not (tables / "e4_random.tex").exists()

    def dead(p, s, n):
        raise ConnectionError("down")
    E.e4_simulation(cfg, sim, ["fixed", "bandit", "llm"], MockClient(responder=dead), 2, "random")  # all fallback
    figures.main(["--results", str(res), "--paper", str(paper)])
    assert not (tables / "e4_random.tex").exists()

    E.e4_simulation(cfg, sim, ["fixed", "random", "rule_chain", "bandit"], None, 2, "random")
    E.e4_simulation(cfg, sim, ["fixed", "random", "rule_chain", "bandit"], None, 2, "informed")
    figures.main(["--results", str(res), "--paper", str(paper)])
    tex = (tables / "e4_random.tex").read_text(encoding="utf-8")
    assert "assumptions, not field data" in tex and r"\begin{tabular}" in tex
    assert (figs / "e4_learning_curves.pdf").exists() and (figs / "e4_time_in_field.pdf").exists()
    assert "Smallest possible p with 2 runs: 0.5" in (tables / "e4_random_tests.tex").read_text(encoding="utf-8")


# ------------------------------------------------------------------ entry point
def test_main_e4_writes_results_and_mock_is_not_logged(tmp_path, results, monkeypatch):
    from farmguard.eval import __main__ as cli
    log = tmp_path / "RESULTS.md"
    monkeypatch.setattr(cli, "RESULTS_LOG", log)
    sim = tmp_path / "sim.yaml"
    sim.write_text((ROOT / "config" / "sim.yaml").read_text(encoding="utf-8").replace("nights: 60", "nights: 4"),
                   encoding="utf-8")
    out = cli.main(["e4", "--provider", "mock", "--policies", "fixed,llm", "--repeats", "2", "--mode", "random",
                    "--sim", str(sim)])
    assert "mock" in " ".join(out["warnings"])
    assert not log.exists()
    with (tmp_path / "results" / "e4_random_outcomes.csv").open(encoding="utf-8") as fh:
        assert {r["policy"] for r in csv.DictReader(fh)} == {"fixed", "llm"}
    cli.main(["e4", "--policies", "fixed,bandit", "--repeats", "2", "--mode", "informed", "--sim", str(sim)])
    logged = log.read_text(encoding="utf-8")
    assert "bandit:" in logged and "fixed:" in logged


def test_main_preflight_stops_on_dead_backend(monkeypatch, results):
    from farmguard.agent.llm import providers
    from farmguard.eval import __main__ as cli

    def dead(p, s, n):
        raise ConnectionError("down")
    monkeypatch.setattr(providers, "build_client", lambda cfg, role="vlm": MockClient(responder=dead))
    with pytest.raises(SystemExit, match="not usable"):
        cli.main(["e4", "--provider", "mock", "--policies", "llm", "--no-log"])
