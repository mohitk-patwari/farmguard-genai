"""Compare ByteTrack settings on real clips. Run after the detector changes:

    python scripts/tune_tracker.py --clips data/clips/boar__*.webm data/clips/cattle__*

No ground-truth identities exist yet, so this reports fragmentation proxies, not MOTA. Fewer tracks is
only better if coverage (frames with a tracked animal) holds. Writes results/tracker_tuning.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import tempfile
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from farmguard.config import load_config
from farmguard.fast.detector import UltralyticsDetector
from farmguard.fast.trigger import Trigger

BASE = {"tracker_type": "bytetrack", "track_high_thresh": 0.25, "track_low_thresh": 0.1,
        "new_track_thresh": 0.25, "track_buffer": 30, "match_thresh": 0.8, "fuse_score": True}
GRID = {"default": {},
        "buffer60": {"track_buffer": 60},
        "new0.4": {"new_track_thresh": 0.4},
        "new0.4_buffer60": {"new_track_thresh": 0.4, "track_buffer": 60},
        "new0.4_buffer60_match0.9": {"new_track_thresh": 0.4, "track_buffer": 60, "match_thresh": 0.9},
        "new0.5_buffer60_match0.9": {"new_track_thresh": 0.5, "track_buffer": 60, "match_thresh": 0.9}}


def score_clip(clip: str, det: UltralyticsDetector, fps: float, max_frames: int) -> dict:
    cap = cv2.VideoCapture(clip)
    step = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 25) / fps))
    # Species cooldown off: this counts what fragmentation alone would do.
    trig = Trigger(window=7, min_hits=4, track_cooldown_s=60)
    lengths: dict[int, int] = {}
    i = n = covered = events = 0
    labels: dict[str, int] = {}
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            if frame.shape[1] > 960:
                frame = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))
            dets = [d for d in det.detect(frame, i) if d.species]
            covered += any(d.track_id is not None for d in dets)
            for d in dets:
                labels[d.species] = labels.get(d.species, 0) + 1
                if d.track_id is not None:
                    lengths[d.track_id] = lengths.get(d.track_id, 0) + 1
            events += len(trig.update([(d, "field") for d in dets], {Trigger.key_for(d) for d in dets}, n / fps))
            n += 1
        i += 1
    cap.release()
    return {"frames": n, "covered": covered, "tracks": len(lengths),
            "short_tracks": sum(v < 5 for v in lengths.values()), "events_no_species_cooldown": events,
            "labels": labels}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--device", default="0")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    cfg = load_config()
    weights = Path(a.model) if a.model else cfg.path("detector.model")
    clips = [c for pat in a.clips for c in glob.glob(pat)] or a.clips
    out: dict = {"model": weights.name, "note": "proxies only, no identity ground truth", "configs": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for name, change in GRID.items():
            path = Path(tmp) / f"{name}.yaml"
            path.write_text(yaml.safe_dump({**BASE, **change}), encoding="utf-8")
            per_clip = {}
            for clip in clips:  # a fresh detector per clip: track ids must not carry over
                det = UltralyticsDetector(weights, confidence=cfg.get("detector.confidence"),
                                          track_conf=cfg.get("detector.track_conf"), device=a.device,
                                          tracker=str(path), species_mapper=cfg.species_for_label,
                                          imgsz=cfg.get("detector.imgsz"))
                per_clip[Path(clip).name] = score_clip(clip, det, a.fps, a.max_frames)
                del det
            totals = {k: sum(c[k] for c in per_clip.values())
                      for k in ("frames", "covered", "tracks", "short_tracks", "events_no_species_cooldown")}
            out["configs"][name] = {"settings": change, "totals": totals, "clips": per_clip}
            print(name, json.dumps(totals), flush=True)
    dest = ROOT / "results" / "tracker_tuning.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(dest)


if __name__ == "__main__":
    main()
