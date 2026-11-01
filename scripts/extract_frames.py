"""Cut clips into frames for labelling: data/frames/<clip stem>/<index>.jpg + manifest.csv.

Uses the pipeline's own VideoSource, so frame sampling and resizing match what the system sees and
hand-drawn boxes line up with detector boxes. OpenCV only, no ffmpeg needed.

  python scripts/extract_frames.py data/clips/*.mp4
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import cv2

from farmguard.config import load_config
from farmguard.fast.video_source import VideoSource


def extract(clip: str, out_root: str | Path = "data/frames", fps: float = 10, width: int = 960) -> Path:
    out = Path(out_root) / Path(clip).stem
    out.mkdir(parents=True, exist_ok=True)
    src = VideoSource(clip, target_fps=fps, resize_width=width)
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "file", "time_s", "clip"])
        for idx, t, frame, _ in src.frames():
            path = out / f"{idx:06d}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            w.writerow([idx, path.as_posix(), round(t, 3), Path(clip).as_posix()])
    src.release()
    return out


if __name__ == "__main__":
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clips", nargs="+")
    p.add_argument("--out", default="data/frames")
    p.add_argument("--fps", type=float, default=cfg.get("video.target_fps", 10))
    p.add_argument("--width", type=int, default=cfg.get("video.resize_width", 960))
    args = p.parse_args()
    for c in sorted({m for pat in args.clips for m in glob.glob(pat)}):
        print(extract(c, args.out, args.fps, args.width))
