"""Utilities: placeholder deterrent sounds, zone drawing, synthetic demo video."""
from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import cv2
import numpy as np

SR = 22050


def _write_wav(path: Path, x: np.ndarray) -> None:
    x = np.clip(x / (np.max(np.abs(x)) + 1e-9) * 0.9, -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def make_sounds(out_dir: str | Path, seconds: float = 4.0, seed: int = 0) -> list[Path]:
    """Synthesised PLACEHOLDERS so the pipeline runs. Replace with real recordings (bee swarm,
    tiger roar, boar alarm call...) from a licensed library before any field test; synthetic
    tones are not claimed to deter anything."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    noise = rng.standard_normal(t.size)

    def env(rate, duty=0.5):
        return ((t * rate) % 1.0 < duty).astype(float)

    sounds = {
        "bee_swarm": sum(np.sign(np.sin(2 * np.pi * f * t)) for f in (220, 233, 247, 262)) *
                     (0.6 + 0.4 * np.sin(2 * np.pi * 7 * t)),
        "tiger_roar": np.convolve(noise, np.ones(60) / 60, "same") * np.sin(2 * np.pi * 0.35 * t) ** 2 *
                      (1 + np.sin(2 * np.pi * 90 * t)),
        "leopard_growl": np.convolve(noise, np.ones(40) / 40, "same") * env(1.6, 0.4) *
                         (1 + np.sin(2 * np.pi * 120 * t)),
        "gunshot": noise * np.exp(-((t % 1.3) * 25)),
        "dog_bark": np.sin(2 * np.pi * 520 * t * (1 - 0.3 * (t % 0.45))) * env(2.2, 0.18) +
                    0.3 * noise * env(2.2, 0.18),
        "boar_alarm": np.sin(2 * np.pi * 140 * t) * env(3.0, 0.25) * (1 + 0.5 * noise),
        "siren": np.sin(2 * np.pi * (700 + 400 * np.sin(2 * np.pi * 0.5 * t)) * t),
        "drum_beat": np.sin(2 * np.pi * 80 * t) * np.exp(-((t % 0.5) * 18)),
        "human_shout": (np.sin(2 * np.pi * 300 * t) + 0.5 * np.sin(2 * np.pi * 900 * t)) * env(0.8, 0.6),
    }
    paths = []
    for name, x in sounds.items():
        p = out / f"{name}.wav"
        _write_wav(p, np.asarray(x, dtype=float))
        paths.append(p)
    return paths


def draw_zone(video: str, out_json: str | Path, width: int = 960) -> None:
    """Click polygon points on the first frame; Enter saves, Backspace undoes, Esc quits."""
    cap = cv2.VideoCapture(int(video) if video.isdigit() else video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read {video}")
    if width and frame.shape[1] > width:
        s = width / frame.shape[1]
        frame = cv2.resize(frame, (width, int(frame.shape[0] * s)))
    pts: list[tuple[int, int]] = []

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))

    cv2.namedWindow("draw field zone")
    cv2.setMouseCallback("draw field zone", on_mouse)
    while True:
        img = frame.copy()
        if pts:
            cv2.polylines(img, [np.array(pts, np.int32)], len(pts) > 2, (0, 200, 255), 2)
        cv2.putText(img, "click corners | Enter=save  Backspace=undo  Esc=quit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("draw field zone", img)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10) and len(pts) >= 3:
            break
        if k == 8 and pts:
            pts.pop()
        if k == 27:
            cv2.destroyAllWindows()
            return
    cv2.destroyAllWindows()
    Path(out_json).write_text(json.dumps({"frame_size": [frame.shape[1], frame.shape[0]],
                                          "zones": [{"name": "field", "points": pts,
                                                     "target_species": []}]}, indent=2))


def make_demo(out_dir: str | Path, seconds: float = 40, fps: int = 10, w: int = 960,
              h: int = 540, ignores_first: bool = True) -> dict[str, str]:
    """Synthetic demo: a brown 'boar' blob walks into the field, eats, and leaves later.
    If ignores_first, it stays through the first alarm's observation window so the agent has to
    retry. Writes demo.mp4 + demo_annotations.json (MockDetector) + demo_zones.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vpath = out / "demo.mp4"
    vw = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    ann: dict[str, list] = {}
    n = int(seconds * fps)
    enter, leave_start = 3.0, (24.0 if ignores_first else 10.0)
    rng = np.random.default_rng(0)
    sx, sy = w / 960, h / 540
    for i in range(n):
        t = i / fps
        img = np.zeros((h, w, 3), np.uint8)
        img[:] = (40, 110, 60)
        img[: h // 3] = (170, 140, 100)
        for r in range(0, w, max(10, int(40 * sx))):  # crop rows
            cv2.line(img, (r, h // 3), (r + int(60 * sx), h), (30, 140, 50), 2)
        if t < enter:
            x = None
        elif t < enter + 3:
            x = int(-80 + (t - enter) / 3 * 380)
        elif t < leave_start:
            x = 300 + int(10 * math.sin(t))
        elif t < leave_start + 4:
            x = int(300 + (t - leave_start) / 4 * 800)
        else:
            x = None
        if x is not None:
            x = int(x * sx)
        if x is not None and -100 * sx < x < w:
            y = int(360 * sy)
            cv2.ellipse(img, (x, y), (int(70 * sx), int(40 * sy)), 0, 0, 360, (40, 60, 90), -1)
            cv2.circle(img, (x + int(65 * sx), y - int(10 * sy)), int(22 * sy), (35, 50, 80), -1)
            x1, y1 = max(0, x - int(75 * sx)), y - int(45 * sy)
            x2, y2 = min(w - 1, x + int(90 * sx)), y + int(45 * sy)
            if x2 - x1 > 20 * sx:
                ann[str(i)] = [{"xyxy": [x1, y1, x2, y2], "label": "wild boar",
                                "conf": round(0.8 + 0.1 * rng.random(), 2), "track_id": 1}]
        cv2.putText(img, f"t={t:5.1f}s", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        vw.write(img)
    vw.release()
    apath = out / "demo_annotations.json"
    apath.write_text(json.dumps({"frames": ann}))
    zpath = out / "demo_zones.json"
    zpath.write_text(json.dumps({"frame_size": [w, h], "zones": [
        {"name": "field", "points": [[0, h // 3], [w, h // 3], [w, h], [0, h]], "target_species": []}]}))
    return {"video": str(vpath), "annotations": str(apath), "zones": str(zpath)}
