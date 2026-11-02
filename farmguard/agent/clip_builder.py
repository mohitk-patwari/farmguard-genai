"""Builds the images the VLM sees for one event (AlexKapy camera-trap-ai-pipeline pattern):
  1. a padded crop of the animal (15-25% padding, auto-contrast, upscaled to >= 512 px)
  2. the full trigger frame with the box drawn in red (context)
  3. neighbouring frames (burst / short sequence)
Accuracy-when-committed rose from 65.0% (crop only) to 80.9% (crop + full frame + burst) there.
"""
from __future__ import annotations

import cv2
import numpy as np

from farmguard.events import FramePacket

# Qwen2.5-VL's image tokens scale with pixels. 6 images at 768 px were ~6500 tokens against Ollama's 4096.
MAX_SIDE = 448
MAX_IMAGES = 4  # crop + full frame + 2 neighbours


def to_jpeg(img: np.ndarray, max_side: int = MAX_SIDE, quality: int = 85) -> bytes:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def enhance(img: np.ndarray) -> np.ndarray:
    """CLAHE on the luminance channel: helps night / low-contrast crops."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def padded_crop(frame: np.ndarray, xyxy, padding: float = 0.25, min_short_edge: int = 512) -> np.ndarray:
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    pw, ph = (x2 - x1) * padding, (y2 - y1) * padding
    x1, y1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
    x2, y2 = min(W, int(x2 + pw)), min(H, int(y2 + ph))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return frame
    crop = enhance(crop)
    short = min(crop.shape[:2])
    if short < min_short_edge:
        s = min_short_edge / short
        crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)),
                          interpolation=cv2.INTER_CUBIC)
    return crop


def frame_with_box(frame: np.ndarray, xyxy) -> np.ndarray:
    img = frame.copy()
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), max(2, img.shape[1] // 300))
    return img


def build_event_images(trigger_pkt: FramePacket, neighbours: list[FramePacket], bbox,
                       padding: float = 0.25, max_side: int = MAX_SIDE,
                       mode: str = "crop+full+burst", max_images: int = MAX_IMAGES) -> tuple[list[bytes], str]:
    """mode: crop | full | crop+full | crop+full+burst  (E2 ablation). Returns (jpegs, layout)."""
    images: list[bytes] = []
    layout: list[str] = []
    if "crop" in mode:
        images.append(to_jpeg(padded_crop(trigger_pkt.frame, bbox, padding), max_side))
        layout.append("a close-up crop of the detected animal")
    if "full" in mode:
        images.append(to_jpeg(frame_with_box(trigger_pkt.frame, bbox), max_side))
        layout.append("the full frame with the animal boxed in red")
    if "burst" in mode:
        others = [p for p in neighbours if p.index != trigger_pkt.index]
        others = sample_frames(others, max(0, max_images - len(images)), last_heavy=False)
        for p in others:
            images.append(to_jpeg(p.frame, max_side))
        if others:
            layout.append(f"{len(others)} neighbouring frames in time order")
    return images, "; ".join(layout)


def sample_frames(pkts: list[FramePacket], n: int, last_heavy: bool = True) -> list[FramePacket]:
    """Pick n frames from a window; with last_heavy the final frames are always included."""
    if len(pkts) <= n:
        return pkts
    if last_heavy:
        tail = pkts[-(n // 2):]
        head_pool = pkts[: -(n // 2)]
        k = n - len(tail)
        idx = np.linspace(0, len(head_pool) - 1, k).round().astype(int)
        return [head_pool[i] for i in idx] + tail
    idx = np.linspace(0, len(pkts) - 1, n).round().astype(int)
    return [pkts[i] for i in idx]
