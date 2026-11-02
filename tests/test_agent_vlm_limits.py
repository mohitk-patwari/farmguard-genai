"""The live smoke test's context overflow: image size and count, num_ctx, and visible VLM failures."""
from __future__ import annotations

import logging

import cv2
import numpy as np

from farmguard.agent.clip_builder import build_event_images
from farmguard.agent.llm.providers import MockClient, OllamaClient, build_client
from farmguard.agent.vlm_check import VLMChecker
from farmguard.config import load_config
from farmguard.events import FramePacket


def _pkts(n=5):
    return [FramePacket(index=i, video_time=i / 10, frame=np.zeros((720, 1280, 3), np.uint8))
            for i in range(n)]


def test_images_are_capped_and_downscaled():
    pkts = _pkts()
    images, layout = build_event_images(pkts[2], pkts, (100, 100, 400, 400))
    assert len(images) == 4 and "2 neighbouring" in layout
    for jpeg in images:
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert max(img.shape[:2]) <= 448


def test_image_cap_is_a_knob():
    pkts = _pkts()
    assert len(build_event_images(pkts[2], pkts, (0, 0, 9, 9), max_images=3)[0]) == 3
    assert len(build_event_images(pkts[2], pkts, (0, 0, 9, 9), max_images=2)[0]) == 2


def test_defaults_are_3b_with_num_ctx():
    cfg = load_config(overrides={"agent.provider": "ollama"})
    client = build_client(cfg, "vlm")
    assert isinstance(client, OllamaClient)
    assert client.model == "qwen2.5vl:3b" and client.num_ctx == cfg.get("agent.ollama.num_ctx") == 4096
    assert build_client(load_config(overrides={"agent.provider": "ollama", "agent.vlm_model": "qwen2.5vl:7b"}),
                        "vlm").model == "qwen2.5vl:7b"


def test_failed_vlm_call_warns(caplog):
    class Down(MockClient):
        def _call(self, *a):
            raise ValueError("request exceeds the available context size")

    with caplog.at_level(logging.WARNING):
        report = VLMChecker(load_config(), Down()).check([b"x"], "one image", "stray_cattle")
    assert report.species == "stray_cattle" and report.error
    assert "VLM check failed" in caplog.text and "context size" in caplog.text
