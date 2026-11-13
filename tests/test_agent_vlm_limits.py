"""The live smoke test's context overflow: image size and count, num_ctx, and visible VLM failures."""
from __future__ import annotations

import logging

import cv2
import numpy as np

from farmguard.agent.clip_builder import OLLAMA_TOKENS_PER_IMAGE, build_event_images
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
    assert len(images) == 3 and "1 neighbouring" in layout
    for jpeg in images:
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert max(img.shape[:2]) <= 448


def test_default_image_count_fits_num_ctx():
    """Smoke test 2: 4 images (4310-4435 tokens) were all rejected at num_ctx 4096. Leave room for the
    prompt text (~400 tokens with the VLM prompt) and the JSON answer (~200)."""
    cfg = load_config()
    n = cfg.get("agent.clip.max_images")
    assert n * OLLAMA_TOKENS_PER_IMAGE + 600 <= cfg.get("agent.ollama.num_ctx")


def test_live_system_uses_the_image_cap_and_enum_flag(tmp_path):
    from farmguard.fast.detector import MockDetector
    from farmguard.system import build_system
    cfg = load_config(overrides={"agent.policy.constrain_enum": False, "agent.clip.max_images": 2,
                                 "alarm.backend": "none", "memory.db_path": str(tmp_path / "m.sqlite"),
                                 "logging.timing_csv": str(tmp_path / "t.csv"), "logging.events_jsonl": None,
                                 "zones.file": str(tmp_path / "none.json")})
    system = build_system(cfg, detector=MockDetector({}), llm_client=MockClient())
    try:
        assert system.agent.policy.constrain_enum is False and system.agent.retreat.n_vlm_frames == 2
    finally:
        system.agent.stop(timeout=5)


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
