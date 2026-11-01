"""Few-shot examples in the VLM prompt: on/off flag, ordering, labels, no location leak."""
from __future__ import annotations

import cv2
import numpy as np

from farmguard.agent.llm.providers import MockClient
from farmguard.agent.vlm_check import VLMChecker
from farmguard.config import load_config


def _cfg(tmp_path, enabled=True, per=3):
    return load_config(overrides={"agent.fewshot.enabled": enabled, "agent.fewshot.dir": str(tmp_path),
                                  "agent.fewshot.per_species": per})


def _examples(tmp_path, cat, n):
    (tmp_path / cat).mkdir(parents=True)
    for i in range(n):
        cv2.imwrite(str(tmp_path / cat / f"Nashik_farm_{i}.jpg"), np.zeros((800, 800, 3), np.uint8))


def test_off_by_default_sends_only_event_images(tmp_path):
    _examples(tmp_path, "wild_boar", 3)
    client = MockClient()
    VLMChecker(_cfg(tmp_path, enabled=False), client).check([b"event"], "crop", "wild_boar")
    assert client.calls[0]["n_images"] == 1 and "reference" not in client.calls[0]["prompt"]
    assert load_config().get("agent.fewshot.enabled") is False


def test_examples_come_first_and_are_labelled_without_file_names(tmp_path):
    _examples(tmp_path, "wild_boar", 5)
    _examples(tmp_path, "none", 1)
    seen = {}

    class Spy(MockClient):
        def _call(self, prompt, schema, images, system):
            seen["images"] = images
            return super()._call(prompt, schema, images, system)

    client = Spy()
    VLMChecker(_cfg(tmp_path), client).check([b"event"], "crop", "wild_boar")
    prompt = client.calls[0]["prompt"]
    assert len(seen["images"]) == 5 and seen["images"][-1] == b"event"  # 3 boar (capped) + 1 none + event
    assert "1=wild_boar, 2=wild_boar, 3=wild_boar, 4=none" in prompt
    assert "You are given 1 image(s)" in prompt
    assert "nashik" not in prompt.lower()  # file names (which may name the place) never reach the prompt


def test_flag_can_flip_between_e2_conditions(tmp_path):
    _examples(tmp_path, "wild_boar", 3)
    cfg, client = _cfg(tmp_path, enabled=False), MockClient()
    checker = VLMChecker(cfg, client)
    checker.check([b"e"], "crop", None)
    cfg.raw["agent"]["fewshot"]["enabled"] = True
    checker.check([b"e"], "crop", None)
    assert [c["n_images"] for c in client.calls] == [1, 4]


def test_missing_folder_degrades_to_zero_shot(tmp_path):
    client = MockClient()
    report = VLMChecker(_cfg(tmp_path / "nope"), client).check([b"e"], "crop", "wild_boar")
    assert client.calls[0]["n_images"] == 1 and report.error is None


def test_count_is_clamped_in_code():
    client = MockClient(responder=lambda p, s, n: {"species": "monkey", "count": 999, "behaviour": "eating",
                                                   "confidence": "high", "observation": "", "note": ""})
    assert VLMChecker(load_config(), client).check([b"e"], "crop", None).count == 50
