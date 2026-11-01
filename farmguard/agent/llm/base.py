"""Provider-neutral LLM / VLM interface.

Every GenAI call in FarmGuard goes through `LLMClient.generate_json`: a prompt, optional JPEG
images, and a JSON schema. Structured output (enum-constrained JSON) is what keeps the model
from inventing deterrents (seedling-agent `choose_method` tool with an enum; AniREP shows what
happens without it: Gemini made up an ultrasonic frequency).

Robustness patterns borrowed from the repos we reviewed:
  - strip ```json fences before parsing (Shoo `server/ai_pipeline.py` `_parse_response`)
  - one repair round-trip when the JSON is invalid (muye `_repair_qwen_decision`)
  - callers merge missing fields with a rule-based fallback (muye `_merge_with_fallback`)
"""
from __future__ import annotations

import base64
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class LLMResult:
    data: dict[str, Any] | None
    raw: str
    latency_s: float
    error: str | None = None
    model: str = ""


def extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.S)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", cleaned, re.S)  # first {...} block
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def b64(img: bytes) -> str:
    return base64.b64encode(img).decode("ascii")


class LLMClient(ABC):
    name = "base"

    def __init__(self, model: str, temperature: float = 0.2, timeout_s: float = 120):
        self.model = model
        self.temperature = temperature
        self.timeout_s = timeout_s

    @abstractmethod
    def _call(self, prompt: str, schema: dict[str, Any], images: list[bytes],
              system: str | None) -> str:
        """Return the model's raw text (expected to be JSON)."""

    def generate_json(self, prompt: str, schema: dict[str, Any], images: list[bytes] | None = None,
                      system: str | None = None, repair: bool = True) -> LLMResult:
        t0 = time.perf_counter()
        images = images or []
        try:
            raw = self._call(prompt, schema, images, system)
        except Exception as exc:  # network down, model missing, quota...
            log.warning("%s call failed: %s", self.name, exc)
            return LLMResult(None, "", time.perf_counter() - t0, error=str(exc), model=self.model)
        data = extract_json(raw)
        if data is None and repair:
            fix_prompt = (prompt + "\n\nYour previous answer was not valid JSON:\n" + raw[:1500] +
                          "\n\nReply again with ONLY a JSON object that matches the schema.")
            try:
                raw = self._call(fix_prompt, schema, images, system)
                data = extract_json(raw)
            except Exception as exc:
                return LLMResult(None, raw, time.perf_counter() - t0, error=str(exc), model=self.model)
        err = None if data is not None else "invalid_json"
        return LLMResult(data, raw, time.perf_counter() - t0, error=err, model=self.model)
