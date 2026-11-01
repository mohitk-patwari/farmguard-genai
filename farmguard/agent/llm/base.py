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
import random
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
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)  # first {...} block
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def b64(img: bytes) -> str:
    return base64.b64encode(img).decode("ascii")


def is_transient(exc: Exception) -> bool:
    """Worth retrying? Duck-typed so this file needs no SDK: requests and httpx-based SDKs expose
    the status as .status_code / .response.status_code, google-genai as .code."""
    status = (getattr(exc, "status_code", None) or getattr(exc, "code", None)
              or getattr(getattr(exc, "response", None), "status_code", None))
    if isinstance(status, int):
        return status in (408, 409, 425, 429) or status >= 500
    return any(k in type(exc).__name__.lower() for k in ("timeout", "connect", "chunked"))


def _retry_after(exc: Exception) -> float:
    try:
        return float(exc.response.headers.get("retry-after"))  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError):
        return 0.0


class LLMClient(ABC):
    """Each attempt is bounded by timeout_s; a whole generate_json (retries, backoff, repair) by
    max_total_s: a retry or repair only starts if a further full timeout still fits, so a hung
    server is not retried for minutes while the farmer's alert waits. SDK-level retries are off
    so the attempts and latencies logged here are the real ones."""
    name = "base"

    def __init__(self, model: str, temperature: float | None = 0.2, timeout_s: float = 120,
                 max_retries: int = 2, backoff_base_s: float = 1.0, max_total_s: float = 180):
        self.model = model
        self.temperature = temperature  # None = do not send it (some models reject it)
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.max_total_s = max_total_s

    @abstractmethod
    def _call(self, prompt: str, schema: dict[str, Any], images: list[bytes],
              system: str | None) -> str:
        """Return the model's raw text (expected to be JSON). Must honour self.timeout_s."""

    def _fits(self, t0: float, delay: float = 0.0) -> bool:
        return time.perf_counter() - t0 + delay + self.timeout_s <= self.max_total_s

    def _call_retrying(self, prompt: str, schema: dict[str, Any], images: list[bytes],
                       system: str | None, t0: float) -> str:
        attempt = 0
        while True:
            attempt += 1
            t_call = time.perf_counter()
            try:
                raw = self._call(prompt, schema, images, system)
            except Exception as exc:
                log.warning("llm_call provider=%s model=%s attempt=%d images=%d latency_s=%.3f ok=0 error=%s",
                            self.name, self.model, attempt, len(images), time.perf_counter() - t_call,
                            f"{type(exc).__name__}: {exc}"[:300])
                delay = max(_retry_after(exc), self.backoff_base_s * 2 ** (attempt - 1) * random.uniform(0.5, 1))
                if attempt > self.max_retries or not is_transient(exc) or not self._fits(t0, delay):
                    raise
                time.sleep(delay)
                continue
            log.info("llm_call provider=%s model=%s attempt=%d images=%d latency_s=%.3f ok=1",
                     self.name, self.model, attempt, len(images), time.perf_counter() - t_call)
            return raw

    def generate_json(self, prompt: str, schema: dict[str, Any], images: list[bytes] | None = None,
                      system: str | None = None, repair: bool = True) -> LLMResult:
        t0 = time.perf_counter()
        images = images or []
        try:
            raw = self._call_retrying(prompt, schema, images, system, t0)
        except Exception as exc:  # network down, model missing, quota...
            return LLMResult(None, "", time.perf_counter() - t0, error=str(exc), model=self.model)
        data = extract_json(raw)
        if data is None and repair and self._fits(t0):
            fix_prompt = (prompt + "\n\nYour previous answer was not valid JSON:\n" + raw[:1500] +
                          "\n\nReply again with ONLY a JSON object that matches the schema.")
            try:
                raw = self._call_retrying(fix_prompt, schema, images, system, t0)
                data = extract_json(raw)
            except Exception as exc:
                return LLMResult(None, raw, time.perf_counter() - t0, error=str(exc), model=self.model)
        err = None if data is not None else "invalid_json"
        return LLMResult(data, raw, time.perf_counter() - t0, error=err, model=self.model)
