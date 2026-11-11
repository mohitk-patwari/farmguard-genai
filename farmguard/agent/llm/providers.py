"""Concrete providers. Only `requests` is required; SDKs are imported lazily.

Checked against the docs and installed SDKs on 2026-11-01 (tests/test_agent_providers.py pins the
wire format with a local fake server; not yet run against the live services):
  - Ollama:   POST /api/chat, `format` = JSON schema, `images` = base64 list on the message
  - Gemini:   google-genai 2.34, response_mime_type + response_json_schema; HttpOptions.timeout in ms
  - Anthropic: anthropic 2.3 sends API version 2026-10-21, where `temperature` is a 400 on every model
              (the SDK dropped the argument) and forced tool_choice is a 400 on Opus 5.5 / Sonnet 5.5.
              So: output_config.format, no sampling params. Numeric schema limits are not supported
              there, so they are stripped and enforced in code (vlm_check, policy).
  - OpenAI-compatible (vLLM, Groq, OpenAI...): response_format json_schema with strict=true, which
              needs additionalProperties=false and every field required (prompts.py schemas comply)
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from typing import Any

import requests

from farmguard.agent.llm.base import LLMClient, b64

log = logging.getLogger(__name__)


def _post_json(url: str, body: dict[str, Any], timeout: float,
               headers: dict[str, str] | None = None) -> dict[str, Any]:
    r = requests.post(url, json=body, headers=headers, timeout=timeout)
    if not r.ok:  # raise_for_status drops the body, which holds the reason ("model not found")
        log.warning("POST %s -> HTTP %d: %s", url, r.status_code, r.text[:300])
    r.raise_for_status()
    return r.json()


class OllamaClient(LLMClient):
    name = "ollama"

    def __init__(self, model: str, host: str | None = None, num_ctx: int | None = None,
                 seed: int | None = None, **kw):
        super().__init__(model, **kw)
        self.num_ctx = num_ctx
        self.seed = seed  # same model, prompt, temperature and seed: Ollama repeats its answer
        self.host = (host or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")

    def _call(self, prompt, schema, images, system):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        user: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            user["images"] = [b64(i) for i in images]
        msgs.append(user)
        body: dict[str, Any] = {"model": self.model, "messages": msgs, "stream": False, "format": schema}
        options = {"temperature": self.temperature, "seed": self.seed, "num_ctx": self.num_ctx}
        options = {k: v for k, v in options.items() if v is not None}
        if options:
            body["options"] = options
        data = _post_json(f"{self.host}/api/chat", body, self.timeout_s)
        log.info("ollama prompt_tokens=%s num_ctx=%s images=%d", data.get("prompt_eval_count"), self.num_ctx, len(images))
        return data["message"]["content"]


class GeminiClient(LLMClient):
    name = "gemini"

    def __init__(self, model: str = "gemini-3.8-flash", **kw):
        super().__init__(model, **kw)
        from google import genai  # pip install google-genai
        from google.genai import types
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"),
                                   http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)))

    def _call(self, prompt, schema, images, system):
        from google.genai import types
        parts: list[Any] = [types.Part.from_bytes(data=i, mime_type="image/jpeg") for i in images]
        parts.append(prompt)
        cfg = types.GenerateContentConfig(temperature=self.temperature,
                                          response_mime_type="application/json",
                                          response_json_schema=schema,
                                          system_instruction=system)
        resp = self.client.models.generate_content(model=self.model, contents=parts, config=cfg)
        return resp.text or ""


def _strip_numeric_limits(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {k: _strip_numeric_limits(v) for k, v in schema.items() if k not in ("minimum", "maximum")}
    if isinstance(schema, list):
        return [_strip_numeric_limits(v) for v in schema]
    return schema


class AnthropicClient(LLMClient):
    """No `temperature` (see module docstring): runs at the API default whatever settings.yaml says."""
    name = "anthropic"

    def __init__(self, model: str = "claude-haiku-5-5", **kw):
        super().__init__(model, **kw)
        import anthropic  # pip install anthropic
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"),
                                          timeout=self.timeout_s, max_retries=0)

    def _call(self, prompt, schema, images, system):
        content: list[dict[str, Any]] = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64(i)}}
            for i in images]
        content.append({"type": "text", "text": prompt})
        kwargs: dict[str, Any] = {
            "model": self.model, "max_tokens": 16000,  # thinking shares this limit; don't lowball it
            "messages": [{"role": "user", "content": content}],
            "output_config": {"format": {"type": "json_schema", "schema": _strip_numeric_limits(schema)}}}
        if system:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        # ponytail: no server-side `fallbacks`: a silent model swap would corrupt E2's per-model
        # numbers. A refusal goes to our non-GenAI fallback like any other failed call.
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"model refused: {resp.stop_details}")
        return "".join(b.text for b in resp.content if b.type == "text")


class OpenAICompatClient(LLMClient):
    name = "openai_compat"

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, **kw):
        super().__init__(model, **kw)
        self.base_url = (base_url or os.getenv("OPENAI_COMPAT_BASE_URL") or
                         "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_COMPAT_API_KEY", "")

    def _call(self, prompt, schema, images, system):
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64(i)}"}}
            for i in images]
        content.append({"type": "text", "text": prompt})
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": content}]
        body: dict[str, Any] = {"model": self.model, "messages": msgs,
                                "response_format": {"type": "json_schema", "json_schema": {
                                    "name": "answer", "strict": True, "schema": schema}}}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        data = _post_json(f"{self.base_url}/chat/completions", body, self.timeout_s,
                          {"Authorization": f"Bearer {self.api_key}"})
        return data["choices"][0]["message"]["content"]


class MockClient(LLMClient):
    """Deterministic client for tests, CI and offline demos. `responder(prompt, schema, n_images)`
    returns a dict; the default responder returns schema-valid placeholder answers."""
    name = "mock"

    def __init__(self, model: str = "mock", responder: Callable[[str, dict, int], dict] | None = None,
                 **kw):
        super().__init__(model, **kw)
        self.responder = responder or default_mock_responder
        self.calls: list[dict[str, Any]] = []

    def _call(self, prompt, schema, images, system):
        self.calls.append({"prompt": prompt, "n_images": len(images), "system": system})
        return json.dumps(self.responder(prompt, schema, len(images)))


def default_mock_responder(prompt: str, schema: dict, n_images: int) -> dict:
    """Schema-valid placeholders. Retreat answers 'uncertain' (so the tracker decides) and alert
    answers are empty (so the notifier uses its template): the mock never pretends to see."""
    props = schema.get("properties", {})
    if "status" in props:
        return {"observation": "mock", "status": "uncertain", "animals_visible": 0, "confidence": "low"}
    if "message" in props:
        return {}
    out: dict[str, Any] = {}
    for key, spec in props.items():
        if "enum" in spec:
            out[key] = spec["enum"][0]
        elif spec.get("type") == "integer":
            out[key] = 1
        elif spec.get("type") == "number":
            out[key] = 0.6
        elif spec.get("type") == "array":
            out[key] = []
        else:
            out[key] = "mock"
    return out


def build_client(cfg, role: str = "vlm") -> LLMClient:
    """role = "vlm" (images) or "llm" (text). Both use agent.provider. agent.models.<provider> wins
    over vlm_model / llm_model, so --provider gemini does not send Gemini an Ollama model tag."""
    provider = cfg.get("agent.provider", "ollama")
    model = (cfg.get(f"agent.models.{provider}.{role}") or cfg.get(f"agent.{role}_model")
             or cfg.get("agent.vlm_model"))
    kw = {"temperature": cfg.get("agent.temperature", 0.2), "timeout_s": cfg.get("agent.timeout_s", 120),
          "max_retries": cfg.get("agent.max_retries", 2), "backoff_base_s": cfg.get("agent.backoff_base_s", 1.0),
          "max_total_s": cfg.get("agent.max_total_s", 180)}
    if provider == "ollama":
        return OllamaClient(model, num_ctx=cfg.get("agent.ollama.num_ctx"), seed=cfg.get("agent.seed", 42), **kw)
    if provider == "gemini":
        return GeminiClient(model, **kw)
    if provider == "anthropic":
        return AnthropicClient(model, **kw)
    if provider == "openai_compat":
        return OpenAICompatClient(model, **kw)
    if provider == "mock":
        return MockClient(**kw)
    raise ValueError(f"Unknown agent.provider: {provider}")
