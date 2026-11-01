"""Concrete providers. Only `requests` is required; SDKs are imported lazily.

NOTE for Claude Code: model ids and SDK details change. Verify against current docs:
  - Ollama:   POST /api/chat, `format` accepts a JSON schema (Ollama >= 0.5), `images` = base64 list
  - Gemini:   google-genai SDK, response_mime_type="application/json" (+ response_schema)
  - Anthropic: messages API with a forced tool call (tool_choice={"type": "tool", ...})
  - OpenAI-compatible (vLLM, Groq, OpenAI...): /chat/completions with response_format json_schema
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable

import requests

from farmguard.agent.llm.base import LLMClient, b64


class OllamaClient(LLMClient):
    name = "ollama"

    def __init__(self, model: str, host: str | None = None, **kw):
        super().__init__(model, **kw)
        self.host = (host or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")

    def _call(self, prompt, schema, images, system):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        user: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            user["images"] = [b64(i) for i in images]
        msgs.append(user)
        r = requests.post(f"{self.host}/api/chat", timeout=self.timeout_s, json={
            "model": self.model, "messages": msgs, "stream": False, "format": schema,
            "options": {"temperature": self.temperature}})
        r.raise_for_status()
        return r.json()["message"]["content"]


class GeminiClient(LLMClient):
    name = "gemini"

    def __init__(self, model: str = "gemini-2.5-flash", **kw):
        super().__init__(model, **kw)
        from google import genai  # pip install google-genai
        self._genai = genai
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def _call(self, prompt, schema, images, system):
        from google.genai import types
        parts: list[Any] = [types.Part.from_bytes(data=i, mime_type="image/jpeg") for i in images]
        parts.append(prompt + "\n\nJSON schema to follow:\n" + json.dumps(schema))
        cfg = types.GenerateContentConfig(temperature=self.temperature,
                                          response_mime_type="application/json",
                                          system_instruction=system)
        resp = self.client.models.generate_content(model=self.model, contents=parts, config=cfg)
        return resp.text or ""


class AnthropicClient(LLMClient):
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-5", **kw):
        super().__init__(model, **kw)
        import anthropic  # pip install anthropic
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _call(self, prompt, schema, images, system):
        content: list[dict[str, Any]] = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64(i)}}
            for i in images]
        content.append({"type": "text", "text": prompt})
        tool = {"name": "answer", "description": "Return the answer in this exact structure.",
                "input_schema": schema}
        kwargs: dict[str, Any] = dict(model=self.model, max_tokens=800, temperature=self.temperature,
                                      messages=[{"role": "user", "content": content}],
                                      tools=[tool], tool_choice={"type": "tool", "name": "answer"})
        if system:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        for block in resp.content:
            if getattr(block, "type", "") == "tool_use":
                return json.dumps(block.input)
        return "".join(getattr(b, "text", "") for b in resp.content)


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
        r = requests.post(f"{self.base_url}/chat/completions", timeout=self.timeout_s,
                          headers={"Authorization": f"Bearer {self.api_key}"}, json={
                              "model": self.model, "messages": msgs, "temperature": self.temperature,
                              "response_format": {"type": "json_schema", "json_schema": {
                                  "name": "answer", "schema": schema}}})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


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
    """role = "vlm" (images) or "llm" (text). Both use agent.provider."""
    provider = cfg.get("agent.provider", "ollama")
    model = cfg.get(f"agent.{role}_model") or cfg.get("agent.vlm_model")
    kw = dict(temperature=cfg.get("agent.temperature", 0.2), timeout_s=cfg.get("agent.timeout_s", 120))
    if provider == "ollama":
        return OllamaClient(model, **kw)
    if provider == "gemini":
        return GeminiClient(model, **kw)
    if provider == "anthropic":
        return AnthropicClient(model, **kw)
    if provider == "openai_compat":
        return OpenAICompatClient(model, **kw)
    if provider == "mock":
        return MockClient(**kw)
    raise ValueError(f"Unknown agent.provider: {provider}")
