"""Provider wire format and retries, against a local fake HTTP server: real `requests` and the real
SDKs go over a real socket, so this fails if an SDK drops or renames an argument. No real API."""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

from farmguard.agent.llm.base import LLMClient, is_transient
from farmguard.agent.llm.providers import (
    AnthropicClient,
    GeminiClient,
    OllamaClient,
    OpenAICompatClient,
    build_client,
)
from farmguard.agent.prompts import policy_schema
from farmguard.config import load_config

ANSWER = {"action": "siren", "reason": "r", "confidence": 0.5}
SCHEMA = policy_schema(["siren", "escalate_farmer"])
FAST = {"max_retries": 0, "backoff_base_s": 0.0}


@pytest.fixture
def server():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            srv.requests.append({"path": self.path, "body": body, "headers": dict(self.headers)})
            status, reply, headers = srv.script.pop(0) if srv.script else (200, srv.ok_reply, {})
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            for k, v in headers.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(json.dumps(reply).encode())

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.requests, srv.script, srv.ok_reply = [], [], {}
    srv.url = f"http://127.0.0.1:{srv.server_port}"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def test_ollama_wire_format(server):
    server.ok_reply = {"message": {"content": json.dumps(ANSWER)}}
    res = OllamaClient("m", host=server.url).generate_json("p", SCHEMA, [b"img"], "sys")
    assert res.data == ANSWER
    req = server.requests[0]
    assert req["path"] == "/api/chat"
    assert req["body"]["format"] == SCHEMA and req["body"]["stream"] is False
    assert req["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert req["body"]["messages"][1]["images"] == ["aW1n"]
    assert req["body"]["options"] == {"temperature": 0.2}


def test_openai_compat_wire_format(server):
    server.ok_reply = {"choices": [{"message": {"content": json.dumps(ANSWER)}}]}
    client = OpenAICompatClient("m", base_url=server.url, api_key="k", temperature=None)
    assert client.generate_json("p", SCHEMA, [b"img"]).data == ANSWER
    req = server.requests[0]
    fmt = req["body"]["response_format"]
    assert req["path"] == "/chat/completions" and req["headers"]["Authorization"] == "Bearer k"
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"]["additionalProperties"] is False
    assert set(fmt["json_schema"]["schema"]["required"]) == set(fmt["json_schema"]["schema"]["properties"])
    assert "temperature" not in req["body"]
    assert req["body"]["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_anthropic_wire_format(server, monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", server.url)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    server.ok_reply = {"id": "m", "type": "message", "role": "assistant", "model": "m", "stop_reason": "end_turn",
                       "content": [{"type": "thinking", "thinking": "", "signature": "s"},
                                   {"type": "text", "text": json.dumps(ANSWER)}],
                       "usage": {"input_tokens": 1, "output_tokens": 1}}
    res = AnthropicClient("m", **FAST).generate_json("p", SCHEMA, [b"img"], "sys")
    assert res.data == ANSWER and res.error is None
    body = server.requests[0]["body"]
    assert "temperature" not in body and "tool_choice" not in body and body["system"] == "sys"
    out_schema = body["output_config"]["format"]["schema"]
    assert out_schema["properties"]["action"]["enum"] == ["siren", "escalate_farmer"]
    assert "minimum" not in json.dumps(out_schema) and "maximum" not in json.dumps(out_schema)
    assert body["messages"][0]["content"][0]["source"]["data"] == "aW1n"


def test_anthropic_refusal_is_a_failure_not_an_answer(server, monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", server.url)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    server.ok_reply = {"id": "m", "type": "message", "role": "assistant", "model": "m", "stop_reason": "refusal",
                       "content": [], "usage": {"input_tokens": 1, "output_tokens": 0}}
    res = AnthropicClient("m", max_retries=2, backoff_base_s=0.0).generate_json("p", SCHEMA)
    assert res.data is None and "refused" in res.error and len(server.requests) == 1


def test_sdk_retries_are_off_so_our_logged_attempts_are_real(server, monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", server.url)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    server.script = [(500, {"error": "x"}, {})] * 5
    assert AnthropicClient("m", **FAST).generate_json("p", SCHEMA).data is None
    assert len(server.requests) == 1


def test_gemini_wire_format(server, monkeypatch):
    genai = pytest.importorskip("google.genai")
    from google.genai import types
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    server.ok_reply = {"candidates": [{"content": {"role": "model", "parts": [{"text": json.dumps(ANSWER)}]}}]}
    client = GeminiClient("m", **FAST, timeout_s=7)
    assert client.client._api_client._http_options.timeout == 7000
    client.client = genai.Client(api_key="k", http_options=types.HttpOptions(base_url=server.url))
    assert client.generate_json("p", SCHEMA, [b"img"], "sys").data == ANSWER
    body = server.requests[0]["body"]
    gen = body["generationConfig"]
    assert gen["responseMimeType"] == "application/json"
    assert gen["responseJsonSchema"]["properties"]["action"]["enum"] == ["siren", "escalate_farmer"]
    assert body["contents"][0]["parts"][0]["inlineData"]["data"] == "aW1n"
    assert body["systemInstruction"]["parts"][0]["text"] == "sys"


def test_retries_then_succeeds_and_logs_latency_per_call(server, caplog):
    ok = {"message": {"content": json.dumps(ANSWER)}}
    server.script = [(429, {"error": "slow down"}, {"Retry-After": "0"}), (503, {"error": "busy"}, {}), (200, ok, {})]
    client = OllamaClient("m", host=server.url, backoff_base_s=0.0)
    with caplog.at_level(logging.INFO, logger="farmguard.agent.llm.base"):
        res = client.generate_json("p", SCHEMA)
    assert res.data == ANSWER and len(server.requests) == 3
    lines = [r.getMessage() for r in caplog.records if "llm_call" in r.getMessage()]
    assert sum("ok=0" in m for m in lines) == 2 and sum("ok=1" in m and "attempt=3" in m for m in lines) == 1
    assert all("latency_s=" in m for m in lines)


def test_permanent_errors_are_not_retried(server):
    server.script = [(404, {"error": "model 'm' not found"}, {})]
    res = OllamaClient("m", host=server.url, backoff_base_s=0.0).generate_json("p", SCHEMA)
    assert res.data is None and res.error and len(server.requests) == 1


def test_network_down_returns_error_result_quickly():
    res = OllamaClient("m", host="http://127.0.0.1:9", timeout_s=2, **FAST).generate_json("p", SCHEMA)
    assert res.data is None and res.error


class Flaky(LLMClient):
    def __init__(self, outcomes, **kw):
        super().__init__("m", **kw)
        self.outcomes, self.n = list(outcomes), 0

    def _call(self, prompt, schema, images, system):
        self.n += 1
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o


def test_budget_stops_retries_and_repair():
    # backoff is at least 100 s after jitter, so 100 s + the 100 s timeout never fits in 150 s
    client = Flaky([TimeoutError(), '{"a": 1}'], timeout_s=100, max_total_s=150, backoff_base_s=200.0)
    assert client.generate_json("p", {}).data is None and client.n == 1
    client = Flaky(["not json", '{"a": 1}'], timeout_s=100, max_total_s=99)
    res = client.generate_json("p", {})
    assert res.error == "invalid_json" and client.n == 1  # no budget left for the repair call


def test_repair_still_runs_when_budget_allows():
    client = Flaky(["not json", '{"a": 1}'])
    assert client.generate_json("p", {}).data == {"a": 1} and client.n == 2


def test_is_transient_classification():
    resp = requests.Response()
    resp.status_code = 500
    assert is_transient(requests.HTTPError(response=resp))
    resp.status_code = 401
    assert not is_transient(requests.HTTPError(response=resp))
    assert is_transient(requests.ConnectionError()) and is_transient(requests.Timeout())
    assert not is_transient(KeyError("message")) and not is_transient(RuntimeError("model refused"))


def test_every_provider_gets_its_own_model_id():
    cfg = load_config(overrides={"agent.provider": "anthropic"})
    assert build_client(cfg, "vlm").model == cfg.get("agent.models.anthropic.vlm")
    assert build_client(load_config(overrides={"agent.provider": "ollama"}), "llm").model == cfg.get("agent.llm_model")
