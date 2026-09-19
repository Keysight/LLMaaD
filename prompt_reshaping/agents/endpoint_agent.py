"""
agents/endpoint_agent.py — EndpointAgent

Manages vLLM node health and model routing.  Other agents ask this one for
an active endpoint before making model calls.

Registered task types
---------------------
health_check    payload: {ip, port}            → {alive, models, latency_ms}
get_endpoint    payload: {model_alias}         → {ip, port, model_name, alive}
list_endpoints  payload: {}                    → {endpoints: [{alias, ip, port, alive}]}
refresh         payload: {}                    → {results: [{alias, alive}]}
"""

from __future__ import annotations
import time
import requests
from .base import BaseAgent, AgentTask, AgentResult

import os

# Default vLLM node registry — mirrors model_pick.py LOCAL_VLLM_MODEL_PROFILES.
# IPs read from environment variables; fall back to localhost for portability.
DEFAULT_ENDPOINTS = {
    "vicuna": {
        "ip":         os.getenv("VICUNA_IP", "localhost"),
        "port":       int(os.getenv("VICUNA_PORT", 8005)),
        "model_name": "lmsys/vicuna-7b-v1.3",
    },
    "abliterated": {
        "ip":         os.getenv("ABLITERATED_IP", "localhost"),
        "port":       int(os.getenv("ABLITERATED_PORT", 8000)),
        "model_name": "mlabonne/NeuralDaredevil-8B-abliterated",
    },
    "gpt-oss": {
        "ip":         os.getenv("SCORER_IP", "localhost"),
        "port":       int(os.getenv("SCORER_PORT", 8000)),
        "model_name": "openai/gpt-oss-120b",
    },
    "qwen3": {
        "ip":         os.getenv("QWEN3_IP", "localhost"),
        "port":       int(os.getenv("QWEN3_PORT", 8000)),
        "model_name": "Qwen3-32B",
    },
}


class EndpointAgent(BaseAgent):
    name = "endpoint"

    def __init__(self, endpoints: dict = None, timeout_sec: int = 5) -> None:
        super().__init__()
        self._endpoints: dict = {k: dict(v) for k, v in (endpoints or DEFAULT_ENDPOINTS).items()}
        self._timeout = timeout_sec
        self._status: dict = {}  # alias → {alive, latency_ms, last_checked}

    # ── Health probe ───────────────────────────────────────────────────────
    def _probe(self, ip: str, port: int) -> dict:
        url = f"http://{ip}:{port}/v1/models"
        t0 = time.time()
        try:
            r = requests.get(url, timeout=self._timeout)
            latency_ms = round((time.time() - t0) * 1000, 1)
            if r.status_code == 200:
                models = [m["id"] for m in r.json().get("data", [])]
                return {"alive": True, "models": models, "latency_ms": latency_ms}
            return {"alive": False, "models": [], "latency_ms": latency_ms,
                    "error": f"HTTP {r.status_code}"}
        except Exception as e:
            return {"alive": False, "models": [], "latency_ms": -1, "error": str(e)}

    # ── Task handler ───────────────────────────────────────────────────────
    def handle(self, task: AgentTask) -> AgentResult:
        t0 = time.time()

        if task.task_type == "health_check":
            ip   = task.payload.get("ip")
            port = task.payload.get("port", 8000)
            if not ip:
                return AgentResult.err(self.name, "health_check requires 'ip'")
            result = self._probe(ip, port)
            return AgentResult.ok(self.name, result, time.time() - t0)

        if task.task_type == "get_endpoint":
            alias = task.payload.get("model_alias", "")
            ep = self._endpoints.get(alias)
            if not ep:
                return AgentResult.err(self.name, f"Unknown model alias '{alias}'")
            probe = self._probe(ep["ip"], ep["port"])
            self._status[alias] = {**probe, "last_checked": time.time()}
            data = {
                "alias":      alias,
                "ip":         ep["ip"],
                "port":       ep["port"],
                "model_name": ep["model_name"],
                **probe,
            }
            return AgentResult.ok(self.name, data, time.time() - t0)

        if task.task_type == "list_endpoints":
            rows = []
            for alias, ep in self._endpoints.items():
                cached = self._status.get(alias, {})
                rows.append({
                    "alias":      alias,
                    "ip":         ep["ip"],
                    "port":       ep["port"],
                    "model_name": ep["model_name"],
                    "alive":      cached.get("alive"),
                    "latency_ms": cached.get("latency_ms"),
                })
            return AgentResult.ok(self.name, {"endpoints": rows}, time.time() - t0)

        if task.task_type == "refresh":
            results = []
            for alias, ep in self._endpoints.items():
                probe = self._probe(ep["ip"], ep["port"])
                self._status[alias] = {**probe, "last_checked": time.time()}
                results.append({"alias": alias, **probe})
            return AgentResult.ok(self.name, {"results": results}, time.time() - t0)

        return AgentResult.err(self.name, f"Unknown task_type '{task.task_type}'")

    # ── Convenience: register a new endpoint at runtime ───────────────────
    def add_endpoint(self, alias: str, ip: str, port: int, model_name: str) -> None:
        self._endpoints[alias] = {"ip": ip, "port": port, "model_name": model_name}
