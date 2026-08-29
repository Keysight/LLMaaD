"""
agents/base.py — BaseAgent + inter-agent messaging protocol

All agents inherit BaseAgent. Cross-compatibility works by registering peers:
    orchestrator.register(attack_agent)
    orchestrator.register(defense_agent)

Any agent can then dispatch tasks to any registered peer:
    result = self.dispatch("run_judge", payload, to="attack")
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict
import time


@dataclass
class AgentTask:
    task_type: str
    payload:   Dict[str, Any] = field(default_factory=dict)
    sender:    str = ""


@dataclass
class AgentResult:
    status:  str            # "ok" | "error"
    data:    Dict[str, Any] = field(default_factory=dict)
    sender:  str = ""
    elapsed: float = 0.0
    error:   str = ""

    @classmethod
    def ok(cls, sender: str, data: dict, elapsed: float = 0.0) -> AgentResult:
        return cls(status="ok", sender=sender, data=data, elapsed=elapsed)

    @classmethod
    def err(cls, sender: str, error: str) -> AgentResult:
        return cls(status="error", sender=sender, error=error)


class BaseAgent:
    name: str = "base"

    def __init__(self) -> None:
        self._peers: Dict[str, BaseAgent] = {}

    # ── Peer registry ──────────────────────────────────────────────────────
    def register(self, agent: BaseAgent) -> None:
        """Register a peer agent so this agent can dispatch tasks to it."""
        self._peers[agent.name] = agent

    def get_peer(self, name: str) -> BaseAgent:
        if name not in self._peers:
            raise KeyError(f"[{self.name}] No peer registered with name '{name}'")
        return self._peers[name]

    # ── Task dispatch ──────────────────────────────────────────────────────
    def dispatch(self, task_type: str, payload: dict, to: str) -> AgentResult:
        """Send a task to a named peer agent and return its result."""
        peer = self.get_peer(to)
        task = AgentTask(task_type=task_type, payload=payload, sender=self.name)
        return peer.handle(task)

    # ── Task handler (override in subclass) ───────────────────────────────
    def handle(self, task: AgentTask) -> AgentResult:
        return AgentResult.err(self.name, f"Unknown task_type '{task.task_type}'")

    def _dispatch_self(self, task_type: str, payload: dict) -> AgentResult:
        """Internal: call own handle() as if receiving a task."""
        return self.handle(AgentTask(task_type=task_type, payload=payload, sender=self.name))
