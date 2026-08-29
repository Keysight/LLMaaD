"""
agents/defense_agent.py — DefenseAgent

Manages the CMPE defense side: reshaping models, misdirection algorithms,
and producing the final misdirected response.  Asks EndpointAgent for the
active reshaping endpoint before every run.

Registered task types
---------------------
reshape         payload: {prompt, harmful_sentences, algo, run_dir}  → {result, artifacts, scores}
get_algo_info   payload: {algo}                                       → {name, description}
list_algos      payload: {}                                           → {algos: list}
"""

from __future__ import annotations
import time
from dataclasses import asdict

from .base import BaseAgent, AgentTask, AgentResult
from prompt_reshaping.artifacts.traces import (
    PromptInput, JudgeScores, ReshapingModelConfig,
)
from prompt_reshaping.multi.runner import MultiTryRunner

ALGO_DESCRIPTIONS = {
    "algo1":  "jumble → expand → context inject → compress → harmful inject",
    "algo1q": "algo1 + follow-up question appended to misdirected output",
    "algo2":  "abliterated model generates detailed but wrong answer (responseland)",
}

DEFAULT_ALIAS = "abliterated"


class DefenseAgent(BaseAgent):
    name = "defense"

    def __init__(
        self,
        expansion_words: int = 400,
        compression_words: int = 200,
    ) -> None:
        super().__init__()
        self._expansion_words   = expansion_words
        self._compression_words = compression_words
        self._runners: dict[str, MultiTryRunner] = {}

    # ── Runner cache (one per algo) ────────────────────────────────────────
    def _get_runner(self, algo: str) -> MultiTryRunner:
        if algo not in self._runners:
            self._runners[algo] = MultiTryRunner(
                algo_name=algo,
                expansion_words=self._expansion_words,
                compression_words=self._compression_words,
            )
        return self._runners[algo]

    # ── Resolve reshaping endpoint via EndpointAgent ───────────────────────
    def _resolve_endpoint(self, model_alias: str = DEFAULT_ALIAS) -> ReshapingModelConfig:
        if "endpoint" in self._peers:
            ep_result = self.dispatch("get_endpoint", {"model_alias": model_alias}, to="endpoint")
            if ep_result.status == "ok" and ep_result.data.get("alive"):
                d = ep_result.data
                return ReshapingModelConfig(
                    backend="vllm",
                    model_name=d["model_name"],
                    ip=d["ip"],
                    port=d["port"],
                )
        # Fallback to defaults if endpoint agent not registered or endpoint down
        return ReshapingModelConfig()

    # ── Task handler ───────────────────────────────────────────────────────
    def handle(self, task: AgentTask) -> AgentResult:
        t0 = time.time()

        # ── reshape ───────────────────────────────────────────────────────
        if task.task_type == "reshape":
            prompt           = task.payload.get("prompt", "")
            harmful_sentences = task.payload.get("harmful_sentences", "")
            algo             = task.payload.get("algo", "algo1")
            run_dir          = task.payload.get("run_dir", "agent_run")
            model_alias      = task.payload.get("model_alias", DEFAULT_ALIAS)
            enable_gcg       = task.payload.get("enable_gcg_judge", True)

            if not prompt:
                return AgentResult.err(self.name, "reshape requires 'prompt'")
            if algo not in ALGO_DESCRIPTIONS:
                return AgentResult.err(self.name,
                    f"Unknown algo '{algo}'. Choose from: {list(ALGO_DESCRIPTIONS)}")

            reshaping_model = self._resolve_endpoint(model_alias)

            prompt_input = PromptInput(
                input_prompt=prompt,
                harmful_sentences=harmful_sentences or prompt,
                reshaping_model=reshaping_model,
                dataset_name="agent",
                run_dir=run_dir,
            )
            judge_scores = JudgeScores(gcg_judge=enable_gcg)

            runner = self._get_runner(algo)
            result = runner.algo.run(prompt_input, judge_scores)

            return AgentResult.ok(self.name, {
                "algo":              algo,
                "misdirected_output": result.final_scores.response,
                "initial_response":   result.initial_scores.response,
                "gcg_score":          result.final_scores.gcg_judge_score,
                "timings":            result.timings,
                "reshaping_model":    reshaping_model.to_metadata(),
            }, time.time() - t0)

        # ── get_algo_info ─────────────────────────────────────────────────
        if task.task_type == "get_algo_info":
            algo = task.payload.get("algo", "")
            if algo not in ALGO_DESCRIPTIONS:
                return AgentResult.err(self.name, f"Unknown algo '{algo}'")
            return AgentResult.ok(self.name,
                                  {"name": algo, "description": ALGO_DESCRIPTIONS[algo]},
                                  time.time() - t0)

        # ── list_algos ────────────────────────────────────────────────────
        if task.task_type == "list_algos":
            algos = [{"name": k, "description": v} for k, v in ALGO_DESCRIPTIONS.items()]
            return AgentResult.ok(self.name, {"algos": algos}, time.time() - t0)

        return AgentResult.err(self.name, f"Unknown task_type '{task.task_type}'")
