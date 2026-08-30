"""
agents/orchestrator.py — OrchestratorAgent

Coordinates the full attack-vs-defense pipeline.  Delegates to:
  • EndpointAgent  — for model availability
  • AttackAgent    — for loading attack results and running judges
  • DefenseAgent   — for CMPE reshaping

Registered task types
---------------------
run_single      payload: {prompt, harmful_sentences, algo, judges, run_dir}
                → {initial_response, misdirected_output, autodan_judge, scores, timings}

run_batch       payload: {entries, algo, judges, run_dir}
                → {results: list, summary: dict}

run_experiment  payload: {attack_results_path, attack_type, algo, judges,
                           range_start, range_end, successful_only, run_dir}
                → {summary: dict, results: list}

status          payload: {}
                → {peers: list, endpoint_status: dict}
"""

from __future__ import annotations
import json, time
from pathlib import Path

from .base import BaseAgent, AgentTask, AgentResult
from prompt_reshaping.artifacts.writer import ResultWriter

OUTPUT_BASE = Path(__file__).resolve().parents[2] / "results_against_attacks"


class OrchestratorAgent(BaseAgent):
    name = "orchestrator"

    def __init__(self) -> None:
        super().__init__()
        self._writer = ResultWriter(base_dir=str(OUTPUT_BASE / "agent_runs"))

    # ── Helpers ────────────────────────────────────────────────────────────
    def _require_peers(self, *names: str) -> str | None:
        missing = [n for n in names if n not in self._peers]
        if missing:
            return f"Missing required peer agents: {missing}"
        return None

    def _endpoint_refresh(self) -> None:
        if "endpoint" in self._peers:
            self.dispatch("refresh", {}, to="endpoint")

    # ── Single pipeline: attack prompt → CMPE → judge ─────────────────────
    def _run_single_pipeline(
        self,
        prompt: str,
        harmful_sentences: str,
        algo: str,
        judges: dict,
        run_dir: str,
        model_alias: str = "abliterated",
    ) -> dict:
        t0 = time.time()

        # 1. Defense: reshape via CMPE
        defense_result = self.dispatch("reshape", {
            "prompt":            prompt,
            "harmful_sentences": harmful_sentences,
            "algo":              algo,
            "run_dir":           run_dir,
            "model_alias":       model_alias,
            "enable_gcg_judge":  judges.get("gcg", True),
        }, to="defense")

        if defense_result.status == "error":
            return {"error": defense_result.error, "prompt": prompt}

        misdirected_output = defense_result.data.get("misdirected_output", "")
        initial_response   = defense_result.data.get("initial_response", "")

        # 2. Attacker judge: would AutoDAN see this as a success?
        autodan_check = self.dispatch("check_autodan_success",
                                      {"response": misdirected_output},
                                      to="attack")
        autodan_sees_success = autodan_check.data.get("jailbroken", False)

        # 3. Full judge scoring (if extra judges configured)
        scores = {}
        if any(judges.get(k) for k in ["pair_gpt", "strong_reject", "nlp", "gptfuzz"]):
            judge_result = self.dispatch("run_judge", {
                "prompt":       prompt,
                "response":     misdirected_output,
                "judges":       judges,
                "model_alias":  model_alias,
            }, to="attack")
            if judge_result.status == "ok":
                scores = judge_result.data

        scores["gcg_score"] = defense_result.data.get("gcg_score")
        scores["autodan_judge_sees_success"] = autodan_sees_success

        return {
            "prompt":             prompt,
            "initial_response":   initial_response,
            "misdirected_output": misdirected_output,
            "autodan_judge":      autodan_sees_success,
            "scores":             scores,
            "timings":            {
                "defense_sec":  defense_result.elapsed,
                "total_sec":    time.time() - t0,
            },
            "reshaping_model": defense_result.data.get("reshaping_model", {}),
        }

    # ── Task handler ───────────────────────────────────────────────────────
    def handle(self, task: AgentTask) -> AgentResult:
        t0 = time.time()

        # ── status ────────────────────────────────────────────────────────
        if task.task_type == "status":
            ep_data = {}
            if "endpoint" in self._peers:
                ep_result = self.dispatch("list_endpoints", {}, to="endpoint")
                ep_data = ep_result.data if ep_result.status == "ok" else {}
            return AgentResult.ok(self.name, {
                "registered_peers": list(self._peers.keys()),
                "endpoints":        ep_data,
            }, time.time() - t0)

        # ── run_single ────────────────────────────────────────────────────
        if task.task_type == "run_single":
            err = self._require_peers("attack", "defense")
            if err:
                return AgentResult.err(self.name, err)

            self._endpoint_refresh()
            result = self._run_single_pipeline(
                prompt=task.payload.get("prompt", ""),
                harmful_sentences=task.payload.get("harmful_sentences", ""),
                algo=task.payload.get("algo", "algo1"),
                judges=task.payload.get("judges", {"gcg": True}),
                run_dir=task.payload.get("run_dir", "agent_run"),
                model_alias=task.payload.get("model_alias", "abliterated"),
            )
            return AgentResult.ok(self.name, result, time.time() - t0)

        # ── run_batch ─────────────────────────────────────────────────────
        if task.task_type == "run_batch":
            err = self._require_peers("attack", "defense")
            if err:
                return AgentResult.err(self.name, err)

            entries    = task.payload.get("entries", [])
            algo       = task.payload.get("algo", "algo1")
            judges     = task.payload.get("judges", {"gcg": True})
            run_dir    = task.payload.get("run_dir", "agent_run")
            model_alias = task.payload.get("model_alias", "abliterated")

            if not entries:
                return AgentResult.err(self.name, "run_batch requires non-empty 'entries'")

            self._endpoint_refresh()
            results = []
            autodan_pass = 0

            for i, entry in enumerate(entries):
                # Build adversarial prompt (goal + suffix if present)
                adv_result = self.dispatch("build_adversarial", {
                    "goal":   entry.get("goal", entry.get("prompt", "")),
                    "suffix": entry.get("final_suffix", ""),
                }, to="attack")
                prompt = adv_result.data.get("prompt", "") if adv_result.status == "ok" else entry.get("goal", "")

                row = self._run_single_pipeline(
                    prompt=prompt,
                    harmful_sentences=entry.get("goal", ""),
                    algo=algo,
                    judges=judges,
                    run_dir=run_dir,
                    model_alias=model_alias,
                )
                row["index"]           = i
                row["original_attack"] = entry.get("is_success", False)
                results.append(row)

                if row.get("autodan_judge"):
                    autodan_pass += 1

                print(f"  [{i+1:03d}/{len(entries)}] AutoDAN judge → "
                      f"{'misdirected (attacker thinks win)' if row.get('autodan_judge') else 'blocked'}")

            n = len(results)
            summary = {
                "algo":            algo,
                "n":               n,
                "autodan_pass_rate": autodan_pass / n if n else 0,
            }
            return AgentResult.ok(self.name, {"results": results, "summary": summary},
                                  time.time() - t0)

        # ── run_experiment ────────────────────────────────────────────────
        if task.task_type == "run_experiment":
            err = self._require_peers("attack", "defense")
            if err:
                return AgentResult.err(self.name, err)

            path         = task.payload.get("attack_results_path", "")
            attack_type  = task.payload.get("attack_type", "autodan")
            algo         = task.payload.get("algo", "algo1")
            judges       = task.payload.get("judges", {"gcg": True})
            run_dir      = task.payload.get("run_dir", "experiment")
            range_start  = task.payload.get("range_start")
            range_end    = task.payload.get("range_end")
            successful_only = task.payload.get("successful_only", False)
            model_alias  = task.payload.get("model_alias", "abliterated")

            # Load attack results via AttackAgent
            load_result = self.dispatch("load_attack_results", {
                "path":        path,
                "attack_type": attack_type,
            }, to="attack")

            if load_result.status == "error":
                return AgentResult.err(self.name, load_result.error)

            entries = load_result.data["entries"]

            # Optionally filter to successful attacks only
            if successful_only:
                entries = [e for e in entries if e.get("is_success")]

            # Apply range slice
            if range_start is not None and range_end is not None:
                entries = entries[range_start:range_end]

            if not entries:
                return AgentResult.err(self.name, "No entries after filtering")

            print(f"\nExperiment: {attack_type} vs CMPE({algo}), n={len(entries)}")

            # Run batch via run_batch task on self
            batch_result = self._dispatch_self("run_batch", {
                "entries":     entries,
                "algo":        algo,
                "judges":      judges,
                "run_dir":     run_dir,
                "model_alias": model_alias,
            })

            if batch_result.status == "error":
                return AgentResult.err(self.name, batch_result.error)

            # Save JSON summary
            summary = batch_result.data.get("summary", {})
            summary["attack_type"]  = attack_type
            summary["source_file"]  = path

            out_dir = OUTPUT_BASE / "agent_runs"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{attack_type}_vs_{algo}_{run_dir}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "summary": summary,
                    "results": batch_result.data.get("results", []),
                }, f, indent=2)

            print(f"\nResults saved to: {out_path}")
            return AgentResult.ok(self.name, {
                "summary":  summary,
                "out_path": str(out_path),
            }, time.time() - t0)

        return AgentResult.err(self.name, f"Unknown task_type '{task.task_type}'")
