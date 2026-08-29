"""
extract_injections.py

Extracts AgentDojo attack payloads for all (user_task, injection_task) pairs
in a suite and saves them to a JSON file compatible with LLMaaD's AttackAgent.

Usage:
    python extract_injections.py --suite workspace --attack important_instructions \
        --output ../../results_against_attacks/agentdojo_results/workspace_important.json

    python extract_injections.py --suite banking --attack tool_knowledge \
        --model gpt-4o-2024-05-13 \
        --output ../../results_against_attacks/agentdojo_results/banking_tool_knowledge.json

    # Extract all suites / all attacks at once:
    python extract_injections.py --all \
        --output-dir ../../results_against_attacks/agentdojo_results/

Output JSON schema (list of entries):
    {
        "goal":             str,   # injection_task.GOAL
        "final_suffix":     str,   # always ""
        "final_respond":    str,   # always ""
        "is_success":       bool,  # always false (not yet evaluated)
        "attack_type":      str,   # "agentdojo"
        "suite":            str,
        "attack_name":      str,
        "user_task_id":     str,
        "injection_task_id":str,
        "injection_vector": str,   # which env slot was injected
        "injection_string": str,   # full attack payload placed in tool output
        "user_prompt":      str,   # user_task.PROMPT (legitimate task)
    }
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure agentdojo package is importable when running as a script
_AGENTDOJO_SRC = Path(__file__).resolve().parent / "src"
if str(_AGENTDOJO_SRC) not in sys.path:
    sys.path.insert(0, str(_AGENTDOJO_SRC))

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.attacks.attack_registry import ATTACKS
from agentdojo.attacks.baseline_attacks import DirectAttack, IgnorePreviousAttack, InjecAgentAttack  # noqa: F401 register
from agentdojo.attacks.important_instructions_attacks import (  # noqa: F401 register
    ImportantInstructionsAttack,
    ToolKnowledgeAttack,
)
from agentdojo.functions_runtime import EmptyEnv, FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite

BENCHMARK_VERSION = "v1.2.2"

SUPPORTED_SUITES = ["workspace", "banking", "travel", "slack"]
SUPPORTED_ATTACKS = [
    "direct",
    "ignore_previous",
    "injecagent",
    "important_instructions",
    "important_instructions_no_user_name",
    "important_instructions_no_model_name",
    "important_instructions_no_names",
    "tool_knowledge",
]
DEFAULT_MODEL = "gpt-4o-2024-05-13"


class _MockPipeline(BasePipelineElement):
    """Minimal pipeline used only to satisfy attack constructors that need a named pipeline."""

    def __init__(self, model_name: str) -> None:
        self.name = model_name

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: EmptyEnv = EmptyEnv(),
        messages=(),
        extra_args: dict = {},
    ):
        raise NotImplementedError("MockPipeline is only for attack construction")


def extract_suite_attack(
    suite_name: str,
    attack_name: str,
    model: str = DEFAULT_MODEL,
    benchmark_version: str = BENCHMARK_VERSION,
) -> list[dict]:
    """Return a list of injection entries for all (user_task, injection_task) pairs."""
    suite = get_suite(benchmark_version, suite_name)
    pipeline = _MockPipeline(model)

    attack_cls = ATTACKS.get(attack_name)
    if attack_cls is None:
        raise ValueError(f"Unknown attack '{attack_name}'. Available: {list(ATTACKS)}")

    attack = attack_cls(suite, pipeline)

    entries = []
    for user_task_id, user_task in suite.user_tasks.items():
        try:
            candidates = attack.get_injection_candidates(user_task)
        except ValueError:
            # task is not injectable — skip
            continue

        for inj_task_id, inj_task in suite.injection_tasks.items():
            injections = attack.attack(user_task, inj_task)
            for vector_id, injection_string in injections.items():
                entries.append(
                    {
                        "goal": inj_task.GOAL,
                        "final_suffix": "",
                        "final_respond": "",
                        "is_success": False,
                        "attack_type": "agentdojo",
                        "suite": suite_name,
                        "attack_name": attack_name,
                        "benchmark_version": benchmark_version,
                        "model": model,
                        "user_task_id": user_task_id,
                        "injection_task_id": inj_task_id,
                        "injection_vector": vector_id,
                        "injection_string": injection_string,
                        "user_prompt": user_task.PROMPT,
                    }
                )

    print(f"  [{suite_name} / {attack_name}] extracted {len(entries)} injection entries")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract AgentDojo injection payloads as a JSON dataset for LLMaaD CMPE."
    )
    parser.add_argument("--suite", choices=SUPPORTED_SUITES, help="Suite to extract from")
    parser.add_argument("--attack", choices=SUPPORTED_ATTACKS, help="Attack type to use")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name for attack construction (default: gpt-4o-2024-05-13)",
    )
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_combos",
        help="Extract all suite × attack combinations",
    )
    parser.add_argument(
        "--output-dir",
        default="../../results_against_attacks/agentdojo_results",
        help="Output directory when using --all",
    )
    parser.add_argument(
        "--benchmark-version",
        default=BENCHMARK_VERSION,
        help=f"AgentDojo benchmark version (default: {BENCHMARK_VERSION})",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    if args.all_combos:
        out_dir.mkdir(parents=True, exist_ok=True)
        for suite_name in SUPPORTED_SUITES:
            for attack_name in SUPPORTED_ATTACKS:
                try:
                    entries = extract_suite_attack(
                        suite_name, attack_name, args.model, args.benchmark_version
                    )
                except Exception as exc:
                    print(f"  [{suite_name}/{attack_name}] SKIPPED: {exc}")
                    continue
                if not entries:
                    continue
                out_path = out_dir / f"{suite_name}_{attack_name}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(entries, f, indent=2)
                print(f"  Saved → {out_path}")
        print(f"\nDone. Files written to {out_dir}")
        return

    # Single suite / attack
    if not args.suite or not args.attack:
        parser.error("--suite and --attack are required unless --all is specified")

    entries = extract_suite_attack(
        args.suite, args.attack, args.model, args.benchmark_version
    )

    out_path = Path(
        args.output
        or (out_dir / f"{args.suite}_{args.attack}.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    print(f"Saved {len(entries)} entries → {out_path}")


if __name__ == "__main__":
    main()
