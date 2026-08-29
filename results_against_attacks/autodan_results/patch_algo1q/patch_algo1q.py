#!/usr/bin/env python3
"""
patch_algo1q.py — Replace algo1 CMPE outputs with algo1q in MI-FP break epochs.

For each S3 claude_judged result file, finds prompts where CMPE fired and the
attacker declared success (MI-FP cases: claude_case starts with 'misdirected_').
Re-runs cmpe_algo1q on the break epoch's raw_response and replaces misdirected_output.

Everything else (scores, epoch counts, S2 results, claude_case) is unchanged.

Counts:
  turbo   n50      : 23 MI-FP
  turbo   n50-100  : 22 MI-FP
  reasoning n50    :  7 MI-FP
  reasoning n50-100:  0 MI-FP   (nothing to do)
  TOTAL            : 52 LLM calls to abliterated

Usage:
    # Dry-run — see what would be patched, no model calls
    .llmaad/bin/python3 results_against_attacks/autodan_results/patch_algo1q/patch_algo1q.py --dry_run

    # Live run — patch all MI-FP break epochs
    .llmaad/bin/python3 results_against_attacks/autodan_results/patch_algo1q/patch_algo1q.py

    # Live run + re-score patched outputs with gemma (fully rigorous)
    .llmaad/bin/python3 results_against_attacks/autodan_results/patch_algo1q/patch_algo1q.py --rescore

    # Replace ALL unsafe epochs per MI-FP prompt (not just break epoch)
    .llmaad/bin/python3 results_against_attacks/autodan_results/patch_algo1q/patch_algo1q.py --all_epochs
"""

import argparse, json, re, sys, time
from copy import deepcopy
from pathlib import Path

REPO        = Path(__file__).resolve().parents[3]
AUTODAN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(AUTODAN_DIR))

from run_turbo_scenarios import (
    VLLMGenerateAdapter,
    TurboScorer,
    cmpe_algo1q,
    TARGET_MODEL_MAP,
)

RUN_RESULTS = AUTODAN_DIR / "run_results"
OUT_DIR     = Path(__file__).resolve().parent

INPUT_FILES = [
    RUN_RESULTS / "autodan_turbo"    / "turbo_use_strategy_S2S3_algo1_n50_claude_judged.json",
    RUN_RESULTS / "autodan_turbo"    / "turbo_use_strategy_S2S3_algo1_n50-100_claude_judged.json",
    RUN_RESULTS / "autodan_reasoning"/ "reasoning_vanilla_S2S3_algo1_n50_claude_judged.json",
    RUN_RESULTS /                      "reasoning_vanilla_S2S3_algo1_n50-100_claude_judged.json",
]

GREEN  = "\033[92m"; RED  = "\033[91m"; YELLOW = "\033[93m"
BOLD   = "\033[1m";  RESET = "\033[0m"


def find_s3_key(results: dict) -> str:
    for k in results:
        if "S3" in k:
            return k
    raise KeyError(f"No S3 key in results. Keys: {list(results.keys())}")


def is_mifp(entry: dict) -> bool:
    return (
        entry.get("attack_succeeded", False)
        and entry.get("claude_case", "").startswith("misdirected_")
    )


def unsafe_epochs(epochs: list) -> list:
    return [ep for ep in epochs if ep.get("was_harmful") and "misdirected_output" in ep]


def break_epochs(epochs: list) -> list:
    candidates = [ep for ep in unsafe_epochs(epochs) if ep.get("turbo_jailbroken")]
    if not candidates:
        # fallback: last unsafe epoch (score must have caused the break)
        candidates = unsafe_epochs(epochs)[-1:]
    return candidates[-1:]  # only the final break epoch


def patch_file(
    path: Path,
    reshaping_llm,
    scorer,
    dry_run: bool,
    rescore: bool,
    all_epochs: bool,
) -> dict:
    with open(path) as f:
        data = deepcopy(json.load(f))

    s3_key   = find_s3_key(data["results"])
    s3_list  = data["results"][s3_key]
    mifp     = [e for e in s3_list if is_mifp(e)]

    print(f"\n{BOLD}{path.name}{RESET}")
    print(f"  S3 key  : {s3_key}")
    print(f"  S3 total: {len(s3_list)}  |  MI-FP: {len(mifp)}")

    if not mifp:
        print(f"  {YELLOW}No MI-FP entries — nothing to patch{RESET}")
        return data

    total_calls = 0
    patched_ok  = 0
    errors      = 0

    for entry in mifp:
        goal = entry["goal"]
        pid  = entry.get("prompt_id", "?")
        cc   = entry.get("claude_case", "")
        eps  = break_epochs(entry.get("epochs", [])) if not all_epochs else unsafe_epochs(entry.get("epochs", []))

        if not eps:
            print(f"  {YELLOW}#{pid} ({cc}): no UNSAFE epoch found — skip{RESET}")
            continue

        ep_nums = [ep["epoch"] + 1 for ep in eps]
        raw_len = len(eps[-1].get("raw_response", ""))

        if dry_run:
            print(f"  [DRY] #{pid} ({cc}) — would patch epoch(s) {ep_nums} | raw_response={raw_len}ch")
            continue

        for ep in eps:
            raw_response = ep.get("raw_response", "")
            ep_label     = ep["epoch"] + 1
            print(f"  #{pid} ({cc}) epoch {ep_label}...", end=" ", flush=True)
            t0 = time.time()
            try:
                new_output = cmpe_algo1q(reshaping_llm, raw_response, goal)
                ep["misdirected_output_original"] = ep["misdirected_output"]
                ep["misdirected_output"]          = new_output
                ep["algo_patched"]                = "algo1q"
                total_calls += 1

                if rescore and scorer:
                    new_score = scorer.score(goal, new_output)
                    ep["turbo_score_original"] = ep.get("turbo_score")
                    ep["turbo_score"]          = round(new_score, 2)
                    print(f"{GREEN}OK{RESET} ({time.time()-t0:.1f}s) score {ep.get('turbo_score_original')}→{new_score:.1f}")
                else:
                    print(f"{GREEN}OK{RESET} ({time.time()-t0:.1f}s)")
                patched_ok += 1

            except Exception as exc:
                print(f"{RED}ERROR: {exc}{RESET}")
                errors += 1

    if not dry_run:
        for entry in s3_list:
            entry["algo"] = "algo1q"

        new_s3_key = s3_key.replace("algo1", "algo1q")
        if new_s3_key != s3_key:
            data["results"][new_s3_key] = data["results"].pop(s3_key)

        data["patch_info"] = {
            "patched_at":    time.strftime("%Y-%m-%dT%H:%M:%S"),
            "original_algo": "algo1",
            "patched_algo":  "algo1q",
            "mifp_count":    len(mifp),
            "epochs_patched": patched_ok,
            "epochs_errored": errors,
            "all_epochs_mode": all_epochs,
            "rescored":       rescore,
        }
        print(f"  {GREEN}Patched {patched_ok} epoch(s), {errors} error(s){RESET}")

    return data


def output_name(path: Path) -> Path:
    stem = path.stem
    stem = stem.replace("_algo1_", "_algo1q_").replace("_algo1.", "_algo1q.")
    if not stem.endswith("_patched"):
        stem += "_patched"
    return OUT_DIR / (stem + ".json")


def main():
    p = argparse.ArgumentParser(description="Patch algo1 MI-FP CMPE outputs with algo1q")
    p.add_argument("--dry_run",    action="store_true", help="Show plan, no model calls")
    p.add_argument("--rescore",    action="store_true", help="Re-score patched outputs with gemma")
    p.add_argument("--all_epochs", action="store_true", help="Replace all UNSAFE epochs, not just break epoch")
    p.add_argument("--reshaping_model", default="abliterated")
    p.add_argument("--scorer_model",    default="gemma")
    args = p.parse_args()

    if args.dry_run:
        reshaping_llm = None
        scorer        = None
        print(f"{YELLOW}[DRY RUN] No model calls{RESET}")
    else:
        rcfg = TARGET_MODEL_MAP[args.reshaping_model]
        reshaping_llm = VLLMGenerateAdapter(rcfg["model"], rcfg["ip"], rcfg["port"])
        print(f"Reshaping: {rcfg['model']} @ {rcfg['ip']}:{rcfg['port']}")

        scorer = None
        if args.rescore:
            scfg       = TARGET_MODEL_MAP[args.scorer_model]
            scorer_llm = VLLMGenerateAdapter(scfg["model"], scfg["ip"], scfg["port"])
            scorer     = TurboScorer(scorer_llm)
            print(f"Scorer:    {scfg['model']} @ {scfg['ip']}:{scfg['port']}")

    mode = "all UNSAFE epochs" if args.all_epochs else "break epoch only"
    print(f"Mode: {mode} | rescore: {args.rescore}")

    grand_total = 0
    for path in INPUT_FILES:
        if not path.exists():
            print(f"{YELLOW}[skip] {path.name} not found{RESET}")
            continue

        data = patch_file(path, reshaping_llm, scorer, args.dry_run, args.rescore, args.all_epochs)

        if not args.dry_run:
            out = output_name(path)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"  Saved → {out.name}")
            grand_total += data.get("patch_info", {}).get("epochs_patched", 0)

    if not args.dry_run:
        print(f"\n{BOLD}{GREEN}Done. Total epochs patched: {grand_total}{RESET}")
    else:
        print(f"\n{BOLD}Dry run complete.{RESET}")


if __name__ == "__main__":
    main()
